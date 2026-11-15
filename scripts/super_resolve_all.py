"""Pass every image of another 2D dataset through the model at the size of the classifier.

Meant for the Kaggle tumour images of brain-mri-few-shot-learning. Each image is
resized to IMAGE_SIZE x IMAGE_SIZE, which is its HR, degraded exactly like the
UltraCortex slices (blur, x2 reduction and noise), which is its LR, and enlarged
back with bicubic interpolation and with the trained model (the mean of
ALL_SAMPLES samples). Every image and every class goes through the same steps,
and the HR gives PSNR and SSIM for each image.

    python scripts/super_resolve_all.py

Uses best.pt of the run and writes, with the subfolders of SR_INPUT_DIR (train/glioma/...)

    ALL_OUTPUT_DIR/hr/          256 x 256
    ALL_OUTPUT_DIR/lr/          128 x 128
    ALL_OUTPUT_DIR/bicubic/     256 x 256
    ALL_OUTPUT_DIR/diffusion/   256 x 256
    ALL_OUTPUT_DIR/metrics.csv  psnr, psnr_brain, ssim, ssim_brain and mae against the HR

An image with a side below 256 pixels has no real detail at that size, its HR is
already enlarged, so metrics.csv marks it with real_hr = 0. Images that are done
are skipped, so after a cut Colab session the script can simply be run again.
"""

import csv
import sys
import zlib
from pathlib import Path

import cv2
import numpy as np
import torch
from skimage.metrics import structural_similarity

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mri_diffusion import config
from mri_diffusion.data import to_image, to_model
from mri_diffusion.degradation import degrade, make_condition
from mri_diffusion.diffusion import Diffusion
from mri_diffusion.metrics import compare
from mri_diffusion.utils import build_model, get_device

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
VERSIONS = ("hr", "lr", "bicubic", "diffusion")
METHODS = ("bicubic", "diffusion")
COLUMNS = ["image", "height", "width", "real_hr", "method", "psnr", "psnr_brain", "ssim", "ssim_brain", "mae"]


def find_images(root):
    return sorted(p for p in Path(root).rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def load_gray(path):
    """8-bit image as grayscale float32 in [0, 1], as in super_resolve_folder.py."""
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"cannot read {path}")
    return image.astype(np.float32) / 255.0


def to_hr(image, size=config.IMAGE_SIZE):
    """Square size x size image, like the Resize((size, size)) of the classifier.

    INTER_AREA when both sides shrink, bicubic when a side has to grow.
    """
    height, width = image.shape
    method = cv2.INTER_AREA if min(height, width) >= size else cv2.INTER_CUBIC
    return np.clip(cv2.resize(image, (size, size), interpolation=method), 0.0, 1.0).astype(np.float32)


def seed_of(rel):
    """Seed from the file name, so an image gets the same LR and samples in any run."""
    return zlib.crc32(str(rel).encode())


def make_lr(hr, rel):
    """LR with the degradation of the UltraCortex slices."""
    return degrade(hr, np.random.default_rng(seed_of(rel))).astype(np.float32)


def diffusion_mean(model, diffusion, bicubic, device, amp=False, generator=None,
                   samples=config.ALL_SAMPLES, steps=config.ALL_SAMPLING_STEPS):
    """Mean of `samples` samples for a batch of images, with their bicubic images as the condition."""
    condition = torch.stack([to_model(b) for b in bicubic]).to(device)
    total = 0.0
    for _ in range(samples):
        total = total + diffusion.sample(model, condition, steps=steps, generator=generator, amp=amp).float()
    return [to_image(x) for x in total / samples]


def to_8bit(image):
    return np.round(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)


def out_path(root, version, rel):
    return (Path(root) / version / rel).with_suffix(".png")


def is_done(root, rel):
    return all(out_path(root, v, rel).exists() for v in VERSIONS)


def metric_rows(rel, shape, saved):
    """Metrics of bicubic and diffusion against the HR, on the 8-bit images that were saved."""
    hr = saved["hr"].astype(np.float32) / 255.0
    head = hr > config.BRAIN_THRESHOLD
    height, width = shape
    rows = []
    for method in METHODS:
        pred = saved[method].astype(np.float32) / 255.0
        m = compare(hr, pred, config.BRAIN_THRESHOLD)
        _, ssim_map = structural_similarity(hr, pred, data_range=1.0, full=True)
        ssim_brain = float(ssim_map[head].mean()) if head.any() else float("nan")
        rows.append([str(rel), height, width, int(min(height, width) >= config.IMAGE_SIZE), method,
                     m["psnr"], m["psnr_brain"], m["ssim"], ssim_brain, m["mae"]])
    return rows


def append_metrics(path, rows):
    new = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if new:
            writer.writerow(COLUMNS)
        writer.writerows(rows)


def fill_missing_metrics(src, dst, metrics_path, done_paths):
    """Metrics for images that were saved but cut off before their row was written."""
    have = set()
    if metrics_path.exists():
        with open(metrics_path, newline="") as f:
            have = {row["image"] for row in csv.DictReader(f)}
    rows = []
    for path in done_paths:
        rel = path.relative_to(src)
        if str(rel) not in have:
            saved = {v: cv2.imread(str(out_path(dst, v, rel)), cv2.IMREAD_GRAYSCALE) for v in VERSIONS}
            rows += metric_rows(rel, load_gray(path).shape, saved)
    if rows:
        append_metrics(metrics_path, rows)


def main():
    src, dst = config.SR_INPUT_DIR, config.ALL_OUTPUT_DIR
    metrics_path = dst / "metrics.csv"
    paths = find_images(src)
    done = {p for p in paths if is_done(dst, p.relative_to(src))}
    todo = [p for p in paths if p not in done]
    print(f"{len(paths)} images, {len(done)} already done, {len(todo)} to do")
    fill_missing_metrics(src, dst, metrics_path, sorted(done))
    if not todo:
        return

    device = get_device()
    amp = config.USE_AMP and device.type == "cuda"
    state = torch.load(config.RUNS_DIR / config.RUN_NAME / "best.pt", map_location=device)
    model = build_model().to(device)
    model.load_state_dict(state["ema"])
    model.eval()
    diffusion = Diffusion()
    print(f"checkpoint from step {state['step']}, {config.ALL_SAMPLES} sample(s) of "
          f"{config.ALL_SAMPLING_STEPS} steps per image, device {device.type}")

    finished = 0
    for start in range(0, len(todo), config.SR_BATCH_SIZE):
        batch = todo[start:start + config.SR_BATCH_SIZE]
        rels = [p.relative_to(src) for p in batch]
        originals = [load_gray(p) for p in batch]
        hrs = [to_hr(image) for image in originals]
        lrs = [make_lr(hr, rel) for hr, rel in zip(hrs, rels)]
        bicubic = [make_condition(lr, config.IMAGE_SIZE) for lr in lrs]
        generator = torch.Generator().manual_seed(seed_of(rels[0]))
        diffused = diffusion_mean(model, diffusion, bicubic, device, amp, generator)

        rows = []
        for rel, original, images in zip(rels, originals, zip(hrs, lrs, bicubic, diffused)):
            saved = {v: to_8bit(image) for v, image in zip(VERSIONS, images)}
            for v, image in saved.items():
                path = out_path(dst, v, rel)
                path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(path), image)
            rows += metric_rows(rel, original.shape, saved)
        append_metrics(metrics_path, rows)
        finished += len(batch)
        print(f"{finished}/{len(todo)} done", flush=True)


if __name__ == "__main__":
    main()
