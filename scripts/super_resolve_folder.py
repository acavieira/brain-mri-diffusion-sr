"""Give twice the resolution to the small images of another 2D dataset.

Meant for the Kaggle tumour images that the classifiers of
brain-mri-few-shot-learning use, but any folder of PNG or JPG images works.
Images whose longest side is below SR_TARGET_SIZE, the size the classifier
will use, are enlarged x2 with the trained model and, for comparison, with
bicubic interpolation. Larger images already have enough detail and are copied
unchanged, so the output folders only differ where something was enlarged.

    python scripts/super_resolve_folder.py

Uses best.pt of the run and writes

    SR_OUTPUT_DIR/diffusion/   same subfolders as SR_INPUT_DIR (train/glioma/...)
    SR_OUTPUT_DIR/bicubic/
    SR_OUTPUT_DIR/log.csv      what happened to each image

Images that are already done are skipped, so after a cut Colab session the
script can simply be run again.
"""

import csv
import shutil
import sys
import zlib
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mri_diffusion import config
from mri_diffusion.data import to_image, to_model
from mri_diffusion.diffusion import Diffusion
from mri_diffusion.utils import build_model, get_device

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
METHODS = ("diffusion", "bicubic")
# the U-Net halves the image once per level after the first, and the skip
# connections only line up when both sides divide by 2 that many times
MULTIPLE = 2 ** (len(config.CHANNEL_MULTS) - 1)


def find_images(root):
    return sorted(p for p in Path(root).rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def load_gray(path):
    """8-bit image as grayscale float32 in [0, 1].

    Plain division by 255 instead of the percentile scaling used for the NIfTI
    volumes, so that bright structures such as enhancing tumours keep their
    values and the output can be written back as 8-bit without clipping them.
    """
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"cannot read {path}")
    return image.astype(np.float32) / 255.0


def needs_sr(shape, target=config.SR_TARGET_SIZE):
    return max(shape) < target


def pad_to_multiple(image, multiple=MULTIPLE):
    """Zero padding at the bottom and on the right so both sides divide by `multiple`."""
    height, width = image.shape
    return np.pad(image, ((0, (-height) % multiple), (0, (-width) % multiple)))


def super_resolve(model, diffusion, images, device, amp=False, generator=None,
                  samples=config.SR_SAMPLES, steps=config.SR_SAMPLING_STEPS):
    """x2 with bicubic and with the model for a list of images of the same size.

    The bicubic image is also the condition of the model, as in training.
    """
    height, width = images[0].shape
    bicubic = [np.clip(cv2.resize(im, (2 * width, 2 * height), interpolation=cv2.INTER_CUBIC), 0.0, 1.0)
               for im in images]
    condition = torch.stack([to_model(pad_to_multiple(b)) for b in bicubic]).to(device)
    total = 0.0
    for _ in range(samples):
        total = total + diffusion.sample(model, condition, steps=steps, generator=generator, amp=amp).float()
    mean = total / samples
    diffusion_images = [to_image(x)[:2 * height, :2 * width] for x in mean]
    return bicubic, diffusion_images


def save_png(image, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.round(image * 255.0).astype(np.uint8))


def is_done(base, suffix):
    """True when the image is in every output folder, either enlarged (.png) or copied."""
    return all((config.SR_OUTPUT_DIR / m / base).with_suffix(".png").exists()
               or (config.SR_OUTPUT_DIR / m / base).with_suffix(suffix).exists() for m in METHODS)


def append_log(rows):
    log = config.SR_OUTPUT_DIR / "log.csv"
    new = not log.exists()
    with open(log, "a", newline="") as f:
        writer = csv.writer(f)
        if new:
            writer.writerow(["image", "height", "width", "action"])
        writer.writerows(rows)


def main():
    src, dst = config.SR_INPUT_DIR, config.SR_OUTPUT_DIR
    paths = find_images(src)
    if not paths:
        raise FileNotFoundError(f"no images in {src}")
    dst.mkdir(parents=True, exist_ok=True)

    # first pass, copy the large images and group the small ones by size
    groups, copied, skipped = defaultdict(list), [], 0
    for path in paths:
        rel = path.relative_to(src)
        if is_done(rel, path.suffix):
            skipped += 1
            continue
        image = load_gray(path)
        if needs_sr(image.shape):
            groups[image.shape].append((rel, image))
        else:
            for m in METHODS:
                out = dst / m / rel
                out.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, out)
            copied.append([str(rel), *image.shape, "copied"])
    if copied:
        append_log(copied)
    todo = sum(len(items) for items in groups.values())
    print(f"{len(paths)} images, {skipped} already done, {len(copied)} copied, {todo} to super-resolve")
    if todo == 0:
        return

    device = get_device()
    amp = config.USE_AMP and device.type == "cuda"
    state = torch.load(config.RUNS_DIR / config.RUN_NAME / "best.pt", map_location=device)
    model = build_model().to(device)
    model.load_state_dict(state["ema"])
    model.eval()
    diffusion = Diffusion()
    print(f"checkpoint from step {state['step']}, {config.SR_SAMPLES} sample(s) of "
          f"{config.SR_SAMPLING_STEPS} steps per image, device {device.type}")

    done = 0
    for shape, items in sorted(groups.items()):
        for start in range(0, len(items), config.SR_BATCH_SIZE):
            batch = items[start:start + config.SR_BATCH_SIZE]
            # seeded from the file name, so an image gets the same result in any run
            generator = torch.Generator().manual_seed(zlib.crc32(str(batch[0][0]).encode()))
            bicubic, sr = super_resolve(model, diffusion, [im for _, im in batch], device, amp, generator)
            rows = []
            for (rel, _), b, s in zip(batch, bicubic, sr):
                save_png(b, (dst / "bicubic" / rel).with_suffix(".png"))
                save_png(s, (dst / "diffusion" / rel).with_suffix(".png"))
                rows.append([str(rel), *shape, "x2"])
            append_log(rows)
            done += len(batch)
            print(f"{done}/{todo} super-resolved", flush=True)


if __name__ == "__main__":
    main()
