"""Super-resolve the test slices and compare with bicubic and Lanczos.

    python scripts/evaluate.py

Uses best.pt of the run. Every method gets exactly the same LR slice, so the
numbers can be compared slice by slice. Each slice is sampled TEST_SAMPLES
times: "diffusion" is the first sample and "diffusion_mean" the mean of all
of them. The standard deviation across the samples is kept as a map of where
the model is unsure. Writes test/metrics.csv (one row per slice and method),
test/summary.csv, test/sr.npz with the outputs and a few figures in
test/examples.
"""

import csv
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mri_diffusion import config
from mri_diffusion.data import SliceDataset, to_image, to_model
from mri_diffusion.degradation import upscale
from mri_diffusion.diffusion import Diffusion
from mri_diffusion.metrics import compare
from mri_diffusion.utils import build_model, get_device

EXAMPLES = 6


def save_example(path, hr, lr, outputs, spread):
    lr_view = cv2.resize(lr, hr.shape[::-1], interpolation=cv2.INTER_NEAREST)
    best = "diffusion_mean" if "diffusion_mean" in outputs else "diffusion"
    images = [(hr, "HR"), (lr_view, "LR")]
    images += [(image, name) for name, image in outputs.items() if name != "lanczos"]
    images.append((np.abs(outputs[best] - hr), f"|{best} - HR|"))
    if spread is not None:
        images.append((spread, "spread of the samples"))
    fig, axes = plt.subplots(1, len(images), figsize=(3 * len(images), 3.2))
    for ax, (image, title) in zip(axes, images):
        is_map = title.startswith(("|", "spread"))
        ax.imshow(image, cmap="magma" if is_map else "gray", vmin=0, vmax=0.2 if is_map else 1.0)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    device = get_device()
    amp = config.USE_AMP and device.type == "cuda"
    run_dir = config.RUNS_DIR / config.RUN_NAME
    out_dir = run_dir / "test"
    (out_dir / "examples").mkdir(parents=True, exist_ok=True)

    test_set = SliceDataset(config.CACHE_DIR / "test.npz", fixed_noise=True)
    model = build_model().to(device)
    state = torch.load(run_dir / "best.pt", map_location=device)
    model.load_state_dict(state["ema"])
    model.eval()
    diffusion = Diffusion()
    generator = torch.Generator().manual_seed(config.SEED)
    n = config.TEST_SAMPLES
    methods = ["bicubic", "lanczos", "diffusion"] + (["diffusion_mean"] if n > 1 else [])
    print(f"{len(test_set)} test slices, checkpoint from step {state['step']}, "
          f"{n} samples of {config.TEST_SAMPLING_STEPS} sampling steps")

    examples = set(np.linspace(0, len(test_set) - 1, EXAMPLES).round().astype(int))
    rows, saved = [], {"sample": [], "mean": [], "spread": []}
    for start in range(0, len(test_set), config.BATCH_SIZE):
        indices = range(start, min(start + config.BATCH_SIZE, len(test_set)))
        pairs = [test_set.pair(i) for i in indices]
        cond = torch.stack([to_model(condition) for _, _, condition in pairs]).to(device)
        # each sample starts from new noise, the seeded generator keeps runs repeatable
        samples = np.stack([
            [to_image(x) for x in diffusion.sample(model, cond, steps=config.TEST_SAMPLING_STEPS,
                                                   generator=generator, amp=amp)]
            for _ in range(n)
        ])

        for k, i in enumerate(indices):
            hr, lr, bicubic = pairs[k]
            outputs = {
                "bicubic": bicubic,
                "lanczos": upscale(lr, hr.shape[0], cv2.INTER_LANCZOS4),
                "diffusion": samples[0, k],
            }
            spread = None
            saved["sample"].append(samples[0, k])
            if n > 1:
                # one sample has twice the error of the mean of all possible samples
                # (Blau & Michaeli 2018), averaging a few gets part of the way there
                outputs["diffusion_mean"] = samples[:, k].mean(axis=0)
                spread = samples[:, k].std(axis=0)
                saved["mean"].append(outputs["diffusion_mean"])
                saved["spread"].append(spread)
            info = {"subject": test_set.subjects[i], "file": test_set.files[i],
                    "orientation": test_set.orientations[i], "slice": int(test_set.indices[i])}
            for method in methods:
                rows.append({**info, "method": method,
                             **compare(hr, outputs[method], config.BRAIN_THRESHOLD)})
            if i in examples:
                save_example(out_dir / "examples" / f"slice_{i:04d}.png", hr, lr, outputs, spread)
        print(f"{indices[-1] + 1}/{len(test_set)} slices done")

    with open(out_dir / "metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    np.savez_compressed(out_dir / "sr.npz", **{name: np.stack(images).astype(np.float16)
                                               for name, images in saved.items() if images})

    metric_names = ("psnr", "psnr_brain", "ssim", "mae")
    with open(out_dir / "summary.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method"] + [f"{m}_{s}" for m in metric_names for s in ("mean", "std")])
        for method in methods:
            values = {m: [r[m] for r in rows if r["method"] == method] for m in metric_names}
            # nan versions because a slice without brain has no psnr_brain
            writer.writerow([method] + [f"{fn(values[m]):.4f}" for m in metric_names
                                        for fn in (np.nanmean, np.nanstd)])
            print(f"{method:14s} psnr {np.nanmean(values['psnr']):.2f}  "
                  f"psnr_brain {np.nanmean(values['psnr_brain']):.2f}  "
                  f"ssim {np.nanmean(values['ssim']):.3f}")


if __name__ == "__main__":
    main()
