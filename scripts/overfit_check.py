"""Overfit a few slices to check that the pipeline can learn at all.

    python scripts/overfit_check.py

Takes 8 training slices at 64x64 and a small U-Net. The loss should go down
and the PSNR of the sampled slices should pass the bicubic value printed at
the start, on real UltraCortex slices that took 500 to 750 steps. Passing
only says that nothing is broken, it says nothing about generalisation.
Takes a few minutes on a laptop.
"""

import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mri_diffusion import config
from mri_diffusion.data import SliceDataset, to_image, to_model
from mri_diffusion.degradation import degrade, make_condition
from mri_diffusion.diffusion import Diffusion
from mri_diffusion.metrics import psnr
from mri_diffusion.model import UNet
from mri_diffusion.utils import get_device, seed_everything

SLICES = 8
SIZE = 64
STEPS = 1500
CHECK_EVERY = 250


def main():
    seed_everything(config.SEED)
    device = get_device()
    dataset = SliceDataset(config.CACHE_DIR / "train.npz", fixed_noise=True)

    picks = np.linspace(0, len(dataset) - 1, SLICES).round().astype(int)
    hr_images = [cv2.resize(dataset.images[i].astype(np.float32), (SIZE, SIZE),
                            interpolation=cv2.INTER_AREA) for i in picks]
    rng = np.random.default_rng(config.SEED)
    conditions = [make_condition(degrade(hr, rng), SIZE) for hr in hr_images]
    hr = torch.stack([to_model(image) for image in hr_images]).to(device)
    cond = torch.stack([to_model(image) for image in conditions]).to(device)

    model = UNet(base=32, mults=(1, 2, 2), res_blocks=1, attention_res=(16,),
                 dropout=0.0, image_size=SIZE).to(device)
    diffusion = Diffusion()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

    bicubic = np.mean([psnr(h, c) for h, c in zip(hr_images, conditions)])
    print(f"bicubic psnr on these slices: {bicubic:.2f} dB")

    for step in range(1, STEPS + 1):
        model.train()
        loss = diffusion.loss(model, hr, cond)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % CHECK_EVERY == 0:
            model.eval()
            sr = diffusion.sample(model, cond, steps=50, generator=torch.Generator().manual_seed(0))
            score = np.mean([psnr(h, to_image(s)) for h, s in zip(hr_images, sr)])
            print(f"step {step:5d}  loss {loss.item():.4f}  diffusion psnr {score:.2f} dB")

    out = config.RUNS_DIR / "overfit_check.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, SLICES, figsize=(2 * SLICES, 6))
    for k in range(SLICES):
        for row, (image, title) in enumerate([(hr_images[k], "HR"), (conditions[k], "bicubic"),
                                              (to_image(sr[k]), "diffusion")]):
            axes[row, k].imshow(image, cmap="gray", vmin=0, vmax=1)
            axes[row, k].set_title(title, fontsize=8)
            axes[row, k].axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
