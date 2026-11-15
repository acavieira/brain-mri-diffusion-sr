"""Image metrics, defined the same way as in brain-mri-sr-baselines."""

import numpy as np
from skimage.metrics import structural_similarity


def psnr(hr, pred, mask=None):
    """PSNR for images in [0, 1], optionally only inside a mask."""
    if mask is not None:
        hr, pred = hr[mask], pred[mask]
    mse = float(np.mean((hr - pred) ** 2))
    return float("inf") if mse == 0 else 10.0 * np.log10(1.0 / mse)


def ssim(hr, pred):
    return float(structural_similarity(hr, pred, data_range=1.0))


def compare(hr, pred, brain_threshold):
    """psnr over the whole slice, psnr_brain without the zero padding, ssim and mae."""
    mask = hr > brain_threshold
    return {
        "psnr": psnr(hr, pred),
        "psnr_brain": psnr(hr, pred, mask) if mask.any() else float("nan"),
        "ssim": ssim(hr, pred),
        "mae": float(np.mean(np.abs(hr - pred))),
    }
