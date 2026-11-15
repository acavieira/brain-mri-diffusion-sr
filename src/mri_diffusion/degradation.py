"""Synthetic low-resolution images.

The steps and values match brain-mri-sr-baselines, so the diffusion model and
the interpolation methods are judged on the same kind of input.
"""

import cv2
import numpy as np

from mri_diffusion import config


def degrade(hr, rng, scale=config.SCALE, blur_sigma=config.BLUR_SIGMA,
            noise_sigma=config.NOISE_SIGMA):
    """Blur an HR slice in [0, 1], shrink it by `scale` and add Gaussian noise."""
    image = cv2.GaussianBlur(hr, (0, 0), blur_sigma) if blur_sigma > 0 else hr
    height, width = image.shape
    lr_size = (max(1, round(width / scale)), max(1, round(height / scale)))
    # INTER_AREA averages the pixels that are merged, which is what a
    # coarser acquisition grid does
    lr = cv2.resize(image, lr_size, interpolation=cv2.INTER_AREA)
    if noise_sigma > 0:
        lr = lr + rng.normal(0.0, noise_sigma, size=lr.shape).astype(np.float32)
    return np.clip(lr, 0.0, 1.0).astype(np.float32)


def upscale(lr, size, method=cv2.INTER_CUBIC):
    """Interpolate an LR image back to size x size."""
    resized = cv2.resize(lr, (size, size), interpolation=method)
    return np.clip(resized, 0.0, 1.0).astype(np.float32)


def make_condition(lr, size):
    # SR3 (Saharia et al. 2023) feeds the network the LR image after bicubic
    # upsampling, so it has the same size as the image being generated
    return upscale(lr, size, cv2.INTER_CUBIC)
