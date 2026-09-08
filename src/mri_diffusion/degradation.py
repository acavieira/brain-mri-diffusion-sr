"""Synthetic LR degradation and condition preparation."""

import cv2
import numpy as np


def degrade_image(
    hr_image,
    scale,
    blur_sigma,
    noise_sigma,
    random_generator,
):
    """Create a synthetic LR image from one HR image."""

    hr_image = np.asarray(
        hr_image,
        dtype=np.float32,
    )

    if hr_image.ndim != 2:
        raise ValueError(
            "The HR image must be two-dimensional"
        )

    if not np.isfinite(hr_image).all():
        raise ValueError(
            "The HR image contains non-finite values"
        )

    if scale < 2:
        raise ValueError(
            "Scale must be greater than or equal to 2"
        )

    if blur_sigma < 0:
        raise ValueError(
            "Blur sigma cannot be negative"
        )

    if noise_sigma < 0:
        raise ValueError(
            "Noise sigma cannot be negative"
        )

    if noise_sigma > 0 and random_generator is None:
        raise ValueError(
            "A random generator is required when adding noise"
        )

    degraded = hr_image.copy()

    if blur_sigma > 0:
        degraded = cv2.GaussianBlur(
            degraded,
            (0, 0),
            blur_sigma,
        )

    height, width = degraded.shape

    lr_height = max(
        1,
        round(height / scale),
    )
    lr_width = max(
        1,
        round(width / scale),
    )

    lr_image = cv2.resize(
        degraded,
        (lr_width, lr_height),
        interpolation=cv2.INTER_AREA,
    )

    if noise_sigma > 0:
        noise = random_generator.normal(
            loc=0.0,
            scale=noise_sigma,
            size=lr_image.shape,
        ).astype(
            np.float32
        )

        lr_image = lr_image + noise

    return np.clip(
        lr_image,
        0.0,
        1.0,
    ).astype(
        np.float32,
        copy=False,
    )


def create_condition(
    lr_image,
    target_size,
):
    """Upscale an LR image with Bicubic interpolation."""

    lr_image = np.asarray(
        lr_image,
        dtype=np.float32,
    )

    if lr_image.ndim != 2:
        raise ValueError(
            "The LR image must be two-dimensional"
        )

    if not np.isfinite(lr_image).all():
        raise ValueError(
            "The LR image contains non-finite values"
        )

    if target_size <= 0:
        raise ValueError(
            "Target size must be positive"
        )

    condition = cv2.resize(
        lr_image,
        (target_size, target_size),
        interpolation=cv2.INTER_CUBIC,
    )

    return np.clip(
        condition,
        0.0,
        1.0,
    ).astype(
        np.float32,
        copy=False,
    )