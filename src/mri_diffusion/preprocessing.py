"""MRI intensity normalization and HR slice preparation."""

import cv2
import numpy as np


def normalize_volume_to_float01(
    volume,
    low_percentile,
    high_percentile,
):
    """Normalize one complete MRI volume to the interval [0, 1]."""

    values = np.asarray(
        volume,
        dtype=np.float32,
    )

    if values.ndim != 3:
        raise ValueError(
            f"Expected a three-dimensional volume, got shape {values.shape}"
        )

    if not 0 <= low_percentile < high_percentile <= 100:
        raise ValueError(
            "Percentiles must satisfy "
            "0 <= low_percentile < high_percentile <= 100"
        )

    finite_mask = np.isfinite(values)
    finite_values = values[finite_mask]

    if finite_values.size == 0:
        raise ValueError(
            "The volume contains no finite intensity values"
        )

    low_value = float(
        np.percentile(
            finite_values,
            low_percentile,
        )
    )
    high_value = float(
        np.percentile(
            finite_values,
            high_percentile,
        )
    )

    if high_value <= low_value:
        return np.zeros_like(
            values,
            dtype=np.float32,
        )

    normalized = np.zeros_like(
        values,
        dtype=np.float32,
    )

    normalized[finite_mask] = (
        values[finite_mask] - low_value
    ) / (
        high_value - low_value
    )

    np.clip(
        normalized,
        0.0,
        1.0,
        out=normalized,
    )

    return normalized


def resize_with_padding(image, target_size):
    """Resize a 2D slice without distortion and pad it to a square."""

    image = np.asarray(
        image,
        dtype=np.float32,
    )

    if image.ndim != 2:
        raise ValueError(
            f"Expected a two-dimensional slice, got shape {image.shape}"
        )

    if target_size <= 0:
        raise ValueError(
            "Target size must be positive"
        )

    if not np.isfinite(image).all():
        raise ValueError(
            "The slice contains non-finite intensity values"
        )

    height, width = image.shape

    if height == 0 or width == 0:
        raise ValueError(
            "The slice dimensions must be positive"
        )

    scale_factor = min(
        target_size / height,
        target_size / width,
    )

    resized_height = max(
        1,
        round(height * scale_factor),
    )
    resized_width = max(
        1,
        round(width * scale_factor),
    )

    resized = cv2.resize(
        image,
        (resized_width, resized_height),
        interpolation=cv2.INTER_AREA,
    )

    prepared = np.zeros(
        (target_size, target_size),
        dtype=np.float32,
    )

    top = (
        target_size - resized_height
    ) // 2
    left = (
        target_size - resized_width
    ) // 2

    prepared[
        top:top + resized_height,
        left:left + resized_width,
    ] = resized

    return prepared


def prepare_hr_reference(
    normalized_slice,
    target_size,
):
    """Prepare one normalized slice as the HR reference image."""

    prepared = resize_with_padding(
        image=normalized_slice,
        target_size=target_size,
    )

    return np.clip(
        prepared,
        0.0,
        1.0,
    ).astype(
        np.float32,
        copy=False,
    )