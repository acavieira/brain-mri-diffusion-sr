"""Functions for finding and reading UltraCortex MRI volumes."""

import re
from pathlib import Path

import nibabel as nib
import numpy as np

from mri_diffusion.config import CANONICAL_AXIS_BY_ORIENTATION


_SUBJECT_ID_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])sub-[A-Za-z0-9]+"
)

_T1W_SUFFIXES = (
    "_T1w.nii",
    "_T1w.nii.gz",
)


def find_t1w_files(data_dir):
    """Find raw T1-weighted NIfTI files in flat or BIDS folders."""
    data_dir = Path(data_dir)

    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"Data directory does not exist: {data_dir}"
        )

    paths = []

    for path in data_dir.rglob("*"):
        if not path.is_file():
            continue

        if not path.name.endswith(_T1W_SUFFIXES):
            continue

        is_derivative = any(
            part.lower() == "derivatives"
            for part in path.parts
        )

        if is_derivative:
            continue

        paths.append(path)

    if not paths:
        raise FileNotFoundError(
            f"No raw T1w NIfTI volumes found in: {data_dir}"
        )

    return sorted(paths)


def subject_id_from_path(path):
    """Extract a BIDS subject identifier such as sub-0."""
    path = Path(path)
    match = _SUBJECT_ID_PATTERN.search(path.as_posix())

    if match is None:
        raise ValueError(
            f"No subject identifier found in path: {path}"
        )

    return match.group(0)


def load_canonical_volume(path):
    """Load a three-dimensional NIfTI volume in canonical orientation."""
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(
            f"NIfTI volume does not exist: {path}"
        )

    image = nib.load(str(path))
    image = nib.as_closest_canonical(image)

    if len(image.shape) != 3:
        raise ValueError(
            "Expected a three-dimensional NIfTI volume, "
            f"got shape {image.shape}"
        )

    volume = image.get_fdata(dtype=np.float32)

    if not np.isfinite(volume).any():
        raise ValueError(
            f"Volume has no finite intensity values: {path}"
        )

    return volume


def select_centered_indices(axis_length, count):
    """Return exactly the requested number of central indices."""
    if axis_length <= 0:
        raise ValueError("Axis length must be positive")

    if count <= 0:
        raise ValueError("The requested slice count must be positive")

    if count > axis_length:
        raise ValueError(
            "The requested slice count cannot exceed the axis length"
        )

    start = (axis_length - count) // 2
    end = start + count

    return list(range(start, end))


def extract_slice(volume, orientation, index):
    """Extract one consistently oriented 2D slice."""
    if volume.ndim != 3:
        raise ValueError(
            f"Expected a three-dimensional volume, got ndim {volume.ndim}"
        )

    if orientation not in CANONICAL_AXIS_BY_ORIENTATION:
        raise ValueError(
            f"Unknown orientation: {orientation}"
        )

    axis = CANONICAL_AXIS_BY_ORIENTATION[orientation]
    axis_length = volume.shape[axis]

    if index < 0 or index >= axis_length:
        raise IndexError(
            f"Slice index {index} is out of range for "
            f"{orientation} axis with length {axis_length}"
        )

    raw_slice = np.take(
        volume,
        index,
        axis=axis,
    )

    oriented_slice = np.flipud(raw_slice.T)

    return np.ascontiguousarray(
        oriented_slice,
        dtype=np.float32,
    )