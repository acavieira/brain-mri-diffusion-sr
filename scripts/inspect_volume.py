"""Inspect one local UltraCortex volume without modifying it."""

import argparse
import sys
from pathlib import Path

import numpy as np

# TODO: Remove this hack when the project has a packaging configuration
# Temporary solution while the project has no packaging configuration!!!
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


from mri_diffusion.config import (
    CANONICAL_AXIS_BY_ORIENTATION,
    DATA_DIR,
    DATASET_ACCESSION,
    DATASET_NAME,
    DATASET_VERSION,
    ORIENTATIONS,
    SLICES_PER_ORIENTATION,
)
from mri_diffusion.data import (
    find_t1w_files,
    load_canonical_volume,
    select_centered_indices,
    subject_id_from_path,
)


def main():
    parser = argparse.ArgumentParser(
        description="Inspect one brain MRI volume"
    )
    parser.add_argument(
        "--input",
        type=Path,
        help=(
            "Optional NIfTI path. "
            "If omitted, the first T1w volume in data/ is used."
        ),
    )
    args = parser.parse_args()

    if args.input is None:
        input_path = find_t1w_files(DATA_DIR)[0]
    else:
        input_path = args.input

    volume = load_canonical_volume(input_path)
    finite_values = volume[np.isfinite(volume)]

    print(
        f"Dataset: {DATASET_NAME} "
        f"({DATASET_ACCESSION}, version {DATASET_VERSION})"
    )
    print(f"File: {input_path}")
    print(f"Subject: {subject_id_from_path(input_path)}")
    print(f"Canonical shape: {volume.shape}")
    print(f"Data type: {volume.dtype}")
    print(
        "Finite intensity range: "
        f"{finite_values.min():.3f} to {finite_values.max():.3f}"
    )

    for orientation in ORIENTATIONS:
        axis = CANONICAL_AXIS_BY_ORIENTATION[orientation]
        axis_length = volume.shape[axis]

        indices = select_centered_indices(
            axis_length,
            SLICES_PER_ORIENTATION,
        )

        print(
            f"{orientation}: {len(indices)} slices "
            f"({indices[0]} to {indices[-1]})"
        )


if __name__ == "__main__":
    main()