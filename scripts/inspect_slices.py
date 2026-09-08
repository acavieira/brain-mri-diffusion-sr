import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mri_diffusion.config import (
    CANONICAL_AXIS_BY_ORIENTATION,
    DATA_DIR,
    ORIENTATIONS,
    SLICES_PER_ORIENTATION,
)
from mri_diffusion.data import (
    extract_slice,
    find_t1w_files,
    load_canonical_volume,
    select_centered_indices,
)


PREVIEW_COUNT = 5


def get_display_limits(image):
    """Return intensity limits for visual inspection."""

    finite_values = image[np.isfinite(image)]

    if finite_values.size == 0:
        return 0.0, 1.0

    lower_limit = np.percentile(finite_values, 1)
    upper_limit = np.percentile(finite_values, 99)

    if lower_limit == upper_limit:
        upper_limit = lower_limit + 1.0

    return lower_limit, upper_limit


def select_preview_indices(selected_indices):
    """Choose representative indices from the selected slice interval."""

    positions = np.linspace(
        0,
        len(selected_indices) - 1,
        PREVIEW_COUNT,
    )

    return [
        selected_indices[round(position)]
        for position in positions
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Display representative slices from each MRI orientation"
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Path to a local T1-weighted NIfTI volume",
    )
    args = parser.parse_args()

    if args.input is None:
        input_path = find_t1w_files(DATA_DIR)[0]
    else:
        input_path = args.input

    volume = load_canonical_volume(input_path)

    figure, plot_axes = plt.subplots(
        len(ORIENTATIONS),
        PREVIEW_COUNT,
        figsize=(15, 9),
    )

    for row, orientation in enumerate(ORIENTATIONS):
        volume_axis = CANONICAL_AXIS_BY_ORIENTATION[orientation]
        axis_length = volume.shape[volume_axis]

        selected_indices = select_centered_indices(
            axis_length=axis_length,
            count=SLICES_PER_ORIENTATION,
        )
        preview_indices = select_preview_indices(selected_indices)

        for column, slice_index in enumerate(preview_indices):
            image = extract_slice(
                volume=volume,
                orientation=orientation,
                index=slice_index,
            )

            lower_limit, upper_limit = get_display_limits(image)

            plot_axis = plot_axes[row, column]
            plot_axis.imshow(
                image,
                cmap="gray",
                vmin=lower_limit,
                vmax=upper_limit,
            )
            plot_axis.set_title(
                f"{orientation.capitalize()}\nSlice {slice_index}"
            )
            plot_axis.axis("off")

    figure.suptitle(input_path.name, fontsize=16)
    figure.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()