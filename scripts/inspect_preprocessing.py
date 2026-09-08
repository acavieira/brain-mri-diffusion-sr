import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "src"),
)

from mri_diffusion.config import (
    CANONICAL_AXIS_BY_ORIENTATION,
    DATA_DIR,
    HIGH_PERCENTILE,
    LOW_PERCENTILE,
    ORIENTATIONS,
    TARGET_SIZE,
)
from mri_diffusion.data import (
    extract_slice,
    find_t1w_files,
    load_canonical_volume,
)
from mri_diffusion.preprocessing import (
    normalize_volume_to_float01,
    prepare_hr_reference,
)


def main():
    parser = argparse.ArgumentParser(
        description="Inspect normalized and prepared HR slices"
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

    normalized_volume = normalize_volume_to_float01(
        volume=volume,
        low_percentile=LOW_PERCENTILE,
        high_percentile=HIGH_PERCENTILE,
    )

    figure, plot_axes = plt.subplots(
        len(ORIENTATIONS),
        2,
        figsize=(8, 11),
    )

    for row, orientation in enumerate(ORIENTATIONS):
        volume_axis = CANONICAL_AXIS_BY_ORIENTATION[
            orientation
        ]
        center_index = volume.shape[volume_axis] // 2

        normalized_slice = extract_slice(
            volume=normalized_volume,
            orientation=orientation,
            index=center_index,
        )

        hr_reference = prepare_hr_reference(
            normalized_slice=normalized_slice,
            target_size=TARGET_SIZE,
        )

        plot_axes[row, 0].imshow(
            normalized_slice,
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
        )
        plot_axes[row, 0].set_title(
            f"{orientation.capitalize()}\n"
            f"Normalized {normalized_slice.shape}"
        )
        plot_axes[row, 0].axis("off")

        plot_axes[row, 1].imshow(
            hr_reference,
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
        )
        plot_axes[row, 1].set_title(
            f"{orientation.capitalize()}\n"
            f"Prepared {hr_reference.shape}"
        )
        plot_axes[row, 1].axis("off")

        print(
            f"{orientation}: "
            f"{normalized_slice.shape} -> "
            f"{hr_reference.shape}"
        )

    figure.suptitle(input_path.name, fontsize=14)
    figure.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()