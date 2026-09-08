import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "src"),
)

from mri_diffusion.config import (
    BLUR_SIGMA,
    CANONICAL_AXIS_BY_ORIENTATION,
    DATA_DIR,
    HIGH_PERCENTILE,
    LOW_PERCENTILE,
    NOISE_SIGMA,
    ORIENTATIONS,
    RANDOM_SEED,
    SCALE,
    TARGET_SIZE,
)
from mri_diffusion.data import (
    extract_slice,
    find_t1w_files,
    load_canonical_volume,
)
from mri_diffusion.degradation import (
    create_condition,
    degrade_image,
)
from mri_diffusion.preprocessing import (
    normalize_volume_to_float01,
    prepare_hr_reference,
)


def main():
    parser = argparse.ArgumentParser(
        description="Inspect synthetic MRI degradation"
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

    random_generator = np.random.default_rng(
        RANDOM_SEED
    )

    figure, plot_axes = plt.subplots(
        len(ORIENTATIONS),
        4,
        figsize=(14, 10),
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

        hr_image = prepare_hr_reference(
            normalized_slice=normalized_slice,
            target_size=TARGET_SIZE,
        )

        lr_image = degrade_image(
            hr_image=hr_image,
            scale=SCALE,
            blur_sigma=BLUR_SIGMA,
            noise_sigma=NOISE_SIGMA,
            random_generator=random_generator,
        )

        condition = create_condition(
            lr_image=lr_image,
            target_size=TARGET_SIZE,
        )

        absolute_error = np.abs(
            hr_image - condition
        )

        images = (
            hr_image,
            lr_image,
            condition,
            absolute_error,
        )
        titles = (
            f"{orientation.capitalize()} HR\n{hr_image.shape}",
            f"LR\n{lr_image.shape}",
            f"Bicubic condition\n{condition.shape}",
            "Absolute error",
        )
        color_maps = (
            "gray",
            "gray",
            "gray",
            "magma",
        )

        for column in range(4):
            plot_axes[row, column].imshow(
                images[column],
                cmap=color_maps[column],
                vmin=0.0,
                vmax=1.0,
            )
            plot_axes[row, column].set_title(
                titles[column]
            )
            plot_axes[row, column].axis("off")

        print(
            f"{orientation}: "
            f"HR {hr_image.shape}, "
            f"LR {lr_image.shape}, "
            f"condition {condition.shape}"
        )

    figure.suptitle(
        input_path.name,
        fontsize=14,
    )
    figure.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()