import sys
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "src"),
)

from mri_diffusion.config import (
    BLUR_SIGMA,
    DATA_DIR,
    HIGH_PERCENTILE,
    LOW_PERCENTILE,
    NOISE_SIGMA,
    RANDOM_SEED,
    SCALE,
    SLICES_PER_ORIENTATION,
    TARGET_SIZE,
)
from mri_diffusion.data import find_t1w_files
from mri_diffusion.dataset import (
    build_sample_index,
    prepare_sample,
)


def main():
    paths = find_t1w_files(DATA_DIR)

    samples = build_sample_index(
        paths=paths,
        slices_per_orientation=SLICES_PER_ORIENTATION,
        random_seed=RANDOM_SEED,
    )

    selected_index = len(samples) // 2
    selected_sample = samples[selected_index]

    prepared_sample = prepare_sample(
        sample=selected_sample,
        target_size=TARGET_SIZE,
        low_percentile=LOW_PERCENTILE,
        high_percentile=HIGH_PERCENTILE,
        scale=SCALE,
        blur_sigma=BLUR_SIGMA,
        noise_sigma=NOISE_SIGMA,
    )

    print(f"Volumes: {len(paths)}")
    print(f"Samples: {len(samples)}")
    print(
        f"Selected position: {selected_index}"
    )
    print(
        f"Sample ID: {prepared_sample['sample_id']}"
    )
    print(
        f"Subject: {prepared_sample['subject_id']}"
    )
    print(
        f"Split: {prepared_sample['split']}"
    )
    print(
        f"Orientation: {prepared_sample['orientation']}"
    )
    print(
        f"Slice index: {prepared_sample['slice_index']}"
    )
    print(
        f"Degradation seed: "
        f"{prepared_sample['degradation_seed']}"
    )
    print(
        f"HR shape: {prepared_sample['hr'].shape}"
    )
    print(
        f"LR shape: {prepared_sample['lr'].shape}"
    )
    print(
        f"Condition shape: "
        f"{prepared_sample['condition'].shape}"
    )
    print(
        f"HR range: "
        f"{prepared_sample['hr'].min():.3f} to "
        f"{prepared_sample['hr'].max():.3f}"
    )
    print(
        f"LR range: "
        f"{prepared_sample['lr'].min():.3f} to "
        f"{prepared_sample['lr'].max():.3f}"
    )
    print(
        f"Condition range: "
        f"{prepared_sample['condition'].min():.3f} to "
        f"{prepared_sample['condition'].max():.3f}"
    )


if __name__ == "__main__":
    main()