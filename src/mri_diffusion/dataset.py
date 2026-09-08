"""Sample indexing and MRI pair preparation."""

from pathlib import Path

import numpy as np

from mri_diffusion.config import (
    CANONICAL_AXIS_BY_ORIENTATION,
    ORIENTATIONS,
)
from mri_diffusion.data import (
    extract_slice,
    load_canonical_volume,
    select_centered_indices,
    subject_id_from_path,
)
from mri_diffusion.degradation import (
    create_condition,
    degrade_image,
)
from mri_diffusion.preprocessing import (
    normalize_volume_to_float01,
    prepare_hr_reference,
)
from mri_diffusion.splits import SPLIT_NAMES


def find_subject_split(
    subject_id,
    subject_splits,
):
    """Find the single split assigned to one subject."""

    matching_splits = [
        split_name
        for split_name in SPLIT_NAMES
        if subject_id in subject_splits.get(
            split_name,
            [],
        )
    ]

    if len(matching_splits) == 0:
        raise ValueError(
            f"Subject is not assigned to a split: {subject_id}"
        )

    if len(matching_splits) > 1:
        raise ValueError(
            f"Subject appears in multiple splits: {subject_id}"
        )

    return matching_splits[0]


def build_sample_index(
    paths,
    slices_per_orientation,
    random_seed,
    subject_splits=None,
):
    """Create the metadata index for all selected MRI slices."""

    paths = sorted(
        Path(path)
        for path in paths
    )

    if not paths:
        raise ValueError(
            "At least one MRI volume is required"
        )

    samples = []

    for path in paths:
        subject_id = subject_id_from_path(path)

        if subject_splits is None:
            split_name = None
        else:
            split_name = find_subject_split(
                subject_id=subject_id,
                subject_splits=subject_splits,
            )

        volume = load_canonical_volume(path)

        for orientation in ORIENTATIONS:
            volume_axis = CANONICAL_AXIS_BY_ORIENTATION[
                orientation
            ]
            axis_length = volume.shape[volume_axis]

            slice_indices = select_centered_indices(
                axis_length=axis_length,
                count=slices_per_orientation,
            )

            for slice_index in slice_indices:
                sample_number = len(samples)

                samples.append(
                    {
                        "sample_id": (
                            f"sample-{sample_number:06d}"
                        ),
                        "volume_path": path,
                        "volume_name": path.name,
                        "subject_id": subject_id,
                        "split": split_name,
                        "orientation": orientation,
                        "slice_index": slice_index,
                        "degradation_seed": (
                            random_seed + sample_number
                        ),
                    }
                )

    return samples


def prepare_sample(
    sample,
    target_size,
    low_percentile,
    high_percentile,
    scale,
    blur_sigma,
    noise_sigma,
):
    """Load one indexed sample and create its HR, LR, and condition."""

    volume = load_canonical_volume(
        sample["volume_path"]
    )

    normalized_volume = normalize_volume_to_float01(
        volume=volume,
        low_percentile=low_percentile,
        high_percentile=high_percentile,
    )

    normalized_slice = extract_slice(
        volume=normalized_volume,
        orientation=sample["orientation"],
        index=sample["slice_index"],
    )

    hr_image = prepare_hr_reference(
        normalized_slice=normalized_slice,
        target_size=target_size,
    )

    random_generator = np.random.default_rng(
        sample["degradation_seed"]
    )

    lr_image = degrade_image(
        hr_image=hr_image,
        scale=scale,
        blur_sigma=blur_sigma,
        noise_sigma=noise_sigma,
        random_generator=random_generator,
    )

    condition = create_condition(
        lr_image=lr_image,
        target_size=target_size,
    )

    prepared_sample = sample.copy()
    prepared_sample["hr"] = hr_image
    prepared_sample["lr"] = lr_image
    prepared_sample["condition"] = condition

    return prepared_sample