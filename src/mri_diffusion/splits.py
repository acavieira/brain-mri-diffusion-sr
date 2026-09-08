import csv
import math
import random
from pathlib import Path

from mri_diffusion.data import subject_id_from_path


SPLIT_NAMES = (
    "train",
    "validation",
    "test",
)


def collect_subject_ids(paths):
    """Return the unique subject identifiers found in the file paths."""

    subject_ids = {
        subject_id_from_path(path)
        for path in paths
    }

    return sorted(subject_ids)


def create_subject_splits(
    subject_ids,
    train_fraction,
    validation_fraction,
    test_fraction,
    random_seed,
):
    """Divide subjects into deterministic and non-overlapping groups."""

    unique_subjects = sorted(set(subject_ids))

    if len(unique_subjects) < 3:
        raise ValueError(
            "At least three subjects are required to create "
            "training, validation, and test splits"
        )

    fractions = (
        train_fraction,
        validation_fraction,
        test_fraction,
    )

    if any(fraction <= 0 for fraction in fractions):
        raise ValueError("All split fractions must be positive")

    if not math.isclose(sum(fractions), 1.0):
        raise ValueError("The split fractions must sum to 1.0")

    subject_count = len(unique_subjects)

    validation_count = max(
        1,
        round(subject_count * validation_fraction),
    )
    test_count = max(
        1,
        round(subject_count * test_fraction),
    )
    train_count = (
        subject_count
        - validation_count
        - test_count
    )

    if train_count < 1:
        raise ValueError(
            "The split fractions leave no subjects for training"
        )

    shuffled_subjects = unique_subjects.copy()

    random_generator = random.Random(random_seed)
    random_generator.shuffle(shuffled_subjects)

    train_end = train_count
    validation_end = train_count + validation_count

    subject_splits = {
        "train": sorted(
            shuffled_subjects[:train_end]
        ),
        "validation": sorted(
            shuffled_subjects[train_end:validation_end]
        ),
        "test": sorted(
            shuffled_subjects[validation_end:]
        ),
    }

    return subject_splits


def write_subject_splits(subject_splits, output_path):
    """Write the subject split assignments to a CSV file."""

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["subject_id", "split"])

        for split_name in SPLIT_NAMES:
            if split_name not in subject_splits:
                raise ValueError(
                    f"Missing subject split: {split_name}"
                )

            for subject_id in sorted(
                subject_splits[split_name]
            ):
                writer.writerow(
                    [subject_id, split_name]
                )