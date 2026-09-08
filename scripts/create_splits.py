import argparse
import sys
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "src"),
)

from mri_diffusion.config import (
    DATA_DIR,
    RANDOM_SEED,
    SPLIT_FILE,
    TEST_FRACTION,
    TRAIN_FRACTION,
    VALIDATION_FRACTION,
)
from mri_diffusion.data import find_t1w_files
from mri_diffusion.splits import (
    SPLIT_NAMES,
    collect_subject_ids,
    create_subject_splits,
    write_subject_splits,
)


def main():
    parser = argparse.ArgumentParser(
        description="Create deterministic subject-level dataset splits"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="Directory containing the local MRI dataset",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=SPLIT_FILE,
        help="Output CSV file",
    )
    args = parser.parse_args()

    paths = find_t1w_files(args.data_dir)
    subject_ids = collect_subject_ids(paths)

    try:
        subject_splits = create_subject_splits(
            subject_ids=subject_ids,
            train_fraction=TRAIN_FRACTION,
            validation_fraction=VALIDATION_FRACTION,
            test_fraction=TEST_FRACTION,
            random_seed=RANDOM_SEED,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error

    write_subject_splits(
        subject_splits=subject_splits,
        output_path=args.output,
    )

    print(f"T1w volumes found: {len(paths)}")
    print(f"Unique subjects found: {len(subject_ids)}")

    for split_name in SPLIT_NAMES:
        split_count = len(subject_splits[split_name])
        print(f"{split_name}: {split_count} subjects")

    print(f"Split file: {args.output}")


if __name__ == "__main__":
    main()