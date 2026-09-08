import csv
from pathlib import Path

import pytest

from mri_diffusion.splits import (
    collect_subject_ids,
    create_subject_splits,
    write_subject_splits,
)


def test_collect_subject_ids_groups_multiple_sessions():
    paths = [
        Path(
            "data/sub-01/ses-1/anat/"
            "sub-01_ses-1_T1w.nii"
        ),
        Path(
            "data/sub-01/ses-2/anat/"
            "sub-01_ses-2_T1w.nii"
        ),
        Path(
            "data/sub-02/ses-1/anat/"
            "sub-02_ses-1_T1w.nii"
        ),
    ]

    subject_ids = collect_subject_ids(paths)

    assert subject_ids == [
        "sub-01",
        "sub-02",
    ]


def test_subject_splits_are_deterministic_and_disjoint():
    subject_ids = [
        f"sub-{index:02d}"
        for index in range(20)
    ]

    first_result = create_subject_splits(
        subject_ids=subject_ids,
        train_fraction=0.70,
        validation_fraction=0.15,
        test_fraction=0.15,
        random_seed=23,
    )
    second_result = create_subject_splits(
        subject_ids=subject_ids,
        train_fraction=0.70,
        validation_fraction=0.15,
        test_fraction=0.15,
        random_seed=23,
    )

    assert first_result == second_result

    train_subjects = set(first_result["train"])
    validation_subjects = set(
        first_result["validation"]
    )
    test_subjects = set(first_result["test"])

    assert len(train_subjects) == 14
    assert len(validation_subjects) == 3
    assert len(test_subjects) == 3

    assert train_subjects.isdisjoint(
        validation_subjects
    )
    assert train_subjects.isdisjoint(
        test_subjects
    )
    assert validation_subjects.isdisjoint(
        test_subjects
    )

    all_assigned_subjects = (
        train_subjects
        | validation_subjects
        | test_subjects
    )

    assert all_assigned_subjects == set(subject_ids)


def test_subject_splits_require_three_subjects():
    with pytest.raises(
        ValueError,
        match="At least three subjects",
    ):
        create_subject_splits(
            subject_ids=["sub-01", "sub-02"],
            train_fraction=0.70,
            validation_fraction=0.15,
            test_fraction=0.15,
            random_seed=23,
        )


def test_subject_splits_require_valid_fractions():
    with pytest.raises(
        ValueError,
        match="sum to 1.0",
    ):
        create_subject_splits(
            subject_ids=[
                "sub-01",
                "sub-02",
                "sub-03",
            ],
            train_fraction=0.60,
            validation_fraction=0.15,
            test_fraction=0.15,
            random_seed=23,
        )


def test_write_subject_splits(tmp_path):
    subject_splits = {
        "train": ["sub-02", "sub-01"],
        "validation": ["sub-03"],
        "test": ["sub-04"],
    }
    output_path = (
        tmp_path
        / "splits"
        / "subject_splits.csv"
    )

    write_subject_splits(
        subject_splits=subject_splits,
        output_path=output_path,
    )

    with output_path.open(
        encoding="utf-8",
        newline="",
    ) as csv_file:
        rows = list(csv.DictReader(csv_file))

    assert rows == [
        {
            "subject_id": "sub-01",
            "split": "train",
        },
        {
            "subject_id": "sub-02",
            "split": "train",
        },
        {
            "subject_id": "sub-03",
            "split": "validation",
        },
        {
            "subject_id": "sub-04",
            "split": "test",
        },
    ]