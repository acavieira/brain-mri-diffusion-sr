from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from mri_diffusion.dataset import (
    build_sample_index,
    find_subject_split,
    prepare_sample,
)


def write_test_volume(path, shape):
    data = np.arange(
        np.prod(shape),
        dtype=np.float32,
    ).reshape(shape)

    image = nib.Nifti1Image(
        data,
        np.eye(4),
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    nib.save(
        image,
        str(path),
    )


def test_find_subject_split():
    subject_splits = {
        "train": ["sub-01"],
        "validation": ["sub-02"],
        "test": ["sub-03"],
    }

    assert find_subject_split(
        subject_id="sub-02",
        subject_splits=subject_splits,
    ) == "validation"


def test_find_subject_split_rejects_missing_subject():
    subject_splits = {
        "train": ["sub-01"],
        "validation": [],
        "test": [],
    }

    with pytest.raises(
        ValueError,
        match="not assigned",
    ):
        find_subject_split(
            subject_id="sub-02",
            subject_splits=subject_splits,
        )


def test_find_subject_split_rejects_leakage():
    subject_splits = {
        "train": ["sub-01"],
        "validation": [],
        "test": ["sub-01"],
    }

    with pytest.raises(
        ValueError,
        match="multiple splits",
    ):
        find_subject_split(
            subject_id="sub-01",
            subject_splits=subject_splits,
        )


def test_build_sample_index(tmp_path):
    path = (
        tmp_path
        / "data"
        / "sub-01_ses-1_T1w.nii"
    )
    write_test_volume(
        path=path,
        shape=(8, 10, 12),
    )

    subject_splits = {
        "train": ["sub-01"],
        "validation": [],
        "test": [],
    }

    samples = build_sample_index(
        paths=[path],
        slices_per_orientation=4,
        random_seed=23,
        subject_splits=subject_splits,
    )

    assert len(samples) == 12

    first_sample = samples[0]

    assert first_sample["sample_id"] == "sample-000000"
    assert first_sample["volume_path"] == path
    assert first_sample["subject_id"] == "sub-01"
    assert first_sample["split"] == "train"
    assert first_sample["orientation"] == "axial"
    assert first_sample["slice_index"] == 4
    assert first_sample["degradation_seed"] == 23

    last_sample = samples[-1]

    assert last_sample["sample_id"] == "sample-000011"
    assert last_sample["orientation"] == "sagittal"
    assert last_sample["slice_index"] == 5
    assert last_sample["degradation_seed"] == 34


def test_build_sample_index_without_splits(tmp_path):
    path = (
        tmp_path
        / "data"
        / "sub-01_ses-1_T1w.nii"
    )
    write_test_volume(
        path=path,
        shape=(8, 10, 12),
    )

    samples = build_sample_index(
        paths=[path],
        slices_per_orientation=2,
        random_seed=23,
    )

    assert len(samples) == 6
    assert all(
        sample["split"] is None
        for sample in samples
    )


def test_build_sample_index_rejects_empty_paths():
    with pytest.raises(
        ValueError,
        match="At least one",
    ):
        build_sample_index(
            paths=[],
            slices_per_orientation=4,
            random_seed=23,
        )


def test_prepare_sample_returns_expected_images(
    tmp_path,
):
    path = (
        tmp_path
        / "data"
        / "sub-01_ses-1_T1w.nii"
    )
    write_test_volume(
        path=path,
        shape=(8, 10, 12),
    )

    samples = build_sample_index(
        paths=[path],
        slices_per_orientation=2,
        random_seed=23,
    )

    first_result = prepare_sample(
        sample=samples[0],
        target_size=16,
        low_percentile=1,
        high_percentile=99,
        scale=2,
        blur_sigma=0.65,
        noise_sigma=0.02,
    )

    second_result = prepare_sample(
        sample=samples[0],
        target_size=16,
        low_percentile=1,
        high_percentile=99,
        scale=2,
        blur_sigma=0.65,
        noise_sigma=0.02,
    )

    assert first_result["hr"].shape == (16, 16)
    assert first_result["lr"].shape == (8, 8)
    assert first_result["condition"].shape == (16, 16)

    assert first_result["hr"].dtype == np.float32
    assert first_result["lr"].dtype == np.float32
    assert first_result["condition"].dtype == np.float32

    assert first_result["hr"].min() >= 0.0
    assert first_result["hr"].max() <= 1.0
    assert first_result["lr"].min() >= 0.0
    assert first_result["lr"].max() <= 1.0
    assert first_result["condition"].min() >= 0.0
    assert first_result["condition"].max() <= 1.0

    np.testing.assert_array_equal(
        first_result["hr"],
        second_result["hr"],
    )
    np.testing.assert_array_equal(
        first_result["lr"],
        second_result["lr"],
    )
    np.testing.assert_array_equal(
        first_result["condition"],
        second_result["condition"],
    )