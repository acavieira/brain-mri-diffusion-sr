"""Tests for NIfTI discovery, loading and slice extraction."""

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from mri_diffusion.data import (
    extract_slice,
    find_t1w_files,
    load_canonical_volume,
    select_centered_indices,
    subject_id_from_path,
)


def write_nifti(path, shape, fill_value=0):
    """Create a small synthetic NIfTI file for a test."""
    data = np.full(
        shape,
        fill_value,
        dtype=np.float32,
    )
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


def test_find_flat_t1w_file(tmp_path):
    path = (
        tmp_path
        / "data"
        / "sub-0_ses-1_T1w.nii"
    )
    write_nifti(path, (4, 5, 6))

    assert find_t1w_files(path.parent) == [path]


def test_find_nested_bids_t1w_file(tmp_path):
    path = (
        tmp_path
        / "data"
        / "sub-0"
        / "ses-1"
        / "anat"
        / "sub-0_ses-1_T1w.nii.gz"
    )
    write_nifti(path, (4, 5, 6))

    assert find_t1w_files(tmp_path / "data") == [path]


def test_find_t1w_excludes_derivatives(tmp_path):
    data_dir = tmp_path / "data"

    raw_path = (
        data_dir
        / "sub-0_ses-1_T1w.nii"
    )
    derivative_path = (
        data_dir
        / "derivatives"
        / "sub-1_ses-1_T1w.nii"
    )

    write_nifti(raw_path, (4, 5, 6))
    write_nifti(derivative_path, (4, 5, 6))

    assert find_t1w_files(data_dir) == [raw_path]


def test_find_t1w_rejects_missing_directory(tmp_path):
    with pytest.raises(FileNotFoundError):
        find_t1w_files(tmp_path / "missing")


def test_find_t1w_rejects_empty_directory(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    with pytest.raises(FileNotFoundError):
        find_t1w_files(data_dir)


@pytest.mark.parametrize(
    "path",
    [
        Path("data/sub-0_ses-1_T1w.nii"),
        Path(
            "data/sub-0/ses-1/anat/"
            "sub-0_ses-1_T1w.nii"
        ),
    ],
)
def test_subject_id_from_flat_and_nested_paths(path):
    assert subject_id_from_path(path) == "sub-0"


def test_subject_id_rejects_path_without_subject():
    with pytest.raises(ValueError):
        subject_id_from_path(
            Path("data/session-1_T1w.nii")
        )


def test_load_canonical_volume_returns_float32(tmp_path):
    path = tmp_path / "sub-0_T1w.nii"
    write_nifti(path, (4, 5, 6))

    volume = load_canonical_volume(path)

    assert volume.shape == (4, 5, 6)
    assert volume.dtype == np.float32


def test_load_canonical_volume_rejects_4d(tmp_path):
    path = tmp_path / "sub-0_T1w.nii"
    write_nifti(path, (4, 5, 6, 2))

    with pytest.raises(
        ValueError,
        match="three-dimensional",
    ):
        load_canonical_volume(path)


def test_load_canonical_volume_rejects_invalid_values(tmp_path):
    path = tmp_path / "sub-0_T1w.nii"
    write_nifti(
        path,
        (4, 5, 6),
        fill_value=np.nan,
    )

    with pytest.raises(
        ValueError,
        match="no finite intensity",
    ):
        load_canonical_volume(path)


def test_select_centered_indices_has_exact_count():
    indices = select_centered_indices(
        axis_length=236,
        count=30,
    )

    assert len(indices) == 30
    assert indices[0] == 103
    assert indices[-1] == 132


def test_select_centered_indices_rejects_zero():
    with pytest.raises(ValueError):
        select_centered_indices(
            axis_length=236,
            count=0,
        )


def test_select_centered_indices_rejects_too_many_slices():
    with pytest.raises(ValueError):
        select_centered_indices(
            axis_length=10,
            count=11,
        )


@pytest.mark.parametrize(
    ("orientation", "expected_shape"),
    [
        ("sagittal", (4, 3)),
        ("coronal", (4, 2)),
        ("axial", (3, 2)),
    ],
)
def test_extract_slice_shapes(orientation, expected_shape):
    volume = np.zeros(
        (2, 3, 4),
        dtype=np.float64,
    )

    extracted = extract_slice(
        volume,
        orientation,
        0,
    )

    assert extracted.shape == expected_shape
    assert extracted.dtype == np.float32
    assert extracted.flags["C_CONTIGUOUS"]


def test_extract_slice_rejects_unknown_orientation():
    volume = np.zeros((2, 3, 4))

    with pytest.raises(
        ValueError,
        match="Unknown orientation",
    ):
        extract_slice(
            volume,
            "diagonal",
            0,
        )


def test_extract_slice_rejects_out_of_range_index():
    volume = np.zeros((2, 3, 4))

    with pytest.raises(IndexError):
        extract_slice(
            volume,
            "axial",
            4,
        )

def test_extract_slice_uses_expected_display_orientation():
    volume = np.arange(
        2 * 3 * 4,
        dtype=np.float32,
    ).reshape(2, 3, 4)

    extracted = extract_slice(
        volume=volume,
        orientation="axial",
        index=0,
    )

    expected = np.array(
        [
            [8, 20],
            [4, 16],
            [0, 12],
        ],
        dtype=np.float32,
    )

    np.testing.assert_array_equal(extracted, expected)