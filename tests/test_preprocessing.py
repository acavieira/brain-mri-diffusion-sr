import numpy as np
import pytest

from mri_diffusion.preprocessing import (
    normalize_volume_to_float01,
    prepare_hr_reference,
    resize_with_padding,
)


def test_normalize_volume_to_float01():
    volume = np.array(
        [0.0, 5.0, 10.0],
        dtype=np.float32,
    ).reshape(1, 1, 3)

    normalized = normalize_volume_to_float01(
        volume=volume,
        low_percentile=0,
        high_percentile=100,
    )

    expected = np.array(
        [0.0, 0.5, 1.0],
        dtype=np.float32,
    ).reshape(1, 1, 3)

    np.testing.assert_allclose(
        normalized,
        expected,
    )
    assert normalized.dtype == np.float32


def test_normalize_volume_clips_percentiles():
    volume = np.array(
        [0.0, 1.0, 2.0, 3.0, 4.0],
        dtype=np.float32,
    ).reshape(1, 1, 5)

    normalized = normalize_volume_to_float01(
        volume=volume,
        low_percentile=25,
        high_percentile=75,
    )

    expected = np.array(
        [0.0, 0.0, 0.5, 1.0, 1.0],
        dtype=np.float32,
    ).reshape(1, 1, 5)

    np.testing.assert_allclose(
        normalized,
        expected,
    )


def test_normalize_volume_replaces_non_finite_values():
    volume = np.array(
        [0.0, 1.0, np.nan, np.inf],
        dtype=np.float32,
    ).reshape(1, 1, 4)

    normalized = normalize_volume_to_float01(
        volume=volume,
        low_percentile=0,
        high_percentile=100,
    )

    expected = np.array(
        [0.0, 1.0, 0.0, 0.0],
        dtype=np.float32,
    ).reshape(1, 1, 4)

    np.testing.assert_allclose(
        normalized,
        expected,
    )


def test_normalize_constant_volume_returns_zeros():
    volume = np.full(
        (2, 3, 4),
        7.0,
        dtype=np.float32,
    )

    normalized = normalize_volume_to_float01(
        volume=volume,
        low_percentile=1,
        high_percentile=99,
    )

    assert np.all(normalized == 0.0)


def test_normalize_volume_rejects_invalid_input():
    with pytest.raises(
        ValueError,
        match="three-dimensional",
    ):
        normalize_volume_to_float01(
            volume=np.zeros((4, 4)),
            low_percentile=1,
            high_percentile=99,
        )

    with pytest.raises(
        ValueError,
        match="Percentiles",
    ):
        normalize_volume_to_float01(
            volume=np.zeros((2, 2, 2)),
            low_percentile=99,
            high_percentile=1,
        )

    with pytest.raises(
        ValueError,
        match="no finite",
    ):
        normalize_volume_to_float01(
            volume=np.full(
                (2, 2, 2),
                np.nan,
            ),
            low_percentile=1,
            high_percentile=99,
        )


def test_resize_with_padding_preserves_aspect_ratio():
    image = np.ones(
        (2, 4),
        dtype=np.float32,
    )

    prepared = resize_with_padding(
        image=image,
        target_size=8,
    )

    assert prepared.shape == (8, 8)
    assert prepared.dtype == np.float32

    assert np.all(
        prepared[:2] == 0.0
    )
    assert np.all(
        prepared[2:6] == 1.0
    )
    assert np.all(
        prepared[6:] == 0.0
    )


def test_resize_with_padding_rejects_invalid_input():
    with pytest.raises(
        ValueError,
        match="two-dimensional",
    ):
        resize_with_padding(
            image=np.zeros((2, 2, 2)),
            target_size=8,
        )

    with pytest.raises(
        ValueError,
        match="positive",
    ):
        resize_with_padding(
            image=np.zeros((2, 2)),
            target_size=0,
        )

    with pytest.raises(
        ValueError,
        match="non-finite",
    ):
        resize_with_padding(
            image=np.full((2, 2), np.nan),
            target_size=8,
        )


def test_prepare_hr_reference_clips_values():
    image = np.array(
        [[-1.0, 0.5, 2.0]],
        dtype=np.float32,
    )

    prepared = prepare_hr_reference(
        normalized_slice=image,
        target_size=3,
    )

    assert prepared.shape == (3, 3)
    assert prepared.dtype == np.float32
    assert prepared.min() == 0.0
    assert prepared.max() == 1.0