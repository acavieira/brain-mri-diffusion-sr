import numpy as np
import pytest

from mri_diffusion.degradation import (
    create_condition,
    degrade_image,
)


def test_degrade_image_reduces_resolution():
    hr_image = np.linspace(
        0.0,
        1.0,
        64 * 64,
        dtype=np.float32,
    ).reshape(64, 64)

    lr_image = degrade_image(
        hr_image=hr_image,
        scale=2,
        blur_sigma=0.65,
        noise_sigma=0.0,
        random_generator=None,
    )

    assert lr_image.shape == (32, 32)
    assert lr_image.dtype == np.float32
    assert lr_image.min() >= 0.0
    assert lr_image.max() <= 1.0


def test_degrade_image_is_reproducible():
    hr_image = np.full(
        (32, 32),
        0.5,
        dtype=np.float32,
    )

    first_generator = np.random.default_rng(23)
    second_generator = np.random.default_rng(23)

    first_lr = degrade_image(
        hr_image=hr_image,
        scale=2,
        blur_sigma=0.65,
        noise_sigma=0.02,
        random_generator=first_generator,
    )
    second_lr = degrade_image(
        hr_image=hr_image,
        scale=2,
        blur_sigma=0.65,
        noise_sigma=0.02,
        random_generator=second_generator,
    )

    np.testing.assert_array_equal(
        first_lr,
        second_lr,
    )


def test_noise_changes_with_different_seeds():
    hr_image = np.full(
        (32, 32),
        0.5,
        dtype=np.float32,
    )

    first_lr = degrade_image(
        hr_image=hr_image,
        scale=2,
        blur_sigma=0.65,
        noise_sigma=0.02,
        random_generator=np.random.default_rng(23),
    )
    second_lr = degrade_image(
        hr_image=hr_image,
        scale=2,
        blur_sigma=0.65,
        noise_sigma=0.02,
        random_generator=np.random.default_rng(24),
    )

    assert not np.array_equal(
        first_lr,
        second_lr,
    )


def test_degrade_image_rejects_invalid_arguments():
    image = np.zeros(
        (16, 16),
        dtype=np.float32,
    )

    with pytest.raises(
        ValueError,
        match="two-dimensional",
    ):
        degrade_image(
            hr_image=np.zeros((2, 16, 16)),
            scale=2,
            blur_sigma=0.65,
            noise_sigma=0.02,
            random_generator=np.random.default_rng(23),
        )

    with pytest.raises(
        ValueError,
        match="greater than or equal",
    ):
        degrade_image(
            hr_image=image,
            scale=1,
            blur_sigma=0.65,
            noise_sigma=0.02,
            random_generator=np.random.default_rng(23),
        )

    with pytest.raises(
        ValueError,
        match="Blur sigma",
    ):
        degrade_image(
            hr_image=image,
            scale=2,
            blur_sigma=-1.0,
            noise_sigma=0.02,
            random_generator=np.random.default_rng(23),
        )

    with pytest.raises(
        ValueError,
        match="Noise sigma",
    ):
        degrade_image(
            hr_image=image,
            scale=2,
            blur_sigma=0.65,
            noise_sigma=-1.0,
            random_generator=np.random.default_rng(23),
        )

    with pytest.raises(
        ValueError,
        match="random generator",
    ):
        degrade_image(
            hr_image=image,
            scale=2,
            blur_sigma=0.65,
            noise_sigma=0.02,
            random_generator=None,
        )


def test_create_condition_restores_target_shape():
    lr_image = np.linspace(
        0.0,
        1.0,
        32 * 32,
        dtype=np.float32,
    ).reshape(32, 32)

    condition = create_condition(
        lr_image=lr_image,
        target_size=64,
    )

    assert condition.shape == (64, 64)
    assert condition.dtype == np.float32
    assert condition.min() >= 0.0
    assert condition.max() <= 1.0


def test_create_condition_rejects_invalid_arguments():
    with pytest.raises(
        ValueError,
        match="two-dimensional",
    ):
        create_condition(
            lr_image=np.zeros((2, 16, 16)),
            target_size=32,
        )

    with pytest.raises(
        ValueError,
        match="positive",
    ):
        create_condition(
            lr_image=np.zeros((16, 16)),
            target_size=0,
        )

    with pytest.raises(
        ValueError,
        match="non-finite",
    ):
        create_condition(
            lr_image=np.full((16, 16), np.nan),
            target_size=32,
        )