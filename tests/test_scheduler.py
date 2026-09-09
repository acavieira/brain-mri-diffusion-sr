import pytest
import torch

from mri_diffusion.scheduler import (
    DiffusionScheduler,
)


def create_test_scheduler():
    return DiffusionScheduler(
        diffusion_steps=10,
        beta_start=0.0001,
        beta_end=0.02,
        device="cpu",
    )


def test_scheduler_creates_expected_values():
    scheduler = create_test_scheduler()

    assert scheduler.betas.shape == (10,)
    assert scheduler.alphas.shape == (10,)
    assert scheduler.alpha_bars.shape == (10,)

    assert torch.isclose(
        scheduler.betas[0],
        torch.tensor(0.0001),
    )
    assert torch.isclose(
        scheduler.betas[-1],
        torch.tensor(0.02),
    )

    torch.testing.assert_close(
        scheduler.alphas,
        1.0 - scheduler.betas,
    )

    expected_alpha_bars = torch.cumprod(
        scheduler.alphas,
        dim=0,
    )

    torch.testing.assert_close(
        scheduler.alpha_bars,
        expected_alpha_bars,
    )

    assert torch.all(
        scheduler.alpha_bars[1:]
        < scheduler.alpha_bars[:-1]
    )


def test_add_noise_matches_ddpm_formula():
    scheduler = create_test_scheduler()

    clean_images = torch.ones(
        (2, 1, 2, 2),
        dtype=torch.float32,
    )
    noise = torch.full(
        (2, 1, 2, 2),
        2.0,
        dtype=torch.float32,
    )
    timesteps = torch.tensor(
        [0, 9],
        dtype=torch.long,
    )

    noisy_images = scheduler.add_noise(
        clean_images=clean_images,
        noise=noise,
        timesteps=timesteps,
    )

    selected_alpha_bars = scheduler.alpha_bars[
        timesteps
    ].view(2, 1, 1, 1)

    expected = (
        torch.sqrt(selected_alpha_bars)
        * clean_images
        + torch.sqrt(1.0 - selected_alpha_bars)
        * noise
    )

    torch.testing.assert_close(
        noisy_images,
        expected,
    )


def test_sample_timesteps_returns_valid_values():
    scheduler = create_test_scheduler()

    torch.manual_seed(23)

    timesteps = scheduler.sample_timesteps(
        batch_size=20
    )

    assert timesteps.shape == (20,)
    assert timesteps.dtype == torch.long
    assert timesteps.min().item() >= 0
    assert timesteps.max().item() < 10


def test_scheduler_rejects_invalid_configuration():
    with pytest.raises(
        ValueError,
        match="positive",
    ):
        DiffusionScheduler(
            diffusion_steps=0,
            beta_start=0.0001,
            beta_end=0.02,
            device="cpu",
        )

    with pytest.raises(
        ValueError,
        match="Betas",
    ):
        DiffusionScheduler(
            diffusion_steps=10,
            beta_start=0.02,
            beta_end=0.0001,
            device="cpu",
        )


def test_add_noise_rejects_invalid_inputs():
    scheduler = create_test_scheduler()

    clean_images = torch.zeros(
        (2, 1, 4, 4),
        dtype=torch.float32,
    )
    noise = torch.zeros_like(clean_images)

    with pytest.raises(
        ValueError,
        match=r"\[B, C, H, W\]",
    ):
        scheduler.add_noise(
            clean_images=torch.zeros((4, 4)),
            noise=torch.zeros((4, 4)),
            timesteps=torch.tensor(
                [0],
                dtype=torch.long,
            ),
        )

    with pytest.raises(
        ValueError,
        match="same shape",
    ):
        scheduler.add_noise(
            clean_images=clean_images,
            noise=torch.zeros((1, 1, 4, 4)),
            timesteps=torch.tensor(
                [0, 1],
                dtype=torch.long,
            ),
        )

    with pytest.raises(
        ValueError,
        match=r"shape \[B\]",
    ):
        scheduler.add_noise(
            clean_images=clean_images,
            noise=noise,
            timesteps=torch.tensor(
                [0],
                dtype=torch.long,
            ),
        )

    with pytest.raises(
        ValueError,
        match="exceeds",
    ):
        scheduler.add_noise(
            clean_images=clean_images,
            noise=noise,
            timesteps=torch.tensor(
                [0, 10],
                dtype=torch.long,
            ),
        )

def test_posterior_variances_are_valid():
    scheduler = create_test_scheduler()

    assert scheduler.posterior_variances.shape == (
        10,
    )

    assert scheduler.posterior_variances[0] == 0.0

    assert torch.all(
        scheduler.posterior_variances >= 0.0
    )


def test_reverse_step_matches_ddpm_formula():
    scheduler = create_test_scheduler()

    noisy_images = torch.full(
        (1, 1, 2, 2),
        0.5,
        dtype=torch.float32,
    )

    predicted_noise = torch.full(
        (1, 1, 2, 2),
        0.25,
        dtype=torch.float32,
    )

    random_noise = torch.full(
        (1, 1, 2, 2),
        1.5,
        dtype=torch.float32,
    )

    timesteps = torch.tensor(
        [9],
        dtype=torch.long,
    )

    output = scheduler.reverse_step(
        noisy_images=noisy_images,
        predicted_noise=predicted_noise,
        timesteps=timesteps,
        random_noise=random_noise,
    )

    beta = scheduler.betas[9]
    alpha = scheduler.alphas[9]
    alpha_bar = scheduler.alpha_bars[9]

    posterior_variance = (
        scheduler.posterior_variances[9]
    )

    expected_mean = (
        1.0 / torch.sqrt(alpha)
    ) * (
        noisy_images
        - (
            beta
            / torch.sqrt(1.0 - alpha_bar)
        )
        * predicted_noise
    )

    expected = (
        expected_mean
        + torch.sqrt(posterior_variance)
        * random_noise
    )

    torch.testing.assert_close(
        output,
        expected,
    )


def test_reverse_step_adds_no_noise_at_zero():
    scheduler = create_test_scheduler()

    noisy_images = torch.randn(
        (1, 1, 2, 2),
        dtype=torch.float32,
    )

    predicted_noise = torch.randn_like(
        noisy_images
    )

    timesteps = torch.tensor(
        [0],
        dtype=torch.long,
    )

    first_output = scheduler.reverse_step(
        noisy_images=noisy_images,
        predicted_noise=predicted_noise,
        timesteps=timesteps,
        random_noise=torch.zeros_like(
            noisy_images
        ),
    )

    second_output = scheduler.reverse_step(
        noisy_images=noisy_images,
        predicted_noise=predicted_noise,
        timesteps=timesteps,
        random_noise=torch.ones_like(
            noisy_images
        ),
    )

    torch.testing.assert_close(
        first_output,
        second_output,
    )