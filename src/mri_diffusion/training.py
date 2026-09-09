"""Training operations for conditional diffusion."""

import torch
from torch.nn import functional as F


def calculate_training_loss(
    model,
    scheduler,
    hr_images,
    conditions,
):
    """Calculate the DDPM noise-prediction loss."""

    batch_size = hr_images.shape[0]

    timesteps = scheduler.sample_timesteps(
        batch_size=batch_size
    )

    true_noise = torch.randn_like(
        hr_images
    )

    noisy_images = scheduler.add_noise(
        clean_images=hr_images,
        noise=true_noise,
        timesteps=timesteps,
    )

    predicted_noise = model(
        noisy_images,
        conditions,
        timesteps,
    )

    loss = F.mse_loss(
        predicted_noise,
        true_noise,
    )

    return loss