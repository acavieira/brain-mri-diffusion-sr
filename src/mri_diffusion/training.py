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

def train_one_batch(
    model,
    scheduler,
    optimizer,
    hr_images,
    conditions,
    max_gradient_norm,
):
    """Update the model using one batch."""

    if max_gradient_norm <= 0:
        raise ValueError(
            "Maximum gradient norm must be positive"
        )

    model.train()

    optimizer.zero_grad(
        set_to_none=True
    )

    loss = calculate_training_loss(
        model=model,
        scheduler=scheduler,
        hr_images=hr_images,
        conditions=conditions,
    )

    loss.backward()

    gradient_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=max_gradient_norm,
    )

    optimizer.step()

    return (
        loss.item(),
        gradient_norm.item(),
    )