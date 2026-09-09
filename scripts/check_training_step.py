import sys
from pathlib import Path

import torch

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "src"),
)

from mri_diffusion.config import (
    BASE_CHANNELS,
    BETA_END,
    BETA_START,
    CONDITION_CHANNELS,
    DIFFUSION_STEPS,
    GROUP_NORM_GROUPS,
    IMAGE_CHANNELS,
    LEARNING_RATE,
    MAX_GRADIENT_NORM,
    RANDOM_SEED,
    TIME_EMBEDDING_DIM,
    TIME_HIDDEN_DIM,
    WEIGHT_DECAY,
)
from mri_diffusion.device import get_device
from mri_diffusion.scheduler import DiffusionScheduler
from mri_diffusion.training import train_one_batch
from mri_diffusion.unet import ConditionalUNet


def main():
    torch.manual_seed(RANDOM_SEED)

    device = get_device()

    model = ConditionalUNet(
        image_channels=IMAGE_CHANNELS,
        condition_channels=CONDITION_CHANNELS,
        base_channels=BASE_CHANNELS,
        time_embedding_dim=TIME_EMBEDDING_DIM,
        time_hidden_dim=TIME_HIDDEN_DIM,
        group_count=GROUP_NORM_GROUPS,
    ).to(device)

    scheduler = DiffusionScheduler(
        diffusion_steps=DIFFUSION_STEPS,
        beta_start=BETA_START,
        beta_end=BETA_END,
        device=device,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    hr_images = (
        torch.rand(
            (1, IMAGE_CHANNELS, 64, 64),
            dtype=torch.float32,
        )
        * 2.0
        - 1.0
    ).to(device)

    conditions = (
        torch.rand(
            (1, CONDITION_CHANNELS, 64, 64),
            dtype=torch.float32,
        )
        * 2.0
        - 1.0
    ).to(device)

    weight_before_training = (
        model
        .output_convolution
        .weight
        .detach()
        .clone()
    )

    loss, gradient_norm = train_one_batch(
        model=model,
        scheduler=scheduler,
        optimizer=optimizer,
        hr_images=hr_images,
        conditions=conditions,
        max_gradient_norm=MAX_GRADIENT_NORM,
    )

    weight_after_training = (
        model
        .output_convolution
        .weight
        .detach()
    )

    weights_changed = not torch.equal(
        weight_before_training,
        weight_after_training,
    )

    print(f"Device: {device}")
    print(f"Training loss: {loss:.6f}")
    print(f"Gradient norm: {gradient_norm:.6f}")
    print(f"Weights changed: {weights_changed}")


if __name__ == "__main__":
    main()