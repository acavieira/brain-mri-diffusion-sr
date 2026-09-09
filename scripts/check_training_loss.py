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
    RANDOM_SEED,
    TIME_EMBEDDING_DIM,
    TIME_HIDDEN_DIM,
)
from mri_diffusion.device import get_device
from mri_diffusion.scheduler import DiffusionScheduler
from mri_diffusion.training import (
    calculate_training_loss,
)
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

    model.train()

    loss = calculate_training_loss(
        model=model,
        scheduler=scheduler,
        hr_images=hr_images,
        conditions=conditions,
    )

    loss.backward()

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    parameters_with_gradients = [
        parameter
        for parameter in trainable_parameters
        if parameter.grad is not None
    ]

    print(f"Device: {device}")
    print(f"Training loss: {loss.item():.6f}")
    print(
        f"Loss is finite: "
        f"{torch.isfinite(loss).item()}"
    )
    print(
        f"Parameters with gradients: "
        f"{len(parameters_with_gradients)} "
        f"of {len(trainable_parameters)}"
    )


if __name__ == "__main__":
    main()