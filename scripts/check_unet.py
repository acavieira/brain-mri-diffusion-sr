import sys
from pathlib import Path

import torch

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "src"),
)

from mri_diffusion.config import (
    BASE_CHANNELS,
    CONDITION_CHANNELS,
    GROUP_NORM_GROUPS,
    IMAGE_CHANNELS,
    RANDOM_SEED,
    TIME_EMBEDDING_DIM,
    TIME_HIDDEN_DIM,
)
from mri_diffusion.device import get_device
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

    noisy_images = torch.randn(
        (1, IMAGE_CHANNELS, 64, 64),
        dtype=torch.float32,
    ).to(device)

    conditions = torch.randn(
        (1, CONDITION_CHANNELS, 64, 64),
        dtype=torch.float32,
    ).to(device)

    timesteps = torch.tensor(
        [500],
        dtype=torch.long,
        device=device,
    )

    model.eval()

    with torch.inference_mode():
        predicted_noise = model(
            noisy_images,
            conditions,
            timesteps,
        )

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(f"Device: {device}")
    print(f"Noisy image shape: {noisy_images.shape}")
    print(f"Condition shape: {conditions.shape}")
    print(f"Timestep shape: {timesteps.shape}")
    print(
        f"Predicted noise shape: "
        f"{predicted_noise.shape}"
    )
    print(f"Trainable parameters: {parameter_count}")
    print(
        f"Output contains only finite values: "
        f"{torch.isfinite(predicted_noise).all().item()}"
    )


if __name__ == "__main__":
    main()