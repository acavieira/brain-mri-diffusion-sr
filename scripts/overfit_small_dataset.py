import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "src"),
)

from mri_diffusion.config import (
    BASE_CHANNELS,
    BETA_END,
    BETA_START,
    BLUR_SIGMA,
    CONDITION_CHANNELS,
    DATA_DIR,
    DIFFUSION_STEPS,
    GROUP_NORM_GROUPS,
    HIGH_PERCENTILE,
    IMAGE_CHANNELS,
    LEARNING_RATE,
    LOW_PERCENTILE,
    MAX_GRADIENT_NORM,
    NOISE_SIGMA,
    OVERFIT_BATCH_SIZE,
    OVERFIT_EPOCHS,
    OVERFIT_SAMPLE_COUNT,
    OVERFIT_TARGET_SIZE,
    RANDOM_SEED,
    SCALE,
    SLICES_PER_ORIENTATION,
    TIME_EMBEDDING_DIM,
    TIME_HIDDEN_DIM,
    WEIGHT_DECAY,
)
from mri_diffusion.data import find_t1w_files
from mri_diffusion.dataset import (
    MRIDiffusionDataset,
    build_sample_index,
)
from mri_diffusion.device import get_device
from mri_diffusion.scheduler import DiffusionScheduler
from mri_diffusion.training import train_one_epoch
from mri_diffusion.unet import ConditionalUNet


def main():
    torch.manual_seed(RANDOM_SEED)

    device = get_device()

    paths = find_t1w_files(DATA_DIR)

    all_samples = build_sample_index(
        paths=paths,
        slices_per_orientation=SLICES_PER_ORIENTATION,
        random_seed=RANDOM_SEED,
    )

    center_position = len(all_samples) // 2

    first_position = (
        center_position
        - OVERFIT_SAMPLE_COUNT // 2
    )

    selected_samples = all_samples[
        first_position:
        first_position + OVERFIT_SAMPLE_COUNT
    ]

    dataset = MRIDiffusionDataset(
        samples=selected_samples,
        target_size=OVERFIT_TARGET_SIZE,
        low_percentile=LOW_PERCENTILE,
        high_percentile=HIGH_PERCENTILE,
        scale=SCALE,
        blur_sigma=BLUR_SIGMA,
        noise_sigma=NOISE_SIGMA,
    )

    prepared_samples = [
        dataset[index]
        for index in range(len(dataset))
    ]

    data_loader = DataLoader(
        prepared_samples,
        batch_size=OVERFIT_BATCH_SIZE,
        shuffle=False,
    )

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

    epoch_losses = []
    gradient_norms = []

    print(f"Device: {device}")
    print(f"Selected samples: {len(selected_samples)}")
    print(f"Training size: {OVERFIT_TARGET_SIZE}")
    print(f"Epochs: {OVERFIT_EPOCHS}")

    for epoch in range(1, OVERFIT_EPOCHS + 1):
        mean_loss, mean_gradient_norm = (
            train_one_epoch(
                model=model,
                scheduler=scheduler,
                optimizer=optimizer,
                data_loader=data_loader,
                device=device,
                max_gradient_norm=MAX_GRADIENT_NORM,
            )
        )

        epoch_losses.append(mean_loss)
        gradient_norms.append(
            mean_gradient_norm
        )

        if epoch == 1 or epoch % 5 == 0:
            print(
                f"Epoch {epoch:03d}: "
                f"loss={mean_loss:.6f}, "
                f"gradient_norm="
                f"{mean_gradient_norm:.6f}"
            )

    comparison_window = min(
        10,
        OVERFIT_EPOCHS // 2,
    )

    initial_loss = sum(
        epoch_losses[:comparison_window]
    ) / comparison_window

    final_loss = sum(
        epoch_losses[-comparison_window:]
    ) / comparison_window

    print(f"Initial mean loss: {initial_loss:.6f}")
    print(f"Final mean loss: {final_loss:.6f}")
    print(
        f"Loss improved: "
        f"{final_loss < initial_loss}"
    )

    figure, plot_axis = plt.subplots(
        figsize=(9, 4),
    )

    plot_axis.plot(
        range(1, OVERFIT_EPOCHS + 1),
        epoch_losses,
        color="tab:blue",
    )

    plot_axis.set_title(
        "Controlled overfitting on four MRI slices"
    )
    plot_axis.set_xlabel("Epoch")
    plot_axis.set_ylabel("Noise prediction MSE")
    plot_axis.grid(alpha=0.3)

    figure.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

    