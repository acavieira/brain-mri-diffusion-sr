import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "src"),
)

from mri_diffusion.config import (
    BETA_END,
    BETA_START,
    BLUR_SIGMA,
    DATA_DIR,
    DIFFUSION_STEPS,
    HIGH_PERCENTILE,
    LOW_PERCENTILE,
    NOISE_SIGMA,
    RANDOM_SEED,
    SCALE,
    SLICES_PER_ORIENTATION,
    TARGET_SIZE,
)
from mri_diffusion.data import find_t1w_files
from mri_diffusion.dataset import (
    MRIDiffusionDataset,
    build_sample_index,
    model_tensor_to_image,
)
from mri_diffusion.device import get_device
from mri_diffusion.scheduler import DiffusionScheduler


def main():
    device = get_device()

    paths = find_t1w_files(DATA_DIR)

    samples = build_sample_index(
        paths=paths,
        slices_per_orientation=SLICES_PER_ORIENTATION,
        random_seed=RANDOM_SEED,
    )

    dataset = MRIDiffusionDataset(
        samples=samples,
        target_size=TARGET_SIZE,
        low_percentile=LOW_PERCENTILE,
        high_percentile=HIGH_PERCENTILE,
        scale=SCALE,
        blur_sigma=BLUR_SIGMA,
        noise_sigma=NOISE_SIGMA,
    )

    selected_position = len(dataset) // 2
    selected_sample = dataset[selected_position]

    clean_image = selected_sample["hr"]

    clean_images = clean_image.unsqueeze(0).to(
        device
    )

    scheduler = DiffusionScheduler(
        diffusion_steps=DIFFUSION_STEPS,
        beta_start=BETA_START,
        beta_end=BETA_END,
        device=device,
    )

    cpu_generator = torch.Generator()
    cpu_generator.manual_seed(RANDOM_SEED)

    noise = torch.randn(
        clean_images.shape,
        generator=cpu_generator,
        dtype=torch.float32,
    ).to(device)

    timesteps_to_show = (
        0,
        99,
        249,
        499,
        749,
        999,
    )

    figure, plot_axes = plt.subplots(
        2,
        4,
        figsize=(12, 7),
    )

    plot_axes = plot_axes.flatten()

    clean_display = model_tensor_to_image(
        clean_image
    ).squeeze(0).numpy()

    plot_axes[0].imshow(
        clean_display,
        cmap="gray",
        vmin=0.0,
        vmax=1.0,
    )
    plot_axes[0].set_title("Clean HR")
    plot_axes[0].axis("off")

    for plot_position, timestep in enumerate(
        timesteps_to_show,
        start=1,
    ):
        timestep_tensor = torch.tensor(
            [timestep],
            dtype=torch.long,
            device=device,
        )

        noisy_images = scheduler.add_noise(
            clean_images=clean_images,
            noise=noise,
            timesteps=timestep_tensor,
        )

        noisy_image = model_tensor_to_image(
            noisy_images[0, 0]
        ).detach().cpu().numpy()

        alpha_bar = scheduler.alpha_bars[
            timestep
        ].item()

        signal_coefficient = alpha_bar ** 0.5

        noise_coefficient = (
            1.0 - alpha_bar
        ) ** 0.5

        plot_axes[plot_position].imshow(
            noisy_image,
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
        )

        plot_axes[plot_position].set_title(
            f"t = {timestep}\n"
            f"signal = {signal_coefficient:.3f}\n"
            f"noise = {noise_coefficient:.3f}"
        )

        plot_axes[plot_position].axis("off")

        print(
            f"t={timestep}: "
            f"signal={signal_coefficient:.6f}, "
            f"noise={noise_coefficient:.6f}"
        )

    plot_axes[-1].axis("off")

    figure.suptitle(
        f"{selected_sample['sample_id']} "
        f"on {device}",
        fontsize=14,
    )

    figure.tight_layout(
        rect=(0, 0, 1, 0.93),
        h_pad=4.0,
    )

    plt.show()


if __name__ == "__main__":
    main()