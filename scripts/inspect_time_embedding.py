import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "src"),
)

from mri_diffusion.config import (
    TIME_EMBEDDING_DIM,
)
from mri_diffusion.device import get_device
from mri_diffusion.unet import (
    SinusoidalTimeEmbedding,
)


def main():
    device = get_device()

    timestep_values = (
        0,
        99,
        249,
        499,
        749,
        999,
    )

    timesteps = torch.tensor(
        timestep_values,
        dtype=torch.long,
        device=device,
    )

    time_embedding = SinusoidalTimeEmbedding(
        embedding_dim=TIME_EMBEDDING_DIM
    ).to(device)

    embeddings = time_embedding(
        timesteps
    )

    print(f"Device: {device}")
    print(f"Timesteps shape: {timesteps.shape}")
    print(f"Embedding shape: {embeddings.shape}")
    print(
        f"Embedding range: "
        f"{embeddings.min().item():.3f} to "
        f"{embeddings.max().item():.3f}"
    )

    display_embeddings = (
        embeddings
        .detach()
        .cpu()
        .numpy()
    )

    figure, plot_axis = plt.subplots(
        figsize=(11, 4),
    )

    image = plot_axis.imshow(
        display_embeddings,
        aspect="auto",
        cmap="coolwarm",
        vmin=-1.0,
        vmax=1.0,
    )

    plot_axis.set_title(
        "Sinusoidal timestep embeddings"
    )
    plot_axis.set_xlabel(
        "Embedding dimension"
    )
    plot_axis.set_ylabel(
        "Diffusion timestep"
    )

    plot_axis.set_yticks(
        range(len(timestep_values))
    )
    plot_axis.set_yticklabels(
        timestep_values
    )

    figure.colorbar(
        image,
        ax=plot_axis,
        label="Embedding value",
    )

    figure.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()