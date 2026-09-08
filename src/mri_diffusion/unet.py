"""Small building blocks for the conditional U-Net."""

import math

import torch
from torch import nn


class SinusoidalTimeEmbedding(nn.Module):
    """Convert each diffusion timestep into a fixed vector."""

    def __init__(self, embedding_dim):
        super().__init__()

        if embedding_dim <= 0:
            raise ValueError(
                "Embedding dimension must be positive"
            )

        if embedding_dim % 2 != 0:
            raise ValueError(
                "Embedding dimension must be even"
            )

        self.embedding_dim = embedding_dim

    def forward(self, timesteps):
        if timesteps.ndim != 1:
            raise ValueError(
                "Timesteps must have shape [B]"
            )

        half_dimension = self.embedding_dim // 2

        frequency_positions = torch.arange(
            half_dimension,
            device=timesteps.device,
            dtype=torch.float32,
        )

        frequencies = torch.exp(
            -math.log(10000.0)
            * frequency_positions
            / half_dimension
        )

        angles = (
            timesteps.float().unsqueeze(1)
            * frequencies.unsqueeze(0)
        )

        sine_values = torch.sin(angles)
        cosine_values = torch.cos(angles)

        embeddings = torch.cat(
            (
                sine_values,
                cosine_values,
            ),
            dim=1,
        )

        return embeddings