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


class TimeEmbedding(nn.Module):
    """Create a trainable representation of each timestep."""

    def __init__(
        self,
        embedding_dim,
        hidden_dim,
    ):
        super().__init__()

        if hidden_dim <= 0:
            raise ValueError(
                "Hidden dimension must be positive"
            )

        self.sinusoidal_embedding = (
            SinusoidalTimeEmbedding(
                embedding_dim=embedding_dim
            )
        )

        self.projection = nn.Sequential(
            nn.Linear(
                embedding_dim,
                hidden_dim,
            ),
            nn.SiLU(),
            nn.Linear(
                hidden_dim,
                embedding_dim,
            ),
        )

    def forward(self, timesteps):
        fixed_embedding = (
            self.sinusoidal_embedding(
                timesteps
            )
        )

        learned_embedding = self.projection(
            fixed_embedding
        )

        return learned_embedding


class ResidualBlock(nn.Module):
    """Process image features and add timestep information."""

    def __init__(
        self,
        input_channels,
        output_channels,
        time_embedding_dim,
        group_count,
    ):
        super().__init__()

        if input_channels <= 0 or output_channels <= 0:
            raise ValueError(
                "Channel counts must be positive"
            )

        if group_count <= 0:
            raise ValueError(
                "Group count must be positive"
            )

        if input_channels % group_count != 0:
            raise ValueError(
                "Input channels must be divisible by group count"
            )

        if output_channels % group_count != 0:
            raise ValueError(
                "Output channels must be divisible by group count"
            )

        self.time_embedding_dim = time_embedding_dim

        self.first_normalization = nn.GroupNorm(
            num_groups=group_count,
            num_channels=input_channels,
        )

        self.first_convolution = nn.Conv2d(
            in_channels=input_channels,
            out_channels=output_channels,
            kernel_size=3,
            padding=1,
        )

        self.time_projection = nn.Linear(
            time_embedding_dim,
            output_channels,
        )

        self.second_normalization = nn.GroupNorm(
            num_groups=group_count,
            num_channels=output_channels,
        )

        self.second_convolution = nn.Conv2d(
            in_channels=output_channels,
            out_channels=output_channels,
            kernel_size=3,
            padding=1,
        )

        self.activation = nn.SiLU()

        if input_channels == output_channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Conv2d(
                in_channels=input_channels,
                out_channels=output_channels,
                kernel_size=1,
            )

    def forward(
        self,
        image_features,
        time_embedding,
    ):
        if image_features.ndim != 4:
            raise ValueError(
                "Image features must have shape [B, C, H, W]"
            )

        if time_embedding.ndim != 2:
            raise ValueError(
                "Time embedding must have shape [B, D]"
            )

        if image_features.shape[0] != time_embedding.shape[0]:
            raise ValueError(
                "Image features and time embedding "
                "must have the same batch size"
            )

        if time_embedding.shape[1] != self.time_embedding_dim:
            raise ValueError(
                "Time embedding has an unexpected dimension"
            )

        residual = self.skip_connection(
            image_features
        )

        hidden = self.first_normalization(
            image_features
        )
        hidden = self.activation(hidden)
        hidden = self.first_convolution(hidden)

        time_features = self.time_projection(
            time_embedding
        )

        time_features = time_features[
            :,
            :,
            None,
            None,
        ]

        hidden = hidden + time_features

        hidden = self.second_normalization(hidden)
        hidden = self.activation(hidden)
        hidden = self.second_convolution(hidden)

        output = hidden + residual

        return output

class Downsample(nn.Module):
    """Reduce feature height and width by a factor of two."""

    def __init__(self, channels):
        super().__init__()

        if channels <= 0:
            raise ValueError(
                "Channel count must be positive"
            )

        self.channels = channels

        self.convolution = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=2,
            padding=1,
        )

    def forward(self, image_features):
        if image_features.ndim != 4:
            raise ValueError(
                "Image features must have shape [B, C, H, W]"
            )

        if image_features.shape[1] != self.channels:
            raise ValueError(
                "Image features have an unexpected channel count"
            )

        output = self.convolution(
            image_features
        )

        return output

class Upsample(nn.Module):
    """Double feature height and width."""

    def __init__(self, channels):
        super().__init__()

        if channels <= 0:
            raise ValueError(
                "Channel count must be positive"
            )

        self.channels = channels

        self.resize = nn.Upsample(
            scale_factor=2,
            mode="nearest",
        )

        self.convolution = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            padding=1,
        )

    def forward(self, image_features):
        if image_features.ndim != 4:
            raise ValueError(
                "Image features must have shape [B, C, H, W]"
            )

        if image_features.shape[1] != self.channels:
            raise ValueError(
                "Image features have an unexpected channel count"
            )

        resized_features = self.resize(
            image_features
        )

        output = self.convolution(
            resized_features
        )

        return output

class ConditionalInputBlock(nn.Module):
    """Join the noisy image and LR condition."""

    def __init__(
        self,
        image_channels,
        condition_channels,
        output_channels,
    ):
        super().__init__()

        if image_channels <= 0:
            raise ValueError(
                "Image channel count must be positive"
            )

        if condition_channels <= 0:
            raise ValueError(
                "Condition channel count must be positive"
            )

        if output_channels <= 0:
            raise ValueError(
                "Output channel count must be positive"
            )

        self.image_channels = image_channels
        self.condition_channels = condition_channels

        combined_channels = (
            image_channels + condition_channels
        )

        self.convolution = nn.Conv2d(
            in_channels=combined_channels,
            out_channels=output_channels,
            kernel_size=3,
            padding=1,
        )

    def forward(
        self,
        noisy_images,
        conditions,
    ):
        if noisy_images.ndim != 4:
            raise ValueError(
                "Noisy images must have shape [B, C, H, W]"
            )

        if conditions.ndim != 4:
            raise ValueError(
                "Conditions must have shape [B, C, H, W]"
            )

        if noisy_images.shape[0] != conditions.shape[0]:
            raise ValueError(
                "Noisy images and conditions "
                "must have the same batch size"
            )

        if noisy_images.shape[2:] != conditions.shape[2:]:
            raise ValueError(
                "Noisy images and conditions "
                "must have the same spatial dimensions"
            )

        if noisy_images.shape[1] != self.image_channels:
            raise ValueError(
                "Noisy images have an unexpected channel count"
            )

        if conditions.shape[1] != self.condition_channels:
            raise ValueError(
                "Conditions have an unexpected channel count"
            )

        combined = torch.cat(
            (
                noisy_images,
                conditions,
            ),
            dim=1,
        )

        output = self.convolution(
            combined
        )

        return output