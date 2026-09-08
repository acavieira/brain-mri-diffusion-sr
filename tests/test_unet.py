import pytest
import torch

from mri_diffusion.unet import (
    Downsample,
    ResidualBlock,
    SinusoidalTimeEmbedding,
    TimeEmbedding,
)

def create_time_embedding():
    return SinusoidalTimeEmbedding(
        embedding_dim=256
    )


def test_time_embedding_has_expected_shape():
    time_embedding = create_time_embedding()

    timesteps = torch.tensor(
        [0, 99, 499, 999],
        dtype=torch.long,
    )

    embeddings = time_embedding(timesteps)

    assert embeddings.shape == (4, 256)
    assert embeddings.dtype == torch.float32

    assert embeddings.min() >= -1.0
    assert embeddings.max() <= 1.0


def test_zero_timestep_has_expected_values():
    time_embedding = create_time_embedding()

    timesteps = torch.tensor(
        [0],
        dtype=torch.long,
    )

    embeddings = time_embedding(timesteps)

    sine_values = embeddings[:, :128]
    cosine_values = embeddings[:, 128:]

    torch.testing.assert_close(
        sine_values,
        torch.zeros_like(sine_values),
    )

    torch.testing.assert_close(
        cosine_values,
        torch.ones_like(cosine_values),
    )


def test_different_timesteps_have_different_embeddings():
    time_embedding = create_time_embedding()

    timesteps = torch.tensor(
        [10, 500],
        dtype=torch.long,
    )

    embeddings = time_embedding(timesteps)

    assert not torch.equal(
        embeddings[0],
        embeddings[1],
    )


def test_same_timestep_has_same_embedding():
    time_embedding = create_time_embedding()

    timesteps = torch.tensor(
        [250, 250],
        dtype=torch.long,
    )

    embeddings = time_embedding(timesteps)

    torch.testing.assert_close(
        embeddings[0],
        embeddings[1],
    )


def test_time_embedding_rejects_invalid_inputs():
    with pytest.raises(
        ValueError,
        match="positive",
    ):
        SinusoidalTimeEmbedding(
            embedding_dim=0
        )

    with pytest.raises(
        ValueError,
        match="even",
    ):
        SinusoidalTimeEmbedding(
            embedding_dim=255
        )

    time_embedding = create_time_embedding()

    with pytest.raises(
        ValueError,
        match=r"shape \[B\]",
    ):
        time_embedding(
            torch.zeros(
                (2, 1),
                dtype=torch.long,
            )
        )

def create_trainable_time_embedding():
    return TimeEmbedding(
        embedding_dim=256,
        hidden_dim=512,
    )


def test_trainable_time_embedding_has_expected_shape():
    time_embedding = (
        create_trainable_time_embedding()
    )

    timesteps = torch.tensor(
        [0, 99, 499, 999],
        dtype=torch.long,
    )

    embeddings = time_embedding(
        timesteps
    )

    assert embeddings.shape == (4, 256)
    assert embeddings.dtype == torch.float32


def test_trainable_time_embedding_parameter_count():
    time_embedding = (
        create_trainable_time_embedding()
    )

    parameter_count = sum(
        parameter.numel()
        for parameter in time_embedding.parameters()
    )

    assert parameter_count == 262912


def test_trainable_time_embedding_receives_gradients():
    time_embedding = (
        create_trainable_time_embedding()
    )

    timesteps = torch.tensor(
        [10, 500],
        dtype=torch.long,
    )

    embeddings = time_embedding(
        timesteps
    )

    loss = embeddings.square().mean()
    loss.backward()

    for parameter in time_embedding.parameters():
        assert parameter.grad is not None


def test_trainable_time_embedding_rejects_invalid_hidden_dimension():
    with pytest.raises(
        ValueError,
        match="positive",
    ):
        TimeEmbedding(
            embedding_dim=256,
            hidden_dim=0,
        )

def create_residual_block(
    input_channels=64,
    output_channels=64,
):
    return ResidualBlock(
        input_channels=input_channels,
        output_channels=output_channels,
        time_embedding_dim=256,
        group_count=8,
    )


def test_residual_block_preserves_image_shape():
    residual_block = create_residual_block()

    image_features = torch.randn(
        (2, 64, 16, 16),
        dtype=torch.float32,
    )

    time_embedding = torch.randn(
        (2, 256),
        dtype=torch.float32,
    )

    output = residual_block(
        image_features,
        time_embedding,
    )

    assert output.shape == (
        2,
        64,
        16,
        16,
    )


def test_residual_block_changes_channel_count():
    residual_block = create_residual_block(
        input_channels=64,
        output_channels=128,
    )

    image_features = torch.randn(
        (2, 64, 16, 16),
        dtype=torch.float32,
    )

    time_embedding = torch.randn(
        (2, 256),
        dtype=torch.float32,
    )

    output = residual_block(
        image_features,
        time_embedding,
    )

    assert output.shape == (
        2,
        128,
        16,
        16,
    )


def test_residual_block_uses_time_embedding():
    torch.manual_seed(23)

    residual_block = create_residual_block()

    image_features = torch.zeros(
        (1, 64, 8, 8),
        dtype=torch.float32,
    )

    first_time = torch.zeros(
        (1, 256),
        dtype=torch.float32,
    )

    second_time = torch.ones(
        (1, 256),
        dtype=torch.float32,
    )

    first_output = residual_block(
        image_features,
        first_time,
    )

    second_output = residual_block(
        image_features,
        second_time,
    )

    assert not torch.allclose(
        first_output,
        second_output,
    )


def test_residual_block_receives_gradients():
    residual_block = create_residual_block()

    image_features = torch.randn(
        (2, 64, 8, 8),
        dtype=torch.float32,
        requires_grad=True,
    )

    time_embedding = torch.randn(
        (2, 256),
        dtype=torch.float32,
        requires_grad=True,
    )

    output = residual_block(
        image_features,
        time_embedding,
    )

    loss = output.square().mean()
    loss.backward()

    assert image_features.grad is not None
    assert time_embedding.grad is not None

    for parameter in residual_block.parameters():
        assert parameter.grad is not None

def test_downsample_halves_spatial_dimensions():
    downsample = Downsample(
        channels=64
    )

    image_features = torch.randn(
        (2, 64, 16, 16),
        dtype=torch.float32,
    )

    output = downsample(
        image_features
    )

    assert output.shape == (
        2,
        64,
        8,
        8,
    )


def test_downsample_has_expected_parameter_count():
    downsample = Downsample(
        channels=64
    )

    parameter_count = sum(
        parameter.numel()
        for parameter in downsample.parameters()
    )

    assert parameter_count == 36928


def test_downsample_receives_gradients():
    downsample = Downsample(
        channels=64
    )

    image_features = torch.randn(
        (2, 64, 16, 16),
        dtype=torch.float32,
        requires_grad=True,
    )

    output = downsample(
        image_features
    )

    loss = output.square().mean()
    loss.backward()

    assert image_features.grad is not None
    assert downsample.convolution.weight.grad is not None
    assert downsample.convolution.bias.grad is not None


def test_downsample_rejects_invalid_inputs():
    with pytest.raises(
        ValueError,
        match="positive",
    ):
        Downsample(
            channels=0
        )

    downsample = Downsample(
        channels=64
    )

    with pytest.raises(
        ValueError,
        match=r"shape \[B, C, H, W\]",
    ):
        downsample(
            torch.zeros((64, 16, 16))
        )

    with pytest.raises(
        ValueError,
        match="channel count",
    ):
        downsample(
            torch.zeros((2, 32, 16, 16))
        )