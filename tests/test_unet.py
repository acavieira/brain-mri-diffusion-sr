import pytest
import torch

from mri_diffusion.unet import (
    ConditionalInputBlock,
    Downsample,
    ResidualBlock,
    SinusoidalTimeEmbedding,
    TimeEmbedding,
    UNetBottleneck,
    UNetDecoder,
    UNetEncoder,
    Upsample,
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

def test_upsample_doubles_spatial_dimensions():
    upsample = Upsample(
        channels=64
    )

    image_features = torch.randn(
        (2, 64, 8, 8),
        dtype=torch.float32,
    )

    output = upsample(
        image_features
    )

    assert output.shape == (
        2,
        64,
        16,
        16,
    )


def test_upsample_has_expected_parameter_count():
    upsample = Upsample(
        channels=64
    )

    parameter_count = sum(
        parameter.numel()
        for parameter in upsample.parameters()
    )

    assert parameter_count == 36928


def test_upsample_receives_gradients():
    upsample = Upsample(
        channels=64
    )

    image_features = torch.randn(
        (2, 64, 8, 8),
        dtype=torch.float32,
        requires_grad=True,
    )

    output = upsample(
        image_features
    )

    loss = output.square().mean()
    loss.backward()

    assert image_features.grad is not None
    assert upsample.convolution.weight.grad is not None
    assert upsample.convolution.bias.grad is not None


def test_upsample_rejects_invalid_inputs():
    with pytest.raises(
        ValueError,
        match="positive",
    ):
        Upsample(
            channels=0
        )

    upsample = Upsample(
        channels=64
    )

    with pytest.raises(
        ValueError,
        match=r"shape \[B, C, H, W\]",
    ):
        upsample(
            torch.zeros((64, 8, 8))
        )

    with pytest.raises(
        ValueError,
        match="channel count",
    ):
        upsample(
            torch.zeros((2, 32, 8, 8))
        )

def create_conditional_input_block():
    return ConditionalInputBlock(
        image_channels=1,
        condition_channels=1,
        output_channels=64,
    )


def test_conditional_input_block_has_expected_shape():
    input_block = (
        create_conditional_input_block()
    )

    noisy_images = torch.randn(
        (2, 1, 16, 16),
        dtype=torch.float32,
    )

    conditions = torch.randn(
        (2, 1, 16, 16),
        dtype=torch.float32,
    )

    output = input_block(
        noisy_images,
        conditions,
    )

    assert output.shape == (
        2,
        64,
        16,
        16,
    )


def test_conditional_input_block_uses_condition():
    torch.manual_seed(23)

    input_block = (
        create_conditional_input_block()
    )

    noisy_images = torch.zeros(
        (1, 1, 8, 8),
        dtype=torch.float32,
    )

    first_condition = torch.zeros(
        (1, 1, 8, 8),
        dtype=torch.float32,
    )

    second_condition = torch.ones(
        (1, 1, 8, 8),
        dtype=torch.float32,
    )

    first_output = input_block(
        noisy_images,
        first_condition,
    )

    second_output = input_block(
        noisy_images,
        second_condition,
    )

    assert not torch.allclose(
        first_output,
        second_output,
    )


def test_conditional_input_block_parameter_count():
    input_block = (
        create_conditional_input_block()
    )

    parameter_count = sum(
        parameter.numel()
        for parameter in input_block.parameters()
    )

    assert parameter_count == 1216


def test_conditional_input_block_receives_gradients():
    input_block = (
        create_conditional_input_block()
    )

    noisy_images = torch.randn(
        (2, 1, 8, 8),
        dtype=torch.float32,
        requires_grad=True,
    )

    conditions = torch.randn(
        (2, 1, 8, 8),
        dtype=torch.float32,
        requires_grad=True,
    )

    output = input_block(
        noisy_images,
        conditions,
    )

    loss = output.square().mean()
    loss.backward()

    assert noisy_images.grad is not None
    assert conditions.grad is not None
    assert input_block.convolution.weight.grad is not None


def create_unet_encoder():
    return UNetEncoder(
        base_channels=64,
        time_embedding_dim=256,
        group_count=8,
    )


def test_unet_encoder_has_expected_shapes():
    encoder = create_unet_encoder()

    image_features = torch.randn(
        (2, 64, 32, 32),
        dtype=torch.float32,
    )

    time_embedding = torch.randn(
        (2, 256),
        dtype=torch.float32,
    )

    output, skip_connections = encoder(
        image_features,
        time_embedding,
    )

    assert output.shape == (
        2,
        256,
        4,
        4,
    )

    assert len(skip_connections) == 3

    assert skip_connections[0].shape == (
        2,
        64,
        32,
        32,
    )

    assert skip_connections[1].shape == (
        2,
        128,
        16,
        16,
    )

    assert skip_connections[2].shape == (
        2,
        256,
        8,
        8,
    )


def test_unet_encoder_receives_gradients():
    encoder = create_unet_encoder()

    image_features = torch.randn(
        (1, 64, 16, 16),
        dtype=torch.float32,
        requires_grad=True,
    )

    time_embedding = torch.randn(
        (1, 256),
        dtype=torch.float32,
        requires_grad=True,
    )

    output, skip_connections = encoder(
        image_features,
        time_embedding,
    )

    loss = output.square().mean()
    loss.backward()

    assert image_features.grad is not None
    assert time_embedding.grad is not None

    for parameter in encoder.parameters():
        assert parameter.grad is not None


def create_unet_bottleneck():
    return UNetBottleneck(
        channels=256,
        time_embedding_dim=256,
        group_count=8,
    )


def test_unet_bottleneck_preserves_shape():
    bottleneck = create_unet_bottleneck()

    image_features = torch.randn(
        (2, 256, 4, 4),
        dtype=torch.float32,
    )

    time_embedding = torch.randn(
        (2, 256),
        dtype=torch.float32,
    )

    output = bottleneck(
        image_features,
        time_embedding,
    )

    assert output.shape == (
        2,
        256,
        4,
        4,
    )


def test_unet_bottleneck_receives_gradients():
    bottleneck = create_unet_bottleneck()

    image_features = torch.randn(
        (1, 256, 4, 4),
        dtype=torch.float32,
        requires_grad=True,
    )

    time_embedding = torch.randn(
        (1, 256),
        dtype=torch.float32,
        requires_grad=True,
    )

    output = bottleneck(
        image_features,
        time_embedding,
    )

    loss = output.square().mean()
    loss.backward()

    assert image_features.grad is not None
    assert time_embedding.grad is not None

    for parameter in bottleneck.parameters():
        assert parameter.grad is not None


def create_unet_decoder():
    return UNetDecoder(
        base_channels=64,
        time_embedding_dim=256,
        group_count=8,
    )


def create_decoder_inputs():
    image_features = torch.randn(
        (1, 256, 2, 2),
        dtype=torch.float32,
    )

    time_embedding = torch.randn(
        (1, 256),
        dtype=torch.float32,
    )

    skip_connections = (
        torch.randn(
            (1, 64, 16, 16),
            dtype=torch.float32,
        ),
        torch.randn(
            (1, 128, 8, 8),
            dtype=torch.float32,
        ),
        torch.randn(
            (1, 256, 4, 4),
            dtype=torch.float32,
        ),
    )

    return (
        image_features,
        time_embedding,
        skip_connections,
    )


def test_unet_decoder_has_expected_shape():
    decoder = create_unet_decoder()

    (
        image_features,
        time_embedding,
        skip_connections,
    ) = create_decoder_inputs()

    output = decoder(
        image_features,
        time_embedding,
        skip_connections,
    )

    assert output.shape == (
        1,
        64,
        16,
        16,
    )


def test_unet_decoder_receives_gradients():
    decoder = create_unet_decoder()

    image_features = torch.randn(
        (1, 256, 2, 2),
        dtype=torch.float32,
        requires_grad=True,
    )

    time_embedding = torch.randn(
        (1, 256),
        dtype=torch.float32,
        requires_grad=True,
    )

    skip_connections = (
        torch.randn(
            (1, 64, 16, 16),
            dtype=torch.float32,
            requires_grad=True,
        ),
        torch.randn(
            (1, 128, 8, 8),
            dtype=torch.float32,
            requires_grad=True,
        ),
        torch.randn(
            (1, 256, 4, 4),
            dtype=torch.float32,
            requires_grad=True,
        ),
    )

    output = decoder(
        image_features,
        time_embedding,
        skip_connections,
    )

    loss = output.square().mean()
    loss.backward()

    assert image_features.grad is not None
    assert time_embedding.grad is not None

    for skip_connection in skip_connections:
        assert skip_connection.grad is not None

    for parameter in decoder.parameters():
        assert parameter.grad is not None


def test_unet_decoder_requires_three_skips():
    decoder = create_unet_decoder()

    image_features = torch.randn(
        (1, 256, 2, 2),
        dtype=torch.float32,
    )

    time_embedding = torch.randn(
        (1, 256),
        dtype=torch.float32,
    )

    with pytest.raises(
        ValueError,
        match="three skip connections",
    ):
        decoder(
            image_features,
            time_embedding,
            skip_connections=(),
        )