import pytest
import torch

from mri_diffusion.unet import (
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