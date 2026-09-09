import torch
import pytest

from mri_diffusion.scheduler import (
    DiffusionScheduler,
)
from mri_diffusion.training import (
    calculate_noise_loss,
    calculate_training_loss,
    train_one_batch,
    train_one_epoch,
    validate_one_epoch,
)
from mri_diffusion.unet import ConditionalUNet
from torch.utils.data import DataLoader

def create_small_model():
    return ConditionalUNet(
        image_channels=1,
        condition_channels=1,
        base_channels=8,
        time_embedding_dim=16,
        time_hidden_dim=32,
        group_count=4,
    )


def create_test_scheduler():
    return DiffusionScheduler(
        diffusion_steps=10,
        beta_start=0.0001,
        beta_end=0.02,
        device="cpu",
    )


def test_training_loss_is_finite_scalar():
    torch.manual_seed(23)

    model = create_small_model()
    scheduler = create_test_scheduler()

    hr_images = torch.rand(
        (2, 1, 16, 16),
        dtype=torch.float32,
    ) * 2.0 - 1.0

    conditions = torch.rand(
        (2, 1, 16, 16),
        dtype=torch.float32,
    ) * 2.0 - 1.0

    loss = calculate_training_loss(
        model=model,
        scheduler=scheduler,
        hr_images=hr_images,
        conditions=conditions,
    )

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert loss.item() >= 0.0


def test_training_loss_reaches_model_parameters():
    torch.manual_seed(23)

    model = create_small_model()
    scheduler = create_test_scheduler()

    hr_images = torch.rand(
        (1, 1, 16, 16),
        dtype=torch.float32,
    ) * 2.0 - 1.0

    conditions = torch.rand(
        (1, 1, 16, 16),
        dtype=torch.float32,
    ) * 2.0 - 1.0

    loss = calculate_training_loss(
        model=model,
        scheduler=scheduler,
        hr_images=hr_images,
        conditions=conditions,
    )

    loss.backward()

    for parameter in model.parameters():
        assert parameter.grad is not None

def test_train_one_batch_updates_model_parameters():
    torch.manual_seed(23)

    model = create_small_model()
    scheduler = create_test_scheduler()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.001,
        weight_decay=0.0,
    )

    hr_images = torch.rand(
        (1, 1, 16, 16),
        dtype=torch.float32,
    ) * 2.0 - 1.0

    conditions = torch.rand(
        (1, 1, 16, 16),
        dtype=torch.float32,
    ) * 2.0 - 1.0

    weight_before_training = (
        model
        .output_convolution
        .weight
        .detach()
        .clone()
    )

    loss, gradient_norm = train_one_batch(
        model=model,
        scheduler=scheduler,
        optimizer=optimizer,
        hr_images=hr_images,
        conditions=conditions,
        max_gradient_norm=1.0,
    )

    weight_after_training = (
        model
        .output_convolution
        .weight
        .detach()
    )

    assert loss >= 0.0
    assert gradient_norm >= 0.0

    assert not torch.equal(
        weight_before_training,
        weight_after_training,
    )


def test_train_one_batch_rejects_invalid_gradient_norm():
    model = create_small_model()
    scheduler = create_test_scheduler()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.001,
    )

    hr_images = torch.zeros(
        (1, 1, 16, 16),
        dtype=torch.float32,
    )

    conditions = torch.zeros_like(
        hr_images
    )

    with pytest.raises(
        ValueError,
        match="positive",
    ):
        train_one_batch(
            model=model,
            scheduler=scheduler,
            optimizer=optimizer,
            hr_images=hr_images,
            conditions=conditions,
            max_gradient_norm=0.0,
        )

def create_training_data_loader():
    samples = []

    for _ in range(4):
        samples.append(
            {
                "hr": (
                    torch.rand(
                        (1, 16, 16),
                        dtype=torch.float32,
                    )
                    * 2.0
                    - 1.0
                ),
                "condition": (
                    torch.rand(
                        (1, 16, 16),
                        dtype=torch.float32,
                    )
                    * 2.0
                    - 1.0
                ),
            }
        )

    return DataLoader(
        samples,
        batch_size=2,
        shuffle=False,
    )


def test_train_one_epoch_updates_model():
    torch.manual_seed(23)

    model = create_small_model()
    scheduler = create_test_scheduler()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.001,
    )

    data_loader = create_training_data_loader()

    weight_before_training = (
        model
        .output_convolution
        .weight
        .detach()
        .clone()
    )

    mean_loss, mean_gradient_norm = (
        train_one_epoch(
            model=model,
            scheduler=scheduler,
            optimizer=optimizer,
            data_loader=data_loader,
            device="cpu",
            max_gradient_norm=1.0,
        )
    )

    weight_after_training = (
        model
        .output_convolution
        .weight
        .detach()
    )

    assert mean_loss >= 0.0
    assert mean_gradient_norm >= 0.0

    assert not torch.equal(
        weight_before_training,
        weight_after_training,
    )


def test_train_one_epoch_rejects_empty_loader():
    model = create_small_model()
    scheduler = create_test_scheduler()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.001,
    )

    empty_loader = DataLoader(
        [],
        batch_size=2,
    )

    with pytest.raises(
        ValueError,
        match="empty",
    ):
        train_one_epoch(
            model=model,
            scheduler=scheduler,
            optimizer=optimizer,
            data_loader=empty_loader,
            device="cpu",
            max_gradient_norm=1.0,
        )

def test_validation_loss_is_deterministic():
    torch.manual_seed(23)

    model = create_small_model()
    scheduler = create_test_scheduler()
    data_loader = create_training_data_loader()

    first_loss = validate_one_epoch(
        model=model,
        scheduler=scheduler,
        data_loader=data_loader,
        device="cpu",
        random_seed=23,
    )

    second_loss = validate_one_epoch(
        model=model,
        scheduler=scheduler,
        data_loader=data_loader,
        device="cpu",
        random_seed=23,
    )

    assert first_loss == pytest.approx(
        second_loss,
        abs=0.0000001,
    )


def test_validation_does_not_change_model():
    torch.manual_seed(23)

    model = create_small_model()
    scheduler = create_test_scheduler()
    data_loader = create_training_data_loader()

    weight_before_validation = (
        model
        .output_convolution
        .weight
        .detach()
        .clone()
    )

    validation_loss = validate_one_epoch(
        model=model,
        scheduler=scheduler,
        data_loader=data_loader,
        device="cpu",
        random_seed=23,
    )

    weight_after_validation = (
        model
        .output_convolution
        .weight
        .detach()
    )

    assert validation_loss >= 0.0

    assert torch.equal(
        weight_before_validation,
        weight_after_validation,
    )

    for parameter in model.parameters():
        assert parameter.grad is None


def test_validation_rejects_empty_loader():
    model = create_small_model()
    scheduler = create_test_scheduler()

    empty_loader = DataLoader(
        [],
        batch_size=2,
    )

    with pytest.raises(
        ValueError,
        match="empty",
    ):
        validate_one_epoch(
            model=model,
            scheduler=scheduler,
            data_loader=empty_loader,
            device="cpu",
            random_seed=23,
        )