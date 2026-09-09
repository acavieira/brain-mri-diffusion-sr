import torch

from mri_diffusion.scheduler import (
    DiffusionScheduler,
)
from mri_diffusion.training import (
    calculate_training_loss,
)
from mri_diffusion.unet import ConditionalUNet


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