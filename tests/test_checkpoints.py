import torch
from torch import nn

from mri_diffusion.checkpoints import (
    load_checkpoint,
    save_checkpoint,
)


def create_test_model_and_optimizer():
    model = nn.Linear(
        in_features=2,
        out_features=1,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.001,
    )

    return model, optimizer


def test_save_checkpoint_creates_file(tmp_path):
    model, optimizer = (
        create_test_model_and_optimizer()
    )

    checkpoint_path = (
        tmp_path / "checkpoints" / "best_model.pt"
    )

    save_checkpoint(
        path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        epoch=5,
        train_loss=0.4,
        validation_loss=0.5,
    )

    assert checkpoint_path.exists()

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    assert checkpoint["epoch"] == 5
    assert checkpoint["train_loss"] == 0.4
    assert checkpoint["validation_loss"] == 0.5
    assert "model_state_dict" in checkpoint
    assert "optimizer_state_dict" in checkpoint


def test_load_checkpoint_restores_model(tmp_path):
    torch.manual_seed(23)

    model, optimizer = (
        create_test_model_and_optimizer()
    )

    original_weight = (
        model.weight.detach().clone()
    )

    checkpoint_path = (
        tmp_path / "best_model.pt"
    )

    save_checkpoint(
        path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        epoch=3,
        train_loss=0.7,
        validation_loss=0.8,
    )

    with torch.no_grad():
        model.weight.add_(10.0)

    assert not torch.equal(
        model.weight,
        original_weight,
    )

    checkpoint = load_checkpoint(
        path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        device="cpu",
    )

    torch.testing.assert_close(
        model.weight,
        original_weight,
    )

    assert checkpoint["epoch"] == 3
    assert checkpoint["validation_loss"] == 0.8


def test_load_checkpoint_rejects_missing_file(
    tmp_path,
):
    model, optimizer = (
        create_test_model_and_optimizer()
    )

    missing_path = (
        tmp_path / "missing_model.pt"
    )

    try:
        load_checkpoint(
            path=missing_path,
            model=model,
            optimizer=optimizer,
            device="cpu",
        )
    except FileNotFoundError as error:
        assert "does not exist" in str(error)
    else:
        raise AssertionError(
            "Expected FileNotFoundError"
        )