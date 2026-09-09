"""Save and restore diffusion training checkpoints."""

from pathlib import Path

import torch


def save_checkpoint(
    path,
    model,
    optimizer,
    epoch,
    train_loss,
    validation_loss,
):
    """Save the complete training state."""

    checkpoint_path = Path(path)

    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = {
        "epoch": epoch,
        "train_loss": train_loss,
        "validation_loss": validation_loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }

    torch.save(
        checkpoint,
        checkpoint_path,
    )


def load_checkpoint(
    path,
    model,
    optimizer=None,
    device="cpu",
):
    """Restore a previously saved training state."""

    checkpoint_path = Path(path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint does not exist: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    if optimizer is not None:
        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

    return checkpoint