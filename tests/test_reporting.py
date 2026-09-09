"""Save clean diffusion training results."""

import csv
from pathlib import Path


TRAINING_HISTORY_COLUMNS = (
    "epoch",
    "train_loss",
    "validation_loss",
    "gradient_norm",
    "best_checkpoint",
)


def save_training_history(
    path,
    history,
):
    """Save one row for each training epoch."""

    if not history:
        raise ValueError(
            "Training history cannot be empty"
        )

    history_path = Path(path)

    history_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with history_path.open(
        mode="w",
        newline="",
        encoding="utf-8",
    ) as history_file:
        writer = csv.DictWriter(
            history_file,
            fieldnames=TRAINING_HISTORY_COLUMNS,
        )

        writer.writeheader()
        writer.writerows(history)

    return history_path