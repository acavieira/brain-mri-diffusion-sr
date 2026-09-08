"""PyTorch device selection."""

import torch


def get_device():
    """Select CUDA, Apple MPS, or CPU."""

    if torch.cuda.is_available():
        return torch.device("cuda")

    mps_backend = getattr(
        torch.backends,
        "mps",
        None,
    )

    if (
        mps_backend is not None
        and mps_backend.is_available()
    ):
        return torch.device("mps")

    return torch.device("cpu")