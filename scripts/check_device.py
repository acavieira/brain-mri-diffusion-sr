import sys
from pathlib import Path

import torch

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "src"),
)

from mri_diffusion.device import get_device


def main():
    device = get_device()

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    mps_backend = getattr(
        torch.backends,
        "mps",
        None,
    )

    if mps_backend is None:
        print("MPS built: False")
        print("MPS available: False")
    else:
        print(f"MPS built: {mps_backend.is_built()}")
        print(f"MPS available: {mps_backend.is_available()}")

    print(f"Selected device: {device}")

    tensor = torch.ones(
        (2, 2),
        dtype=torch.float32,
        device=device,
    )

    result = tensor + tensor

    print(f"Tensor device: {result.device}")
    print(f"Tensor result: {result.cpu()}")


if __name__ == "__main__":
    main()