import torch

import mri_diffusion.device as device_module


def test_get_device_prefers_cuda(monkeypatch):
    monkeypatch.setattr(
        device_module.torch.cuda,
        "is_available",
        lambda: True,
    )
    monkeypatch.setattr(
        device_module.torch.backends.mps,
        "is_available",
        lambda: True,
    )

    device = device_module.get_device()

    assert device == torch.device("cuda")


def test_get_device_uses_mps_without_cuda(
    monkeypatch,
):
    monkeypatch.setattr(
        device_module.torch.cuda,
        "is_available",
        lambda: False,
    )
    monkeypatch.setattr(
        device_module.torch.backends.mps,
        "is_available",
        lambda: True,
    )

    device = device_module.get_device()

    assert device == torch.device("mps")


def test_get_device_uses_cpu_as_fallback(
    monkeypatch,
):
    monkeypatch.setattr(
        device_module.torch.cuda,
        "is_available",
        lambda: False,
    )
    monkeypatch.setattr(
        device_module.torch.backends.mps,
        "is_available",
        lambda: False,
    )

    device = device_module.get_device()

    assert device == torch.device("cpu")