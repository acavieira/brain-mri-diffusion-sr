import importlib.util
from pathlib import Path

import numpy as np
import torch

from mri_diffusion.diffusion import Diffusion
from mri_diffusion.model import UNet

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "super_resolve_folder.py"
spec = importlib.util.spec_from_file_location("super_resolve_folder", SCRIPT)
sr_script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sr_script)


def test_only_images_below_the_target_size_are_enlarged():
    assert sr_script.needs_sr((225, 225), target=256)
    assert sr_script.needs_sr((201, 251), target=256)
    assert not sr_script.needs_sr((256, 200), target=256)
    assert not sr_script.needs_sr((512, 512), target=256)


def test_padding_makes_both_sides_divisible():
    padded = sr_script.pad_to_multiple(np.ones((201, 250), dtype=np.float32), multiple=8)
    assert padded.shape == (208, 256)
    assert padded[:201, :250].min() == 1.0 and padded[201:].max() == 0.0


def test_super_resolve_doubles_odd_sized_images():
    torch.manual_seed(0)
    model = UNet(base=32, mults=(1, 2, 4, 4), res_blocks=1, attention_res=(), head_channels=32,
                 dropout=0.0, image_size=64).eval()
    images = [np.random.default_rng(i).random((21, 30)).astype(np.float32) for i in range(2)]
    bicubic, sr = sr_script.super_resolve(model, Diffusion(), images, torch.device("cpu"),
                                          samples=2, steps=2)
    for b, s in zip(bicubic, sr):
        assert b.shape == s.shape == (42, 60)
        assert 0.0 <= s.min() and s.max() <= 1.0
