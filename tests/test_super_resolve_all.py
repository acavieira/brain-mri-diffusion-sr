import csv
import importlib.util
from pathlib import Path

import cv2
import numpy as np
import torch

from mri_diffusion.degradation import make_condition
from mri_diffusion.diffusion import Diffusion
from mri_diffusion.model import UNet

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "super_resolve_all.py"
spec = importlib.util.spec_from_file_location("super_resolve_all", SCRIPT)
sr_all = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sr_all)


def test_every_image_becomes_a_square_hr_and_a_half_size_lr():
    rng = np.random.default_rng(0)
    for shape in [(300, 400), (100, 120), (512, 512), (240, 300)]:
        hr = sr_all.to_hr(rng.random(shape).astype(np.float32))
        lr = sr_all.make_lr(hr, Path("train/glioma/a.jpg"))
        assert hr.shape == (256, 256) and lr.shape == (128, 128)
        assert 0.0 <= hr.min() and hr.max() <= 1.0 and 0.0 <= lr.min() and lr.max() <= 1.0


def test_the_lr_noise_depends_only_on_the_file_name():
    hr = np.full((256, 256), 0.5, dtype=np.float32)
    a = sr_all.make_lr(hr, Path("train/glioma/a.jpg"))
    assert np.array_equal(a, sr_all.make_lr(hr, Path("train/glioma/a.jpg")))
    assert not np.array_equal(a, sr_all.make_lr(hr, Path("train/glioma/b.jpg")))


def test_diffusion_mean_keeps_the_size():
    torch.manual_seed(0)
    model = UNet(base=32, mults=(1, 2, 4, 4), res_blocks=1, attention_res=(), head_channels=32,
                 dropout=0.0, image_size=64).eval()
    bicubic = [make_condition(np.random.default_rng(i).random((32, 32)).astype(np.float32), 64) for i in range(2)]
    out = sr_all.diffusion_mean(model, Diffusion(), bicubic, torch.device("cpu"), samples=2, steps=2)
    assert [o.shape for o in out] == [(64, 64), (64, 64)]
    assert all(0.0 <= o.min() and o.max() <= 1.0 for o in out)


def test_metrics_mark_images_without_real_detail_at_256():
    hr = sr_all.to_8bit(np.random.default_rng(1).random((256, 256)))
    saved = {"hr": hr, "lr": hr[::2, ::2], "bicubic": hr, "diffusion": hr}
    big = sr_all.metric_rows(Path("test/no_tumor/x.jpg"), (300, 400), saved)
    small = sr_all.metric_rows(Path("test/no_tumor/y.jpg"), (200, 252), saved)
    assert [r[4] for r in big] == ["bicubic", "diffusion"]
    assert big[0][3] == 1 and small[0][3] == 0
    assert big[0][7] == 1.0                      # identical images have an SSIM of 1


def test_metrics_are_filled_for_images_saved_without_their_row(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    for name in ("a", "b"):
        path = src / "train" / "glioma" / f"{name}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), (np.random.default_rng(len(name)).random((300, 300)) * 255).astype(np.uint8))
    rel = Path("train/glioma/a.jpg")
    for version in sr_all.VERSIONS:
        out = sr_all.out_path(dst, version, rel)
        out.parent.mkdir(parents=True, exist_ok=True)
        size = 128 if version == "lr" else 256
        cv2.imwrite(str(out), np.full((size, size), 120, dtype=np.uint8))
    metrics = dst / "metrics.csv"
    assert sr_all.is_done(dst, rel) and not sr_all.is_done(dst, Path("train/glioma/b.jpg"))
    sr_all.fill_missing_metrics(src, dst, metrics, [src / rel])
    sr_all.fill_missing_metrics(src, dst, metrics, [src / rel])      # a second run adds nothing
    with open(metrics, newline="") as f:
        rows = list(csv.DictReader(f))
    assert [(r["image"], r["method"]) for r in rows] == [(str(rel), "bicubic"), (str(rel), "diffusion")]
