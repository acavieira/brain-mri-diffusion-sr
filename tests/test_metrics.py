import math

import numpy as np

from mri_diffusion.metrics import compare, psnr


def test_psnr_of_identical_images_is_infinite():
    image = np.random.default_rng(0).random((32, 32))
    assert psnr(image, image) == math.inf


def test_psnr_matches_the_formula():
    hr = np.zeros((10, 10))
    pred = np.full((10, 10), 0.1)    # mse 0.01, so 20 dB
    assert abs(psnr(hr, pred) - 20.0) < 1e-9


def test_brain_psnr_ignores_the_padding():
    hr = np.zeros((20, 20))
    hr[5:15, 5:15] = 0.5
    pred = hr.copy()
    pred[0, 0] = 1.0                 # error only in the background
    scores = compare(hr, pred, brain_threshold=0.05)
    assert scores["psnr_brain"] == math.inf and scores["psnr"] < math.inf
