import numpy as np
import torch

from mri_diffusion.data import (SliceDataset, fit_to_square, normalize, split_subjects,
                                subject_of)
from mri_diffusion.degradation import degrade, make_condition


def test_split_has_no_subject_in_two_groups():
    subjects = [f"sub-{i:02d}" for i in range(78)]
    splits = split_subjects(subjects, 0.15, 0.15, seed=23)
    groups = [set(splits[name]) for name in ("train", "val", "test")]
    assert sum(len(g) for g in groups) == 78
    assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
    assert splits == split_subjects(subjects, 0.15, 0.15, seed=23)


def test_pilot_split_keeps_one_subject_for_val_and_test():
    splits = split_subjects([f"sub-{i:02d}" for i in range(10)], 0.15, 0.15, seed=23)
    assert [len(splits[s]) for s in ("train", "val", "test")] == [8, 1, 1]


def test_subject_id_comes_from_the_bids_path():
    assert subject_of("ds005216/sub-12/anat/sub-12_acq-mp2rage_T1w.nii.gz") == "sub-12"


def test_normalize_maps_to_zero_one():
    volume = np.random.default_rng(0).normal(100, 20, size=(10, 10, 10)).astype(np.float32)
    out = normalize(volume)
    assert out.min() == 0.0 and out.max() == 1.0


def test_fit_to_square_keeps_the_aspect_ratio():
    out = fit_to_square(np.ones((200, 100), dtype=np.float32), 256)
    assert out.shape == (256, 256)
    assert np.count_nonzero(out.any(axis=0)) == 128


def test_degradation_halves_the_size_and_repeats_with_the_same_seed():
    hr = np.random.default_rng(0).random((256, 256)).astype(np.float32)
    lr1 = degrade(hr, np.random.default_rng(5))
    lr2 = degrade(hr, np.random.default_rng(5))
    assert lr1.shape == (128, 128) and np.array_equal(lr1, lr2)
    assert make_condition(lr1, 256).shape == (256, 256)


def test_fixed_noise_dataset_gives_the_same_pair_twice(tmp_path):
    images = np.random.default_rng(0).random((4, 64, 64)).astype(np.float16)
    np.savez(tmp_path / "val.npz", images=images, subjects=np.array(["sub-1"] * 4),
             files=np.array(["a.nii.gz"] * 4), orientations=np.array(["axial"] * 4),
             indices=np.arange(4))
    dataset = SliceDataset(tmp_path / "val.npz", fixed_noise=True)
    hr, cond = dataset[2]
    assert hr.shape == (1, 64, 64) and torch.equal(cond, dataset[2][1])
    assert hr.min() >= -1 and hr.max() <= 1
