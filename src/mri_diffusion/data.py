"""Find the UltraCortex volumes, split them by subject and cache the HR slices.

Reading a NIfTI file is slow, so scripts/prepare_data.py runs this once and
saves the slices of each split in one .npz file. Training then works from
memory.
"""

import re
from pathlib import Path

import cv2
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

from mri_diffusion import config
from mri_diffusion.degradation import degrade, make_condition


# array axis of each orientation after nibabel's RAS+ reorientation
AXIS = {"sagittal": 0, "coronal": 1, "axial": 2}


def find_volumes(data_dir):
    """Return the raw T1w NIfTI files, skipping anything under derivatives/."""
    paths = [
        path for path in Path(data_dir).rglob("*_T1w.nii*")
        if "derivatives" not in path.parts
    ]
    if not paths:
        raise FileNotFoundError(f"no *_T1w.nii or *_T1w.nii.gz files in {data_dir}")
    return sorted(paths)


def subject_of(path):
    match = re.search(r"sub-[A-Za-z0-9]+", str(path))
    if match is None:
        raise ValueError(f"no BIDS subject id in {path}")
    return match.group(0)


def split_subjects(subjects, val_fraction, test_fraction, seed):
    """Shuffle the subjects once and cut them into train, val and test.

    The split works on subjects, not on volumes or slices. Some UltraCortex
    subjects have two scans, and both must stay in the same split.
    """
    subjects = sorted(set(subjects))
    np.random.default_rng(seed).shuffle(subjects)
    n_val = max(1, int(len(subjects) * val_fraction))
    n_test = max(1, int(len(subjects) * test_fraction))
    if len(subjects) - n_val - n_test < 1:
        raise ValueError(f"{len(subjects)} subjects are not enough for three splits")
    return {
        "train": sorted(subjects[n_val + n_test:]),
        "val": sorted(subjects[n_test:n_test + n_val]),
        "test": sorted(subjects[:n_test]),
    }


def load_volume(path):
    """Read a NIfTI volume in RAS+ orientation as float32."""
    image = nib.as_closest_canonical(nib.load(str(path)))
    volume = image.get_fdata(dtype=np.float32)
    if volume.ndim != 3:
        raise ValueError(f"{path} is not a 3D volume, shape {volume.shape}")
    return volume


def normalize(volume, low=config.LOW_PERCENTILE, high=config.HIGH_PERCENTILE):
    """Map the low and high percentiles of the whole volume to 0 and 1."""
    finite = np.isfinite(volume)
    lo, hi = np.percentile(volume[finite], [low, high])
    out = np.zeros(volume.shape, dtype=np.float32)
    if hi > lo:
        out[finite] = (volume[finite] - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0)


def central_indices(length, count):
    start = (length - count) // 2
    return range(start, start + count)


def get_slice(volume, orientation, index):
    """Take one slice and flip it so the anatomy is upright, as in the baselines."""
    image = np.take(volume, index, axis=AXIS[orientation])
    return np.ascontiguousarray(np.flipud(image.T))


def fit_to_square(image, size):
    """Resize without distortion and pad with zeros to size x size."""
    height, width = image.shape
    factor = size / max(height, width)
    new_h, new_w = max(1, round(height * factor)), max(1, round(width * factor))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    out = np.zeros((size, size), dtype=np.float32)
    top, left = (size - new_h) // 2, (size - new_w) // 2
    out[top:top + new_h, left:left + new_w] = resized
    return out


def volume_slices(path):
    """Yield (orientation, index, HR slice) for the selected slices of one volume."""
    volume = normalize(load_volume(path))
    for orientation in config.ORIENTATIONS:
        length = volume.shape[AXIS[orientation]]
        for index in central_indices(length, config.SLICES_PER_ORIENTATION):
            image = fit_to_square(get_slice(volume, orientation, index), config.IMAGE_SIZE)
            yield orientation, index, image


def save_split(paths, out_file):
    """Save every selected slice of the given volumes in one .npz file."""
    images, subjects, files, orientations, indices = [], [], [], [], []
    for path in paths:
        for orientation, index, image in volume_slices(path):
            # float16 halves the file size and the rounding error (below 1e-3)
            # is far smaller than the noise added to the LR images
            images.append(image.astype(np.float16))
            subjects.append(subject_of(path))
            files.append(path.name)
            orientations.append(orientation)
            indices.append(index)
    np.savez_compressed(
        out_file,
        images=np.stack(images),
        subjects=np.array(subjects),
        files=np.array(files),
        orientations=np.array(orientations),
        indices=np.array(indices),
    )
    return len(images)


def to_model(image):
    """[0, 1] numpy image to a [-1, 1] tensor with one channel (Ho et al. 2020)."""
    return torch.from_numpy(image).unsqueeze(0) * 2.0 - 1.0


def to_image(tensor):
    """Inverse of to_model, back to a [0, 1] numpy image."""
    return ((tensor.squeeze(0).float().cpu().numpy() + 1.0) / 2.0).clip(0.0, 1.0)


class SliceDataset(Dataset):
    """HR slices with their bicubic condition, both in [-1, 1].

    With fixed_noise=False (training) every read draws new degradation noise,
    so the model rarely sees the same LR image twice. With fixed_noise=True
    (validation and test) slice i always gets the seed SEED + i, which keeps
    the LR input identical across runs and across methods.
    """

    def __init__(self, npz_file, fixed_noise):
        data = np.load(npz_file)
        self.images = data["images"]
        self.subjects = data["subjects"]
        self.files = data["files"]
        self.orientations = data["orientations"]
        self.indices = data["indices"]
        self.fixed_noise = fixed_noise

    def __len__(self):
        return len(self.images)

    def pair(self, i):
        """Return the HR slice, its LR version and the condition, all in [0, 1]."""
        hr = self.images[i].astype(np.float32)
        if self.fixed_noise:
            seed = config.SEED + i
        else:
            # taken from torch so each DataLoader worker gets different noise
            seed = int(torch.randint(0, 2**31 - 1, (1,)))
        lr = degrade(hr, np.random.default_rng(seed))
        return hr, lr, make_condition(lr, hr.shape[0])

    def __getitem__(self, i):
        hr, _, condition = self.pair(i)
        return to_model(hr), to_model(condition)
