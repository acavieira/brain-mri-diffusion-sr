import nibabel as nib
import numpy as np
import torch
from torch.utils.data import DataLoader

from mri_diffusion.dataset import (
    MRIDiffusionDataset,
    build_sample_index,
)


def create_test_dataset(tmp_path):
    path = (
        tmp_path
        / "data"
        / "sub-01_ses-1_T1w.nii"
    )

    data = np.arange(
        8 * 10 * 12,
        dtype=np.float32,
    ).reshape(8, 10, 12)

    image = nib.Nifti1Image(
        data,
        np.eye(4),
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    nib.save(
        image,
        str(path),
    )

    samples = build_sample_index(
        paths=[path],
        slices_per_orientation=2,
        random_seed=23,
    )

    return MRIDiffusionDataset(
        samples=samples,
        target_size=16,
        low_percentile=1,
        high_percentile=99,
        scale=2,
        blur_sigma=0.65,
        noise_sigma=0.02,
    )


def test_torch_dataset_returns_expected_tensors(
    tmp_path,
):
    dataset = create_test_dataset(tmp_path)

    first_item = dataset[0]
    repeated_item = dataset[0]

    assert len(dataset) == 6

    assert first_item["hr"].shape == (
        1,
        16,
        16,
    )
    assert first_item["lr"].shape == (
        1,
        8,
        8,
    )
    assert first_item["condition"].shape == (
        1,
        16,
        16,
    )

    assert first_item["hr"].dtype == torch.float32
    assert first_item["lr"].dtype == torch.float32
    assert (
        first_item["condition"].dtype
        == torch.float32
    )

    assert first_item["split"] == "unassigned"

    assert torch.equal(
        first_item["hr"],
        repeated_item["hr"],
    )
    assert torch.equal(
        first_item["lr"],
        repeated_item["lr"],
    )
    assert torch.equal(
        first_item["condition"],
        repeated_item["condition"],
    )


def test_dataloader_adds_batch_dimension(
    tmp_path,
):
    dataset = create_test_dataset(tmp_path)

    data_loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
    )

    batch = next(iter(data_loader))

    assert batch["hr"].shape == (
        2,
        1,
        16,
        16,
    )
    assert batch["lr"].shape == (
        2,
        1,
        8,
        8,
    )
    assert batch["condition"].shape == (
        2,
        1,
        16,
        16,
    )

    assert batch["slice_index"].shape == (2,)
    assert len(batch["sample_id"]) == 2