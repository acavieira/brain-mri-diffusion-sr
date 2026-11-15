"""Split the subjects and cache the HR slices of every split.

Run it once after downloading UltraCortex, and again only if the subjects or
the slice settings in config.py change:

    python scripts/prepare_data.py

Besides the .npz files it saves preview.png with a few training slices, so
orientation, intensity scaling and padding can be checked by eye.
"""

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mri_diffusion import config
from mri_diffusion.data import SliceDataset, find_volumes, save_split, split_subjects, subject_of


def save_preview(npz_file, out_file, count=6):
    dataset = SliceDataset(npz_file, fixed_noise=True)
    picks = np.linspace(0, len(dataset) - 1, count).round().astype(int)
    fig, axes = plt.subplots(2, count, figsize=(2.6 * count, 5.4))
    for col, i in enumerate(picks):
        hr, lr, _ = dataset.pair(i)
        title = f"{dataset.subjects[i]} {dataset.orientations[i]} {dataset.indices[i]}"
        for row, (image, label) in enumerate([(hr, "HR"), (lr, "LR")]):
            axes[row, col].imshow(image, cmap="gray", vmin=0, vmax=1)
            axes[row, col].set_title(f"{label} {title}", fontsize=7)
            axes[row, col].axis("off")
    fig.tight_layout()
    fig.savefig(out_file, dpi=110)
    plt.close(fig)


def main():
    paths = find_volumes(config.DATA_DIR)
    subjects = sorted({subject_of(path) for path in paths})
    if config.MAX_SUBJECTS is not None:
        subjects = subjects[:config.MAX_SUBJECTS]
    print(f"{len(paths)} volumes found, using {len(subjects)} subjects")

    splits = split_subjects(subjects, config.VAL_FRACTION, config.TEST_FRACTION, config.SEED)
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)

    with open(config.CACHE_DIR / "splits.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["subject", "split"])
        for split, ids in splits.items():
            writer.writerows([subject, split] for subject in ids)

    for split, ids in splits.items():
        files = [path for path in paths if subject_of(path) in ids]
        count = save_split(files, config.CACHE_DIR / f"{split}.npz")
        print(f"{split}: {len(ids)} subjects, {len(files)} volumes, {count} slices")

    save_preview(config.CACHE_DIR / "train.npz", config.CACHE_DIR / "preview.png")
    print(f"saved {config.CACHE_DIR / 'preview.png'}")


if __name__ == "__main__":
    main()
