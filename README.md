# Brain MRI super-resolution with a conditional diffusion model

A 2D conditional DDPM that turns a degraded brain MRI slice into a slice with
twice the resolution. It is trained on the 9.4 T UltraCortex images and
compared with bicubic and Lanczos interpolation on subjects it never saw.

```text
UltraCortex T1w volumes
  -> split by subject (train / val / test)
  -> 30 central slices per orientation, 256 x 256, intensities in [0, 1]
  -> degradation: blur, x2 downsampling, Gaussian noise
  -> condition: bicubic upsampled LR slice
  -> U-Net learns to predict v, a mix of the added noise and the HR slice
  -> DDIM sampling on the test subjects
  -> PSNR, SSIM against HR, next to bicubic and Lanczos
```

Why each piece is the way it is, with the papers behind it, is in
[docs/design.md](docs/design.md). The order to run things in, on the Mac and
on Colab, is in [docs/test_plan.md](docs/test_plan.md).

## Layout

```text
src/mri_diffusion/
  config.py        every setting, no command-line arguments
  data.py          find volumes, split by subject, cache slices, Dataset
  degradation.py   blur, downsampling and noise, same as the baselines
  model.py         conditional U-Net with dropout and self-attention
  diffusion.py     noise schedule, training loss, DDIM / DDPM sampler
  metrics.py       PSNR and SSIM, same definitions as the baselines
  utils.py         device, seeds, model from config, EMA
scripts/
  prepare_data.py  run once, writes data/cache/{train,val,test}.npz
  overfit_check.py learn 8 slices at 64 x 64, checks the pipeline
  train.py         training with EMA, validation and resumable checkpoints
  evaluate.py      test set, metrics per slice and per method, figures
notebooks/
  colab.ipynb      training and evaluation on a Colab GPU
tests/             short pytest checks
docs/              design choices and test plan
```

## Running

```bash
pip install -r requirements.txt
python -m pytest
python scripts/prepare_data.py
python scripts/overfit_check.py
python scripts/train.py
python scripts/evaluate.py
```

The dataset goes in `data/ds005216` with its BIDS folders (OpenNeuro
ds005216, version 1.0.0). `MAX_SUBJECTS` in `config.py` is 10 for the pilot,
set it to `None` to use every subject. Each run writes to `runs/<RUN_NAME>/`:

```text
config.json      settings used
log.csv          losses, validation PSNR and SSIM over training
last.pt          everything needed to resume
best.pt          EMA weights with the best validation psnr_brain
test/            metrics.csv, summary.csv, sr.npz and example figures
```

## Related repositories

- [brain-mri-sr-baselines](https://github.com/acavieira/brain-mri-sr-baselines)
  uses the same normalization, slices, degradation and metrics. `evaluate.py`
  also computes bicubic and Lanczos on exactly the same LR slices, so the
  comparison inside this repository is paired slice by slice.
- [brain-mri-few-shot-learning](https://github.com/acavieira/brain-mri-few-shot-learning)
  holds the tumour classifiers that will later receive the super-resolved
  images.
