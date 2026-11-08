# Test plan

Hardware: a Mac with an Apple chip and the free tier of Google Colab (one
T4 GPU, 16 GB, sessions that get cut after a few hours).

Short version: prepare the data and run the quick checks on the Mac, train
and evaluate on Colab. The Mac can train through MPS, but a 256 x 256
diffusion model needs tens of thousands of steps and a T4 with mixed
precision gets there faster without keeping the laptop busy for a day. The
first step on each machine measures the real speed, so this can be revised.

## 0. Setup on the Mac

```bash
git clone -b refactor https://github.com/acavieira/brain-mri-diffusion-sr.git
cd brain-mri-diffusion-sr
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest
```

All tests should pass in a few seconds. In `config.py` set `NUM_WORKERS = 0`
if the DataLoader complains on macOS.

## 1. Data for the pilot (Mac)

Put the dataset in `data/ds005216` with its BIDS layout
(`sub-XX/ses-Y/anat/sub-XX_ses-Y_T1w.nii.gz`, uncompressed `.nii` files
work too). For the pilot only the first 10 subjects are needed. If the
download is not finished yet, the AWS command line can fetch only the T1w
files:

```bash
aws s3 sync --no-sign-request s3://openneuro.org/ds005216 data/ds005216 \
    --exclude "*" --include "sub-*_T1w.nii.gz"
```

Then, with `MAX_SUBJECTS = 10` in `config.py`:

```bash
python scripts/prepare_data.py
```

With 10 subjects the split is 8 train, 1 validation and 1 test subject, so
about 720, 90 and 90 slices (a little more if one of them has two scans).
Open `data/cache/preview.png` and check that

- the slices are upright and centred,
- the brain fills most of the frame and the padding is black,
- MP2RAGE scans do not have a bright noisy background. If they do, the 0.05
  brain threshold of psnr_brain will also count background pixels, and it
  is better to know that now.

A first run on three volumes (sub-0, sub-45 and sub-56) worked and the
slices were upright and centred. sub-0 and sub-56, which look like MP2RAGE,
do have bright noise around the skull and in the neck. Bicubic gets 20.9 dB
psnr_brain on sub-56 against 25.9 dB on sub-45, mostly because of that
noise. It affects every method in the same way, so the comparison stays
fair, but keep the scan type in mind when comparing subjects.

## 2. Overfitting check (Mac)

```bash
python scripts/overfit_check.py
```

Eight slices at 64 x 64 and a small U-Net, a few minutes on the Mac. The
loss has to go down and the diffusion PSNR has to pass the bicubic value
printed at the start. On real UltraCortex slices it went from 18.4 dB at
step 250 to 29.4 dB at step 1500, against 20.6 dB for bicubic, and passed
bicubic between steps 500 and 750. If it stays far below bicubic,
something is broken and there is no point in going to Colab yet. Look at
`runs/overfit_check.png`.

## 3. Pilot training (Colab)

Copy `data/cache` to Google Drive (for example
`MyDrive/mri-diffusion/cache`) and open `notebooks/colab.ipynb`. The notebook
clones this branch, copies the cache to the Colab disk, keeps the runs on
Drive and calls `train.py`. When a session is cut, run the notebook again
and training continues from `last.pt`.

The settings cell of the notebook has `RUN_NAME = "pilot"` and
`TRAIN_STEPS = 20000`. Everything else comes from `config.py` as it is,
with a validation every 2000 steps.

The script prints the time per step every 100 steps. Write it down, it
decides how long the full run will take.

What to look at in `runs/pilot/log.csv`:

- `train_loss` going down quickly at the start and then slowly,
- `val_psnr_brain` going up and ending above the bicubic value printed at
  the start of training,
- `val_loss` and `val_psnr_brain` getting worse while `train_loss` still
  improves means overfitting. With 8 subjects this can happen. First try
  `DROPOUT = 0.2`, then more slices per orientation.

## 4. Pilot evaluation (Colab)

```bash
python scripts/evaluate.py
```

About 90 test slices, each sampled 4 times with 100 DDIM steps. Check

- `test/summary.csv`, where `diffusion` is one sample and `diffusion_mean`
  the mean of the four, against bicubic and Lanczos,
- `test/metrics.csv` split by orientation, in case one orientation is worse,
- `test/examples`, especially the error maps and the spread of the samples.
  Bright spots inside the brain in the error map are details the model got
  wrong or invented, and the spread shows where it was unsure.

A first CPU test on a subject the model never saw (sub-56, a 1.2M parameter
model at 128 x 128 trained for 2000 steps on sub-0 and sub-45) gave
18.7 dB psnr_brain for one sample, 20.2 dB for the mean of four and 19.7 dB
for bicubic. The mean beat bicubic on all 90 slices. One sample has a fine
grain that changes from sample to sample, and the mean removes most of it.
If evaluation takes too long on the full test set, `TEST_SAMPLES` in
`config.py` sets the number of samples.

The pilot is a check of the pipeline, not a result. One test subject is not
enough to say anything about generalisation.

## 5. Full run

When the pilot works:

1. finish the download and set `MAX_SUBJECTS = None` (56 train,
   11 validation and 11 test subjects),
2. run `prepare_data.py` again and copy the new cache to Drive,
3. in the notebook set `RUN_NAME = "full"`, so the pilot run is kept, and
   `TRAIN_STEPS` to 40000 to 60000, depending on the speed measured in the
   pilot,
4. train over as many Colab sessions as it takes,
5. evaluate on the test subjects.

The test split is only used once, at the end. Any choice of settings is made
on the validation split.

## 6. After that

The trained model will be applied to the tumour dataset. Before writing that
script, decide the image size that goes into the model and the size the
classifiers will see, so the original and super-resolved images reach the
classifiers in the same format.
