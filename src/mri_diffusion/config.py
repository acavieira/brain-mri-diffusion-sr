"""Every setting used by the scripts.

Change the values here instead of passing command-line arguments. The paths
and the number of training steps can also be set with environment variables,
which is easier on Colab where the data and the runs live on Google Drive.
"""

import os
from pathlib import Path


# Paths
# UltraCortex, OpenNeuro ds005216 version 1.0.0 (Mahler et al. 2025)
DATA_DIR = Path(os.environ.get("MRI_DATA_DIR", "data/ds005216"))
CACHE_DIR = Path(os.environ.get("MRI_CACHE_DIR", "data/cache"))
RUNS_DIR = Path(os.environ.get("MRI_RUNS_DIR", "runs"))
RUN_NAME = os.environ.get("MRI_RUN_NAME", "pilot")

# Subjects and splits
MAX_SUBJECTS = 10          # pilot run, use None for every subject
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15
SEED = 23

# Slices, same protocol as brain-mri-sr-baselines
ORIENTATIONS = ("axial", "coronal", "sagittal")
SLICES_PER_ORIENTATION = 30
IMAGE_SIZE = 256
LOW_PERCENTILE = 1.0
HIGH_PERCENTILE = 99.0

# Degradation, same values as brain-mri-sr-baselines
SCALE = 2
BLUR_SIGMA = 0.65
NOISE_SIGMA = 0.02
BRAIN_THRESHOLD = 0.05     # HR pixels above this count as brain in psnr_brain

# Diffusion schedule (Ho et al. 2020)
TIMESTEPS = 1000
BETA_START = 1e-4
BETA_END = 0.02

# U-Net
BASE_CHANNELS = 64
CHANNEL_MULTS = (1, 2, 4, 4)   # feature maps of 256, 128, 64 and 32 pixels
RES_BLOCKS = 2
ATTENTION_RESOLUTIONS = (32,)
HEAD_CHANNELS = 64
DROPOUT = 0.1

# Training
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
WARMUP_STEPS = 1000
TRAIN_STEPS = int(os.environ.get("MRI_TRAIN_STEPS", 40_000))   # the notebook sets 20k for the pilot
GRAD_CLIP = 1.0
EMA_DECAY = 0.999
USE_AMP = True             # mixed precision, only switched on with CUDA
NUM_WORKERS = 2            # 0 is safer on a Mac

# Validation and test sampling
VALIDATE_EVERY = 2000
VAL_SAMPLES = 16
VAL_SAMPLING_STEPS = 50
TEST_SAMPLING_STEPS = 100
