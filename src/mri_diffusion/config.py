from pathlib import Path


# Dataset:
# https://openneuro.org/datasets/ds005216/versions/1.0.0

DATA_DIR = Path("data")
SPLIT_FILE = Path("splits/subject_splits.csv")

DATASET_NAME = "UltraCortex"
DATASET_ACCESSION = "ds005216"
DATASET_VERSION = "1.0.0"

ORIENTATIONS = (
    "axial",
    "coronal",
    "sagittal",
)

CANONICAL_AXIS_BY_ORIENTATION = {
    "sagittal": 0,
    "coronal": 1,
    "axial": 2,
}

SLICES_PER_ORIENTATION = 30
TARGET_SIZE = 256

LOW_PERCENTILE = 1.0
HIGH_PERCENTILE = 99.0

SCALE = 2
BLUR_SIGMA = 0.65
NOISE_SIGMA = 0.02

DIFFUSION_STEPS = 1000
BETA_START = 0.0001
BETA_END = 0.02
TIME_EMBEDDING_DIM = 256
TIME_HIDDEN_DIM = 512

TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15
TEST_FRACTION = 0.15

RANDOM_SEED = 23