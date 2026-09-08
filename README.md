# Brain MRI Conditional Diffusion Super-Resolution

This repository contains simple research code for two-dimensional brain MRI super-resolution using a conditional diffusion model trained from scratch.

The repository is currently only a foundation. The model has not been implemented yet.

## Scope

This project will eventually:

1. Load high-resolution brain MRI volumes in NIfTI format.
2. Split volumes into training, validation, and test sets by subject.
3. Extract axial, coronal, and sagittal 2D slices.
4. Create controlled synthetic low-resolution images.
5. Use the degraded image as the condition of a DDPM.
6. Train a U-Net to predict the Gaussian noise added to the HR image.
7. Select the best checkpoint using validation data.
8. Generate super-resolved images for previously unseen test subjects.
9. Compare diffusion with Bicubic and Lanczos using identical test images.

Brain-tumour classification is outside the scope of this repository and will be developed in a separate downstream repository.

## Development principles

- Small files with one clear responsibility.
- Explicit imports and readable variable names.
- Documented tensor shapes and mathematical operations.
- One small Python `config.py`, added later.
- No complex configuration framework.
- Deterministic random seeds and dataset splits.
- Subject-level splitting before slice extraction.
- No data leakage between train, validation, and test.
- Validation selects the checkpoint.
- The test set is reserved for final evaluation.
- A small overfitting test must pass before full training.
- Inference initially uses the same diffusion schedule as training.

## Planned source structure

The following files are planned but are not created yet:

```text
src/mri_diffusion/
├── config.py
├── data.py
├── degradation.py
├── scheduler.py
├── unet.py
├── training.py
├── sampling.py
├── metrics.py
└── reporting.py
```

## Planned scripts

The following scripts are planned but are not created yet:

```text
scripts/
├── prepare_data.py
├── train.py
├── sample.py
└── evaluate.py
```

## Pipeline

```text
HR NIfTI volumes
    -> subject-level split
    -> HR slices
    -> synthetic LR condition
    -> conditional DDPM training
    -> validation checkpoint selection
    -> test-set sampling
    -> comparison with HR, Bicubic, and Lanczos
```

Classical interpolation baselines are maintained independently in the [brain-mri-sr-baselines repository](https://github.com/acavieira/brain-mri-sr-baselines). Both projects must eventually use identical preprocessing, degradation, splits, and metric definitions.

## Development order

1. Repository foundation.
2. Dataset discovery and subject-level split.
3. NIfTI loading and slice extraction.
4. Synthetic degradation and visual validation.
5. Forward diffusion scheduler and mathematical tests.
6. Conditional U-Net and tensor-shape tests.
7. Controlled overfitting on 4–16 slices.
8. Training and validation loops.
9. Reverse diffusion sampling.
10. Final comparison on unseen test subjects.

## Environment

Dependencies will be added after the Python, PyTorch, CUDA, and hardware environment have been confirmed.