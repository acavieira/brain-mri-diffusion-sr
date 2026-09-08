# Decision Log

## Repository boundary

This repository contains only conditional diffusion super-resolution. Classical interpolation baselines and downstream tumour classification remain independent.

## Rewrite strategy

The old diffusion branch may be consulted as a technical reference, but its source code must not be copied directly.

The new implementation will use explicit imports and small modules with one responsibility.

## Initial method

- 2D conditional DDPM.
- Grayscale HR slice represented by `x_0`.
- Synthetically degraded and Bicubic-upscaled LR slice represented by condition `c`.
- A noisy HR image represented by `x_t`.
- The U-Net receives `(x_t, c, t)`.
- The U-Net predicts the Gaussian noise `epsilon`.
- Training uses mean squared error between true and predicted noise.

The forward diffusion and training loss are:

```text
x_t = sqrt(alpha_bar_t) * x_0
      + sqrt(1 - alpha_bar_t) * epsilon

loss = MSE(epsilon, predicted_epsilon)
```

## Data rules

- Split by subject or independent volume before extracting slices.
- A subject cannot appear in more than one split.
- Training data updates model weights.
- Validation data selects the best checkpoint.
- Test data is used only for final evaluation.

## Initial experimental values

These are starting values inherited from the previous experiment and can later be justified or changed.

| Parameter | Initial value |
| --- | --- |
| Image size | 256 × 256 |
| Scale | 2 |
| Gaussian blur sigma | 0.65 |
| LR noise sigma | 0.02 |
| Slices per orientation | 30 |
| Diffusion steps | 1000 |
| Beta start | 0.0001 |
| Beta end | 0.02 |
| Batch size | 4 |
| Learning rate | 0.0001 |
| Random seed | 23 |

## Configuration

Use one readable `config.py` with constants. Do not use YAML, Hydra, nested configuration classes, registries, or duplicated command-line arguments.

## Minimum pipeline test

Before full training, the model must be tested on 4–16 fixed slices.

The expected evidence is:

- Decreasing noise-prediction loss.
- Valid tensor dimensions.
- Progressively recognisable reconstructions.
- Increasing PSNR and SSIM on those fixed slices.

Passing this test validates the pipeline but does not prove generalisation.

## Sampling

The initial inference implementation must use the same 1000-step schedule used during training.

Do not allow a different number of inference steps by creating a new linear schedule. DDIM or timestep respacing may be implemented later as a separately tested improvement.

## Decisions still pending

- Final dataset directory and filename convention.
- Confirmation that each NIfTI volume represents an independent subject.
- Exact train, validation, and test subject identifiers.
- Python and PyTorch versions.
- CUDA version and available GPU.
- Final train, validation, and test proportions.