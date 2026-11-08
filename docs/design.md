# Design choices

This file explains what the code does and why, with the paper behind each
choice. Section numbers point to the place in the paper where the detail is.

## Task and data

| Choice | Value | Why | Source |
| --- | --- | --- | --- |
| Data | UltraCortex T1w, 9.4 T | Submillimetre HR images without labels. 86 images from 78 healthy adults, MP-RAGE and MP2RAGE at 0.6 to 0.8 mm | Mahler et al. 2025 |
| Split | by subject, 70/15/15, seed 23 | Some subjects have two scans, so splitting by volume or by slice would leak anatomy between train and test | |
| Slices | 30 central slices per orientation, 256 x 256 with zero padding | Same protocol as brain-mri-sr-baselines, so the numbers can be compared | |
| Intensity | 1st and 99th percentile of each volume mapped to 0 and 1, then to [-1, 1] for the network | The DDPM is defined on data scaled to [-1, 1] | Ho et al. 2020, Sec. 3.3 |
| Degradation | Gaussian blur (sigma 0.65), x2 downsampling with area averaging, Gaussian noise (sigma 0.02) | Same as the baselines. Training draws new noise every time a slice is read, validation and test use one fixed seed per slice | |

The slices are cached once by `prepare_data.py`. The old pipeline read the
whole NIfTI file again for every slice, which made training I/O bound.

## Model

| Choice | Value | Why | Source |
| --- | --- | --- | --- |
| Conditioning | bicubic upsampled LR stacked with the noisy slice as a second input channel | The simplest way to condition a diffusion U-Net on an image of the same size | SR3, Saharia et al. 2023, Sec. 2.4 |
| Target | predict v = sqrt(a) * noise - sqrt(1 - a) * x0, plain MSE | DDPM predicts the noise (L_simple) and gets the clean slice back as (x_t - sqrt(1 - a) * eps) / sqrt(a), so near t = 999 any error in eps is multiplied by 157. With v the clean slice is sqrt(a) * x_t - sqrt(1 - a) * v at every step. See "Why v and not the noise" below | Salimans & Ho 2022, Sec. 4 and App. D. Ho et al. 2022, Sec. 2.4. Ho et al. 2020, eq. 14 |
| Backbone | U-Net, 2 residual blocks per resolution, group normalization, SiLU | DDPM architecture | Ho et al. 2020, Sec. 4 and App. B |
| Width | 64 channels, multipliers (1, 2, 4, 4), so 256, 128, 64 and 32 pixel levels, 21.7M parameters | Smaller than the DDPM models (35.7M for CIFAR10, 114M for 256 x 256) because the training set is a few thousand slices and the GPU is a free Colab T4 | Ho et al. 2020, App. B |
| Timestep | sinusoidal embedding followed by a two-layer MLP | Same as DDPM, the Transformer position code | Ho et al. 2020, Sec. 4 |
| Timestep in the blocks | the embedding gives a scale and a shift after the second group norm (AdaGN) | Better FID than adding the embedding (13.06 against 15.08) and only one linear layer per block | Dhariwal & Nichol 2021, Sec. 3.1, Table 3 |
| Attention | multi-head self-attention, 64 channels per head, at 32 x 32 and in the middle block | Ho et al. used attention at 16 x 16. Later work found attention at 32, 16 and 8 with several heads better. 32 x 32 is the lowest level of this U-Net. At 64 x 64 every attention map would be 16 times bigger | Ho et al. 2020, Sec. 4. Nichol & Dhariwal 2021, App. A. Dhariwal & Nichol 2021, Sec. 3 |
| Dropout | 0.1 inside every residual block, before the last convolution | Ho et al. picked 0.1 on CIFAR10 after a sweep over 0.1 to 0.4. Nichol & Dhariwal found 0.1 best with the linear schedule. SR3 used 0.2 for its smallest data setting. The training set here is small, so overfitting is the main risk | Ho et al. 2020, App. B. Nichol & Dhariwal 2021, App. A. Saharia et al. 2023, Sec. 4 |

## Diffusion

| Choice | Value | Why | Source |
| --- | --- | --- | --- |
| Steps | T = 1000 | Standard value | Ho et al. 2020, Sec. 4 |
| Schedule | linear betas from 1e-4 to 0.02 | The cosine schedule only helped at 64 x 64 and below, the linear one works well for high resolution images and our slices are 256 x 256 | Ho et al. 2020, Sec. 4. Nichol & Dhariwal 2021, Sec. 3.2 |
| Sampler | DDIM with eta = 0, 50 steps for validation and 100 for the test set | Same trained model, 10 to 50 times faster than the 1000 step DDPM chain. A DDPM sampler for brain MRI took 20.3 s per slice against 1.4 s for a GAN in one of the reviewed studies | Song et al. 2021, abstract and Sec. 4.1. Chang et al. 2024 |
| Timesteps for DDIM | evenly spaced over the 1000 training steps | The "linear" choice of the DDIM paper | Song et al. 2021, App. D.2 |
| Full DDPM | `sample(..., steps=1000, eta=1)` | With eta = 1 and every step, the update is the ancestral DDPM sampler, so both can be compared with the same code | Song et al. 2021, eq. 16 |
| Clipping | the predicted clean slice is clipped to [-1, 1] and the noise is recomputed from it | Keeps the samples inside the data range, as in the guided-diffusion DDIM code | Dhariwal & Nichol 2021, public code |

## Training

| Choice | Value | Why | Source |
| --- | --- | --- | --- |
| Optimizer | Adam, learning rate 1e-4 | Used by SR3 and by most models in Dhariwal & Nichol | Saharia et al. 2023, Sec. 4. Dhariwal & Nichol 2021, App. H |
| Warm-up | linear over the first 1000 steps | SR3 warms up over 10k of its 1M steps, our runs are about 25 times shorter | Saharia et al. 2023, Sec. 4 |
| EMA | decay 0.999, every evaluation uses the EMA weights | Both DDPM papers use 0.9999 for runs of 500k steps or more. 0.999 averages roughly the last 1000 updates, which suits runs of tens of thousands of steps | Ho et al. 2020, App. B. Dhariwal & Nichol 2021, App. H |
| Batch | 8 slices, mixed precision on CUDA | Fits a 16 GB T4. Gradient clipping at norm 1 is kept from the previous setup | |
| Checkpoint | best psnr_brain on 16 fixed validation slices | The loss says little about the quality of the samples, so the checkpoint is picked on sampled images. The loss is still logged | |

## Why v and not the noise

The first version of this code predicted the noise, like DDPM and SR3. The
overfit check on synthetic slices (8 slices at 64 x 64, the small U-Net of
`overfit_check.py`, 3000 steps) showed where that goes wrong. A one step
estimate of x0 from a noisy slice was good at moderate noise (31 dB at
t = 400) and useless at the end of the chain (12.5 dB at t = 800, 5.4 dB at
t = 999), because there the error in eps is divided by sqrt(a), which is
0.039 at t = 800 and 0.0064 at t = 999. DDIM starts at t = 999, and its
samples ended at 15.5 dB against 26.2 dB for bicubic.

The same check with v as the target, nothing else changed, kept the one step
estimate between 36 and 40 dB at every t. Its samples passed bicubic after
500 steps and reached 39 dB after 3000. Salimans & Ho (2022) describe this
failure of noise prediction when the signal to noise ratio goes to zero, and
Imagen Video (Ho et al. 2022) uses v for all its models, the
super-resolution ones included.

A 64 x 64 slice is a harsh case for the end of the chain and 256 x 256
slices may suffer less, but v costs nothing here. The network and the
sampler stay the same and only one line of the loss changes.

## Evaluation

PSNR over the whole slice, PSNR inside the brain (HR above 0.05, which drops
the zero padding), SSIM (Wang et al. 2004) and MAE, defined exactly as in
brain-mri-sr-baselines. Bicubic and Lanczos are computed on the same LR slice
as the diffusion model, so the comparison is paired.

PSNR and SSIM favour blurry outputs. SR3 scored lower PSNR and SSIM than a
plain regression network while looking sharper (Saharia et al. 2023,
Sec. 4.2.1, Tables 1 and 2). A diffusion model that loses to bicubic on PSNR
can still be the better reconstruction, so the example figures and the error
maps matter as much as the table. The error maps are also where invented
structures would show up.

## Left out on purpose

- Learned variances and the hybrid loss (Nichol & Dhariwal 2021). They help
  when sampling with few steps from a DDPM chain, DDIM covers that here.
- BigGAN residual blocks for up and downsampling (Dhariwal & Nichol 2021) and
  skip connections rescaled by 1/sqrt(2) (Saharia et al. 2023). More code for
  a small gain, and Dhariwal & Nichol found the rescaling did not help.
- A 3D U-Net. Much more memory, and the tumour dataset it will be applied to
  is made of 2D images.
- Latent diffusion. It needs an autoencoder trained first, which is a second
  model to tune.

## References

- Chang, C.-W. et al. (2024). High-resolution MRI synthesis using a data-driven framework with denoising diffusion probabilistic modeling. Physics in Medicine & Biology, 69(4), 045001.
- Dhariwal, P., & Nichol, A. (2021). Diffusion models beat GANs on image synthesis. NeurIPS 34.
- Ho, J., Chan, W., Saharia, C., Whang, J., Gao, R., Gritsenko, A., Kingma, D. P., Poole, B., Norouzi, M., Fleet, D. J., & Salimans, T. (2022). Imagen Video: High definition video generation with diffusion models. arXiv:2210.02303.
- Ho, J., Jain, A., & Abbeel, P. (2020). Denoising diffusion probabilistic models. NeurIPS 33, 6840-6851.
- Mahler, L. et al. (2025). Submillimeter ultra-high field 9.4 T brain MR image collection and manual cortical segmentations. Scientific Data, 12, 635.
- Nichol, A., & Dhariwal, P. (2021). Improved denoising diffusion probabilistic models. ICML.
- Saharia, C., Ho, J., Chan, W., Salimans, T., Fleet, D. J., & Norouzi, M. (2023). Image super-resolution via iterative refinement. IEEE TPAMI, 45(4), 4713-4726.
- Salimans, T., & Ho, J. (2022). Progressive distillation for fast sampling of diffusion models. ICLR.
- Song, J., Meng, C., & Ermon, S. (2021). Denoising diffusion implicit models. ICLR.
- Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004). Image quality assessment: from error visibility to structural similarity. IEEE Transactions on Image Processing, 13(4), 600-612.
