"""Noise schedule, training loss and sampler of the conditional DDPM."""

import torch
import torch.nn.functional as F

from mri_diffusion import config


class Diffusion:
    def __init__(self, timesteps=config.TIMESTEPS, beta_start=config.BETA_START,
                 beta_end=config.BETA_END):
        # linear schedule from Ho et al. 2020. Nichol & Dhariwal (2021) only
        # found it worse at 64x64 and below, our slices are 256x256
        self.timesteps = timesteps
        betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)
        # cumprod in float64, then float32 because MPS has no float64
        self.alpha_bar = torch.cumprod(1.0 - betas, dim=0).float()

    def add_noise(self, x0, noise, t):
        """Sample x_t from q(x_t | x_0) in one step (Ho et al. 2020, eq. 4)."""
        a = self.alpha_bar.to(x0.device)[t].view(-1, 1, 1, 1)
        return a.sqrt() * x0 + (1.0 - a).sqrt() * noise

    def velocity(self, x0, noise, t):
        """Training target v = sqrt(a) * noise - sqrt(1 - a) * x0 (Salimans & Ho 2022, Sec. 4)."""
        a = self.alpha_bar.to(x0.device)[t].view(-1, 1, 1, 1)
        return a.sqrt() * noise - (1.0 - a).sqrt() * x0

    def loss(self, model, x0, condition, t=None, noise=None):
        """MSE between the true and the predicted v.

        The DDPM paper predicts the noise and gets the clean slice back as
        (x_t - sqrt(1 - a) * eps) / sqrt(a). Near the end of the chain a is
        almost 0, so any error in eps is multiplied by up to 157 and sampling
        starts from a useless estimate of x0. With v the clean slice is
        sqrt(a) * x_t - sqrt(1 - a) * v, which stays accurate at every step.
        t and noise are random unless given, validation fixes them.
        """
        if t is None:
            t = torch.randint(0, self.timesteps, (x0.shape[0],), device=x0.device)
        if noise is None:
            noise = torch.randn_like(x0)
        predicted = model(self.add_noise(x0, noise, t), condition, t)
        return F.mse_loss(predicted.float(), self.velocity(x0, noise, t))

    @torch.no_grad()
    def sample(self, model, condition, steps, eta=0.0, generator=None, amp=False):
        """Generate HR slices for a batch of conditions with DDIM (Song et al. 2021).

        steps picks evenly spaced timesteps out of the training schedule, so
        no retraining is needed for a shorter chain. eta=0 is deterministic
        DDIM. steps=TIMESTEPS with eta=1 is the ancestral DDPM sampler of
        Ho et al. 2020 with the posterior variance.
        """
        device = condition.device
        alpha_bar = self.alpha_bar.to(device)
        times = torch.linspace(0, self.timesteps - 1, steps).round().long().flip(0).tolist()
        # noise is drawn on the CPU so a seeded generator works on any device
        x = torch.randn(condition.shape, generator=generator).to(device)

        for i, t in enumerate(times):
            a_t = alpha_bar[t]
            a_prev = alpha_bar[times[i + 1]] if i + 1 < len(times) else torch.ones((), device=device)
            t_batch = torch.full((x.shape[0],), t, device=device, dtype=torch.long)
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                v = model(x, condition, t_batch).float()

            # clean slice from v, clipped to the data range, then the noise that
            # goes with it (the clipping is the same as in the guided-diffusion code)
            x0 = (a_t.sqrt() * x - (1.0 - a_t).sqrt() * v).clamp(-1.0, 1.0)
            eps = (x - a_t.sqrt() * x0) / (1.0 - a_t).sqrt()

            # DDIM update, eq. 12 with sigma from eq. 16 of Song et al. 2021
            sigma = eta * ((1.0 - a_prev) / (1.0 - a_t) * (1.0 - a_t / a_prev)).sqrt()
            x = a_prev.sqrt() * x0 + (1.0 - a_prev - sigma**2).clamp(min=0.0).sqrt() * eps
            if eta > 0 and i + 1 < len(times):
                x = x + sigma * torch.randn(x.shape, generator=generator).to(device)
        return x
