"""Forward diffusion noise scheduler."""

import torch


class DiffusionScheduler:
    """Store the DDPM noise schedule and add noise to HR images."""

    def __init__(
        self,
        diffusion_steps,
        beta_start,
        beta_end,
        device,
    ):
        if diffusion_steps <= 0:
            raise ValueError(
                "Diffusion steps must be positive"
            )

        if not 0 < beta_start < beta_end < 1:
            raise ValueError(
                "Betas must satisfy "
                "0 < beta_start < beta_end < 1"
            )

        self.diffusion_steps = diffusion_steps
        self.device = torch.device(device)

        cpu_betas = torch.linspace(
            beta_start,
            beta_end,
            diffusion_steps,
            dtype=torch.float32,
        )

        cpu_alphas = 1.0 - cpu_betas
        cpu_alpha_bars = torch.cumprod(
            cpu_alphas,
            dim=0,
        )

        self.betas = cpu_betas.to(self.device)
        self.alphas = cpu_alphas.to(self.device)
        self.alpha_bars = cpu_alpha_bars.to(
            self.device
        )

    def sample_timesteps(self, batch_size):
        """Select one random timestep for every image in a batch."""

        if batch_size <= 0:
            raise ValueError(
                "Batch size must be positive"
            )

        return torch.randint(
            low=0,
            high=self.diffusion_steps,
            size=(batch_size,),
            dtype=torch.long,
            device=self.device,
        )

    def add_noise(
        self,
        clean_images,
        noise,
        timesteps,
    ):
        """Create x_t directly from x_0, noise, and timesteps."""

        if clean_images.ndim != 4:
            raise ValueError(
                "Clean images must have shape [B, C, H, W]"
            )

        if clean_images.shape != noise.shape:
            raise ValueError(
                "Clean images and noise must have the same shape"
            )

        batch_size = clean_images.shape[0]

        if timesteps.shape != (batch_size,):
            raise ValueError(
                "Timesteps must have shape [B]"
            )

        if timesteps.dtype != torch.long:
            raise ValueError(
                "Timesteps must use torch.long"
            )

        if clean_images.device != noise.device:
            raise ValueError(
                "Clean images and noise must use the same device"
            )

        if timesteps.device != clean_images.device:
            raise ValueError(
                "Timesteps and images must use the same device"
            )

        if timesteps.min().item() < 0:
            raise ValueError(
                "Timesteps cannot be negative"
            )

        if timesteps.max().item() >= self.diffusion_steps:
            raise ValueError(
                "Timestep exceeds the diffusion schedule"
            )

        selected_alpha_bars = self.alpha_bars[
            timesteps
        ].view(
            batch_size,
            1,
            1,
            1,
        )

        signal_coefficient = torch.sqrt(
            selected_alpha_bars
        )
        noise_coefficient = torch.sqrt(
            1.0 - selected_alpha_bars
        )

        noisy_images = (
            signal_coefficient * clean_images
            + noise_coefficient * noise
        )

        return noisy_images