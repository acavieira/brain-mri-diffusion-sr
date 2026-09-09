"""Forward and reverse diffusion noise scheduler."""

import torch


class DiffusionScheduler:
    """Store the DDPM schedule and perform diffusion steps."""

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

        cpu_previous_alpha_bars = torch.cat(
            (
                torch.ones(
                    1,
                    dtype=torch.float32,
                ),
                cpu_alpha_bars[:-1],
            )
        )

        cpu_posterior_variances = (
            cpu_betas
            * (
                1.0
                - cpu_previous_alpha_bars
            )
            / (
                1.0
                - cpu_alpha_bars
            )
        )

        self.betas = cpu_betas.to(
            self.device
        )

        self.alphas = cpu_alphas.to(
            self.device
        )

        self.alpha_bars = cpu_alpha_bars.to(
            self.device
        )

        self.posterior_variances = (
            cpu_posterior_variances.to(
                self.device
            )
        )

    def sample_timesteps(self, batch_size):
        """Select one random timestep for each image."""

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
        """Create x_t directly from x_0 and noise."""

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

    def reverse_step(
        self,
        noisy_images,
        predicted_noise,
        timesteps,
        random_noise,
    ):
        """Create x_(t-1) from x_t and predicted noise."""

        if noisy_images.ndim != 4:
            raise ValueError(
                "Noisy images must have shape [B, C, H, W]"
            )

        if predicted_noise.shape != noisy_images.shape:
            raise ValueError(
                "Predicted noise and images "
                "must have the same shape"
            )

        if random_noise.shape != noisy_images.shape:
            raise ValueError(
                "Random noise and images "
                "must have the same shape"
            )

        batch_size = noisy_images.shape[0]

        if timesteps.shape != (batch_size,):
            raise ValueError(
                "Timesteps must have shape [B]"
            )

        if timesteps.dtype != torch.long:
            raise ValueError(
                "Timesteps must use torch.long"
            )

        if noisy_images.device != predicted_noise.device:
            raise ValueError(
                "Predicted noise and images "
                "must use the same device"
            )

        if noisy_images.device != random_noise.device:
            raise ValueError(
                "Random noise and images "
                "must use the same device"
            )

        if timesteps.device != noisy_images.device:
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

        selected_betas = self.betas[
            timesteps
        ].view(
            batch_size,
            1,
            1,
            1,
        )

        selected_alphas = self.alphas[
            timesteps
        ].view(
            batch_size,
            1,
            1,
            1,
        )

        selected_alpha_bars = self.alpha_bars[
            timesteps
        ].view(
            batch_size,
            1,
            1,
            1,
        )

        selected_variances = (
            self.posterior_variances[
                timesteps
            ].view(
                batch_size,
                1,
                1,
                1,
            )
        )

        model_mean = (
            1.0
            / torch.sqrt(selected_alphas)
        ) * (
            noisy_images
            - (
                selected_betas
                / torch.sqrt(
                    1.0
                    - selected_alpha_bars
                )
            )
            * predicted_noise
        )

        added_noise = (
            torch.sqrt(selected_variances)
            * random_noise
        )

        nonzero_mask = (
            timesteps > 0
        ).float().view(
            batch_size,
            1,
            1,
            1,
        )

        previous_images = (
            model_mean
            + nonzero_mask * added_noise
        )

        return previous_images