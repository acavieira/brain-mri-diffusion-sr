"""Training operations for conditional diffusion."""

import torch
from torch.nn import functional as F
from mri_diffusion.checkpoints import save_checkpoint


def calculate_noise_loss(
    model,
    scheduler,
    hr_images,
    conditions,
    timesteps,
    true_noise,
):
    """Compare predicted and true diffusion noise."""

    noisy_images = scheduler.add_noise(
        clean_images=hr_images,
        noise=true_noise,
        timesteps=timesteps,
    )

    predicted_noise = model(
        noisy_images,
        conditions,
        timesteps,
    )

    loss = F.mse_loss(
        predicted_noise,
        true_noise,
    )

    return loss

def calculate_training_loss(
    model,
    scheduler,
    hr_images,
    conditions,
):
    """Calculate loss using random timesteps and noise."""

    batch_size = hr_images.shape[0]

    timesteps = scheduler.sample_timesteps(
        batch_size=batch_size
    )

    true_noise = torch.randn_like(
        hr_images
    )

    loss = calculate_noise_loss(
        model=model,
        scheduler=scheduler,
        hr_images=hr_images,
        conditions=conditions,
        timesteps=timesteps,
        true_noise=true_noise,
    )

    return loss

def train_one_batch(
    model,
    scheduler,
    optimizer,
    hr_images,
    conditions,
    max_gradient_norm,
):
    """Update the model using one batch."""

    if max_gradient_norm <= 0:
        raise ValueError(
            "Maximum gradient norm must be positive"
        )

    model.train()

    optimizer.zero_grad(
        set_to_none=True
    )

    loss = calculate_training_loss(
        model=model,
        scheduler=scheduler,
        hr_images=hr_images,
        conditions=conditions,
    )

    loss.backward()

    gradient_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=max_gradient_norm,
    )

    optimizer.step()

    return (
        loss.item(),
        gradient_norm.item(),
    )

def train_one_epoch(
    model,
    scheduler,
    optimizer,
    data_loader,
    device,
    max_gradient_norm,
):
    """Train the model once over every batch."""

    total_loss = 0.0
    total_gradient_norm = 0.0
    total_samples = 0

    for batch in data_loader:
        hr_images = batch["hr"].to(
            device
        )

        conditions = batch["condition"].to(
            device
        )

        batch_size = hr_images.shape[0]

        loss, gradient_norm = train_one_batch(
            model=model,
            scheduler=scheduler,
            optimizer=optimizer,
            hr_images=hr_images,
            conditions=conditions,
            max_gradient_norm=max_gradient_norm,
        )

        total_loss += loss * batch_size

        total_gradient_norm += (
            gradient_norm * batch_size
        )

        total_samples += batch_size

    if total_samples == 0:
        raise ValueError(
            "Training data loader is empty"
        )

    mean_loss = total_loss / total_samples

    mean_gradient_norm = (
        total_gradient_norm / total_samples
    )

    return (
        mean_loss,
        mean_gradient_norm,
    )

def validate_one_epoch(
    model,
    scheduler,
    data_loader,
    device,
    random_seed,
):
    """Calculate deterministic validation loss."""

    model.eval()

    validation_generator = torch.Generator()
    validation_generator.manual_seed(
        random_seed
    )

    total_loss = 0.0
    total_samples = 0

    with torch.inference_mode():
        for batch in data_loader:
            hr_images = batch["hr"].to(
                device
            )

            conditions = batch["condition"].to(
                device
            )

            batch_size = hr_images.shape[0]

            timesteps = torch.randint(
                low=0,
                high=scheduler.diffusion_steps,
                size=(batch_size,),
                generator=validation_generator,
                dtype=torch.long,
            ).to(device)

            true_noise = torch.randn(
                hr_images.shape,
                generator=validation_generator,
                dtype=hr_images.dtype,
            ).to(device)

            loss = calculate_noise_loss(
                model=model,
                scheduler=scheduler,
                hr_images=hr_images,
                conditions=conditions,
                timesteps=timesteps,
                true_noise=true_noise,
            )

            total_loss += (
                loss.item() * batch_size
            )

            total_samples += batch_size

    if total_samples == 0:
        raise ValueError(
            "Validation data loader is empty"
        )

    mean_loss = total_loss / total_samples

    return mean_loss

def fit_model(
    model,
    scheduler,
    optimizer,
    train_loader,
    validation_loader,
    device,
    num_epochs,
    max_gradient_norm,
    validation_seed,
    checkpoint_path,
):
    """Train, validate, and save the best model."""

    if num_epochs <= 0:
        raise ValueError(
            "Number of epochs must be positive"
        )

    history = []
    best_validation_loss = float("inf")

    for epoch in range(1, num_epochs + 1):
        train_loss, gradient_norm = (
            train_one_epoch(
                model=model,
                scheduler=scheduler,
                optimizer=optimizer,
                data_loader=train_loader,
                device=device,
                max_gradient_norm=max_gradient_norm,
            )
        )

        validation_loss = validate_one_epoch(
            model=model,
            scheduler=scheduler,
            data_loader=validation_loader,
            device=device,
            random_seed=validation_seed,
        )

        is_best_checkpoint = (
            validation_loss
            < best_validation_loss
        )

        if is_best_checkpoint:
            best_validation_loss = validation_loss

            save_checkpoint(
                path=checkpoint_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                train_loss=train_loss,
                validation_loss=validation_loss,
            )

        epoch_result = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "gradient_norm": gradient_norm,
            "best_checkpoint": is_best_checkpoint,
        }

        history.append(
            epoch_result
        )

        print(
            f"Epoch {epoch:03d}: "
            f"train_loss={train_loss:.6f}, "
            f"validation_loss="
            f"{validation_loss:.6f}, "
            f"gradient_norm="
            f"{gradient_norm:.6f}, "
            f"best={is_best_checkpoint}"
        )

    return history