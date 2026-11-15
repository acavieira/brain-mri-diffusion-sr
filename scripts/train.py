"""Train the conditional DDPM on the cached training slices.

    python scripts/train.py

Settings come from src/mri_diffusion/config.py. The run folder gets a copy
of the config, a CSV log, last.pt for resuming and best.pt with the EMA
weights that had the highest validation psnr_brain. Running the script again
continues from last.pt, which matters on Colab where sessions get cut.
"""

import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mri_diffusion import config
from mri_diffusion.data import SliceDataset, to_image
from mri_diffusion.diffusion import Diffusion
from mri_diffusion.metrics import compare
from mri_diffusion.utils import EMA, build_model, get_device, seed_everything

LOG_COLUMNS = ["step", "train_loss", "val_loss", "val_psnr_brain", "val_ssim",
               "lr", "minutes", "best"]


def forever(loader):
    while True:
        yield from loader


@torch.no_grad()
def validate(model, diffusion, val_set, picks, device, amp):
    """Loss with fixed t and noise, then PSNR and SSIM of sampled slices.

    The loss alone says little about image quality, so best.pt is chosen
    on the sampled psnr_brain.
    """
    model.eval()
    gen = torch.Generator().manual_seed(config.SEED)
    total, count = 0.0, 0
    for hr, cond in DataLoader(val_set, batch_size=config.BATCH_SIZE):
        hr, cond = hr.to(device), cond.to(device)
        t = torch.randint(0, diffusion.timesteps, (hr.shape[0],), generator=gen).to(device)
        noise = torch.randn(hr.shape, generator=gen).to(device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            loss = diffusion.loss(model, hr, cond, t=t, noise=noise)
        total += loss.item() * hr.shape[0]
        count += hr.shape[0]

    hr = torch.stack([val_set[i][0] for i in picks]).to(device)
    cond = torch.stack([val_set[i][1] for i in picks]).to(device)
    sr = diffusion.sample(model, cond, steps=config.VAL_SAMPLING_STEPS,
                          generator=torch.Generator().manual_seed(config.SEED), amp=amp)
    scores = [compare(to_image(h), to_image(s), config.BRAIN_THRESHOLD) for h, s in zip(hr, sr)]
    # nanmean because a slice without brain has no psnr_brain
    return (total / count,
            float(np.nanmean([s["psnr_brain"] for s in scores])),
            float(np.mean([s["ssim"] for s in scores])))


def main():
    seed_everything(config.SEED)
    device = get_device()
    amp = config.USE_AMP and device.type == "cuda"
    run_dir = config.RUNS_DIR / config.RUN_NAME
    run_dir.mkdir(parents=True, exist_ok=True)

    train_set = SliceDataset(config.CACHE_DIR / "train.npz", fixed_noise=False)
    val_set = SliceDataset(config.CACHE_DIR / "val.npz", fixed_noise=True)
    loader = DataLoader(train_set, batch_size=config.BATCH_SIZE, shuffle=True, drop_last=True,
                        num_workers=config.NUM_WORKERS,
                        persistent_workers=config.NUM_WORKERS > 0)
    # the same evenly spaced validation slices are sampled every time
    picks = np.linspace(0, len(val_set) - 1, config.VAL_SAMPLES).round().astype(int)
    bicubic = np.mean([compare(*val_set.pair(i)[::2], config.BRAIN_THRESHOLD)["psnr_brain"]
                       for i in picks])
    print(f"bicubic psnr_brain on the validation slices: {bicubic:.2f} dB")

    model = build_model().to(device)
    ema = EMA(model, config.EMA_DECAY)
    diffusion = Diffusion()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    # linear warm-up of the learning rate, SR3 warms up over its first 10k steps
    warmup = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min(1.0, (s + 1) / config.WARMUP_STEPS))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    step, best, elapsed = 0, -math.inf, 0.0
    last_path, best_path, log_path = run_dir / "last.pt", run_dir / "best.pt", run_dir / "log.csv"
    if last_path.exists():
        state = torch.load(last_path, map_location=device)
        model.load_state_dict(state["model"])
        ema.model.load_state_dict(state["ema"])
        optimizer.load_state_dict(state["optimizer"])
        warmup.load_state_dict(state["warmup"])
        if state["scaler"]:  # empty when the run was saved without CUDA
            scaler.load_state_dict(state["scaler"])
        step, best, elapsed = state["step"], state["best"], state.get("minutes", 0.0)
        # new random stream, otherwise the resumed run repeats the noise of the first session
        seed_everything(config.SEED + step)
        print(f"resuming from step {step}")
    else:
        settings = {k: v for k, v in vars(config).items() if k.isupper()}
        (run_dir / "config.json").write_text(json.dumps(settings, indent=2, default=str))
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(LOG_COLUMNS)

    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"{len(train_set)} train slices, {len(val_set)} val slices, "
          f"{params:.1f}M parameters, device {device}, amp {amp}")

    batches = forever(loader)
    running, seen = 0.0, 0
    start = tick = time.time()
    model.train()
    while step < config.TRAIN_STEPS:
        hr, cond = next(batches)
        hr, cond = hr.to(device, non_blocking=True), cond.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            loss = diffusion.loss(model, hr, cond)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()
        warmup.step()
        ema.update(model)
        step += 1
        running += loss.item()
        seen += 1

        if step % 100 == 0:
            now = time.time()
            print(f"step {step:6d}  loss {running / seen:.4f}  {(now - tick) / 100:.2f} s/step",
                  flush=True)
            tick = now

        if step % config.VALIDATE_EVERY == 0 or step == config.TRAIN_STEPS:
            val_loss, val_psnr, val_ssim = validate(ema.model, diffusion, val_set, picks, device, amp)
            model.train()
            # total training time, summed over all Colab sessions
            minutes = elapsed + (time.time() - start) / 60
            is_best = val_psnr > best
            if is_best:
                best = val_psnr
                torch.save({"ema": ema.model.state_dict(), "step": step, "val_psnr_brain": val_psnr},
                           best_path)
            torch.save({"model": model.state_dict(), "ema": ema.model.state_dict(),
                        "optimizer": optimizer.state_dict(), "warmup": warmup.state_dict(),
                        "scaler": scaler.state_dict(), "step": step, "best": best,
                        "minutes": minutes}, last_path)

            row = [step, running / seen, val_loss, val_psnr, val_ssim,
                   optimizer.param_groups[0]["lr"], minutes, int(is_best)]
            with open(log_path, "a", newline="") as f:
                csv.writer(f).writerow(row)
            print(f"step {step:6d}  train {running / seen:.4f}  val {val_loss:.4f}  "
                  f"psnr_brain {val_psnr:.2f}  ssim {val_ssim:.3f}  "
                  f"{minutes:.0f} min{'  best' if is_best else ''}")
            running, seen = 0.0, 0


if __name__ == "__main__":
    main()
