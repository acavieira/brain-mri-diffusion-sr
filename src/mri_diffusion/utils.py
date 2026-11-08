"""Small helpers shared by the scripts."""

import copy
import random

import numpy as np
import torch

from mri_diffusion import config
from mri_diffusion.model import UNet


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_model():
    return UNet(
        base=config.BASE_CHANNELS,
        mults=config.CHANNEL_MULTS,
        res_blocks=config.RES_BLOCKS,
        attention_res=config.ATTENTION_RESOLUTIONS,
        head_channels=config.HEAD_CHANNELS,
        dropout=config.DROPOUT,
        image_size=config.IMAGE_SIZE,
    )


class EMA:
    """Exponential moving average of the weights, used for every evaluation.

    Ho et al. (2020) and Dhariwal & Nichol (2021) use a decay of 0.9999 for
    runs of hundreds of thousands of steps. Our runs are much shorter, so the
    default is 0.999, which averages roughly the last 1000 updates.
    """

    def __init__(self, model, decay):
        self.decay = decay
        self.model = copy.deepcopy(model).eval().requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for ema_param, param in zip(self.model.parameters(), model.parameters()):
            ema_param.lerp_(param, 1.0 - self.decay)
