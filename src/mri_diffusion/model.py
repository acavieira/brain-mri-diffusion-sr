"""Conditional U-Net that sees a noisy HR slice and its LR condition and predicts v.

The layout is the DDPM U-Net (Ho et al. 2020): two residual blocks per
resolution, group normalization, a sinusoidal timestep embedding, dropout
inside the residual blocks and self-attention at a low resolution. Two
changes from Dhariwal & Nichol (2021) are kept because they are small and
improved results in their ablations: the timestep scales and shifts the
normalized features (AdaGN), and attention uses several heads of 64 channels.

The LR condition enters the way SR3 does it (Saharia et al. 2023): the
bicubic upsampled LR slice is stacked with the noisy slice as a second
input channel.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn


def timestep_embedding(t, dim):
    """Sinusoidal embedding of the timestep, the Transformer position code used by Ho et al."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
    angles = t.float()[:, None] * freqs[None, :]
    return torch.cat([angles.sin(), angles.cos()], dim=1)


def group_norm(channels):
    return nn.GroupNorm(32, channels)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim, dropout):
        super().__init__()
        self.norm1 = group_norm(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        # one linear layer gives a scale and a shift for every channel (AdaGN)
        self.time = nn.Linear(time_dim, 2 * out_ch)
        self.norm2 = group_norm(out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.time(F.silu(t_emb))[:, :, None, None].chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale) + shift
        # dropout right before the last convolution, where Ho et al. put it
        h = self.conv2(self.dropout(F.silu(h)))
        return h + self.skip(x)


class Attention(nn.Module):
    """Multi-head self-attention over all positions of a feature map."""

    def __init__(self, channels, head_channels):
        super().__init__()
        self.heads = max(1, channels // head_channels)
        self.norm = group_norm(channels)
        self.qkv = nn.Conv2d(channels, 3 * channels, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv(self.norm(x)).reshape(b, 3, self.heads, c // self.heads, h * w)
        # scaled_dot_product_attention expects (batch, heads, positions, channels)
        q, k, v = qkv.transpose(-1, -2).unbind(dim=1)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(-1, -2).reshape(b, c, h, w)
        return x + self.proj(out)


class Block(nn.Module):
    """A residual block, followed by attention when the resolution asks for it."""

    def __init__(self, in_ch, out_ch, time_dim, dropout, attention, head_channels):
        super().__init__()
        self.res = ResBlock(in_ch, out_ch, time_dim, dropout)
        self.attn = Attention(out_ch, head_channels) if attention else nn.Identity()

    def forward(self, x, t_emb):
        return self.attn(self.res(x, t_emb))


class UNet(nn.Module):
    def __init__(self, base=64, mults=(1, 2, 4, 4), res_blocks=2, attention_res=(32,),
                 head_channels=64, dropout=0.1, image_size=256):
        super().__init__()
        self.base = base
        time_dim = 4 * base
        self.time_mlp = nn.Sequential(
            nn.Linear(base, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim)
        )

        # channel 0 is the noisy slice x_t, channel 1 the bicubic condition
        self.input = nn.Conv2d(2, base, 3, padding=1)

        # encoder: every block leaves a skip connection for the decoder
        self.encoder = nn.ModuleList()
        self.downs = nn.ModuleList()
        skips = []
        ch, res = base, image_size
        for level, mult in enumerate(mults):
            blocks = nn.ModuleList()
            for _ in range(res_blocks):
                blocks.append(Block(ch, base * mult, time_dim, dropout,
                                    res in attention_res, head_channels))
                ch = base * mult
                skips.append(ch)
            self.encoder.append(blocks)
            last = level == len(mults) - 1
            self.downs.append(nn.Identity() if last else nn.Conv2d(ch, ch, 3, stride=2, padding=1))
            res = res if last else res // 2

        # middle block of the DDPM U-Net: resblock, attention, resblock
        self.mid1 = ResBlock(ch, ch, time_dim, dropout)
        self.mid_attn = Attention(ch, head_channels)
        self.mid2 = ResBlock(ch, ch, time_dim, dropout)

        # decoder: mirror of the encoder, each block takes one skip connection
        self.decoder = nn.ModuleList()
        self.ups = nn.ModuleList()
        for level, mult in reversed(list(enumerate(mults))):
            blocks = nn.ModuleList()
            for _ in range(res_blocks):
                blocks.append(Block(ch + skips.pop(), base * mult, time_dim, dropout,
                                    res in attention_res, head_channels))
                ch = base * mult
            self.decoder.append(blocks)
            first = level == 0
            self.ups.append(nn.Identity() if first else nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(ch, ch, 3, padding=1)))
            res = res if first else res * 2

        self.output = nn.Sequential(group_norm(ch), nn.SiLU(), nn.Conv2d(ch, 1, 3, padding=1))

    def forward(self, x_t, condition, t):
        t_emb = self.time_mlp(timestep_embedding(t, self.base))
        h = self.input(torch.cat([x_t, condition], dim=1))

        skips = []
        for blocks, down in zip(self.encoder, self.downs):
            for block in blocks:
                h = block(h, t_emb)
                skips.append(h)
            h = down(h)

        h = self.mid2(self.mid_attn(self.mid1(h, t_emb)), t_emb)

        for blocks, up in zip(self.decoder, self.ups):
            for block in blocks:
                h = block(torch.cat([h, skips.pop()], dim=1), t_emb)
            h = up(h)

        return self.output(h)
