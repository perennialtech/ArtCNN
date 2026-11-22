# SPDX-FileCopyrightText: 2026 perennialtech contributors
# SPDX-License-Identifier: MIT

"""Training-only discriminators, GAN losses, and sub-pixel shift consistency for ArtCNN.

All modules here are strictly training-time regularizers (0% inference cost).
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm


def _sn_conv(c_in: int, c_out: int, k: int = 3, stride: int = 1, pad: int = 1) -> nn.Module:
    return spectral_norm(nn.Conv2d(c_in, c_out, k, stride=stride, padding=pad))


class SpectralNormPatchGAN(nn.Module):
    """Lightweight 4-stage Spectral-Norm PatchGAN discriminator returning (logits, features)."""

    def __init__(self, in_ch: int = 1, base_ch: int = 48):
        super().__init__()
        self.b0 = nn.Sequential(_sn_conv(in_ch, base_ch, 3, 1, 1), nn.LeakyReLU(0.2, inplace=True))
        self.b1 = nn.Sequential(_sn_conv(base_ch, base_ch * 2, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True))
        self.b2 = nn.Sequential(_sn_conv(base_ch * 2, base_ch * 4, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True))
        self.b3 = nn.Sequential(_sn_conv(base_ch * 4, base_ch * 4, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True))
        self.head = _sn_conv(base_ch * 4, 1, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        f0 = self.b0(x)
        f1 = self.b1(f0)
        f2 = self.b2(f1)
        f3 = self.b3(f2)
        return self.head(f3), [f0, f1, f2, f3]


class SpectralNormUNetDiscriminator(nn.Module):
    """Real-ESRGAN-style Spectral-Norm U-Net discriminator providing pixel-wise feedback.

    Pixel-wise realism feedback lets the discriminator sharpen anime lineart while
    simultaneously penalizing hallucinated grain/noise in flat cel regions.
    """

    def __init__(self, in_ch: int = 1, base_ch: int = 48):
        super().__init__()
        self.conv0 = _sn_conv(in_ch, base_ch, 3, 1, 1)
        self.down1 = _sn_conv(base_ch, base_ch * 2, 4, 2, 1)
        self.down2 = _sn_conv(base_ch * 2, base_ch * 4, 4, 2, 1)
        self.down3 = _sn_conv(base_ch * 4, base_ch * 8, 4, 2, 1)

        self.up3 = _sn_conv(base_ch * 8, base_ch * 4, 3, 1, 1)
        self.up2 = _sn_conv(base_ch * 4, base_ch * 2, 3, 1, 1)
        self.up1 = _sn_conv(base_ch * 2, base_ch, 3, 1, 1)

        self.conv_out1 = _sn_conv(base_ch, base_ch, 3, 1, 1)
        self.conv_out2 = _sn_conv(base_ch, base_ch, 3, 1, 1)
        self.head = nn.Conv2d(base_ch, 1, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        x0 = F.leaky_relu(self.conv0(x), 0.2, inplace=True)
        x1 = F.leaky_relu(self.down1(x0), 0.2, inplace=True)
        x2 = F.leaky_relu(self.down2(x1), 0.2, inplace=True)
        x3 = F.leaky_relu(self.down3(x2), 0.2, inplace=True)

        u3 = F.interpolate(x3, size=x2.shape[-2:], mode="bilinear", align_corners=False)
        u3 = F.leaky_relu(self.up3(u3), 0.2, inplace=True) + x2

        u2 = F.interpolate(u3, size=x1.shape[-2:], mode="bilinear", align_corners=False)
        u2 = F.leaky_relu(self.up2(u2), 0.2, inplace=True) + x1

        u1 = F.interpolate(u2, size=x0.shape[-2:], mode="bilinear", align_corners=False)
        u1 = F.leaky_relu(self.up1(u1), 0.2, inplace=True) + x0

        h = F.leaky_relu(self.conv_out1(u1), 0.2, inplace=True)
        h = F.leaky_relu(self.conv_out2(h), 0.2, inplace=True)
        return self.head(h), [x1, x2, x3, u1]


def gan_d_loss(real_logits: torch.Tensor, fake_logits: torch.Tensor, kind: str = "hinge") -> torch.Tensor:
    if kind == "hinge":
        return 0.5 * (F.relu(1.0 - real_logits).mean() + F.relu(1.0 + fake_logits).mean())
    if kind == "ragan":
        r_m = real_logits - fake_logits.mean()
        f_m = fake_logits - real_logits.mean()
        return 0.5 * (
            F.binary_cross_entropy_with_logits(r_m, torch.ones_like(r_m))
            + F.binary_cross_entropy_with_logits(f_m, torch.zeros_like(f_m))
        )
    raise ValueError(f"Unknown GAN loss kind: {kind}")


def gan_g_loss(real_logits: torch.Tensor, fake_logits: torch.Tensor, kind: str = "hinge") -> torch.Tensor:
    if kind == "hinge":
        return -fake_logits.mean()
    if kind == "ragan":
        r_m = real_logits - fake_logits.mean()
        f_m = fake_logits - real_logits.mean()
        return 0.5 * (
            F.binary_cross_entropy_with_logits(r_m, torch.zeros_like(r_m))
            + F.binary_cross_entropy_with_logits(f_m, torch.ones_like(f_m))
        )
    raise ValueError(f"Unknown GAN loss kind: {kind}")


def feature_matching_loss(real_feats: list[torch.Tensor], fake_feats: list[torch.Tensor]) -> torch.Tensor:
    return sum(F.l1_loss(ff, rf.detach()) for rf, ff in zip(real_feats, fake_feats, strict=True)) / len(real_feats)


def subpixel_shimmer_loss(
    model: nn.Module,
    hr: torch.Tensor,
    pred: torch.Tensor,
    downscale_fn,
) -> torch.Tensor:
    """Penalize disagreement between upscaling LR(hr) and LR(1-px shifted hr) (= 0.5-px LR shift)."""
    hr_shifted = torch.roll(hr, shifts=(1, 1), dims=(-2, -1))
    lr_shifted = downscale_fn(hr_shifted).to(memory_format=torch.channels_last)
    pred_shifted = model(lr_shifted)
    # Overlap: pred[:, :, 12:-13, 12:-13] aligns with pred_shifted[:, :, 13:-12, 13:-12]
    va = pred.float()[:, :, 12:-13, 12:-13]
    vb = pred_shifted.float()[:, :, 13:-12, 13:-12]
    return torch.sqrt((va - vb).square() + 1e-6).mean()
