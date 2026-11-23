# SPDX-FileCopyrightText: 2026 perennialtech contributors
# SPDX-License-Identifier: MIT

"""Perceptual losses, differentiable structural metrics, and foldable ECB blocks in PyTorch.

All training losses and re-parameterization branches are strictly training-only
and incur 0% inference overhead after folding to standard 3x3 convolutions.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


_SOBEL_X = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]) / 4.0
_SOBEL_Y = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]) / 4.0


def _gaussian_window(window_size: int = 11, sigma: float = 1.5, device=None, dtype=torch.float32) -> torch.Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype) - (window_size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2.0 * sigma * sigma))
    g = g / g.sum()
    return (g[:, None] * g[None, :])[None, None]


def ssim_map(
    pred: torch.Tensor,
    target: torch.Tensor,
    window: Optional[torch.Tensor] = None,
    c1: float = 0.01 ** 2,
    c2: float = 0.03 ** 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-pixel (ssim, cs) maps on [0, 1] float32 tensors of shape (N, C, H, W)."""
    if window is None:
        window = _gaussian_window(11, 1.5, device=pred.device, dtype=pred.dtype)
    channels = pred.shape[1]
    if window.shape[0] != channels:
        window = window.expand(channels, 1, -1, -1).contiguous()
    pad = window.shape[-1] // 2
    mu1 = F.conv2d(pred, window, padding=pad, groups=channels)
    mu2 = F.conv2d(target, window, padding=pad, groups=channels)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = (F.conv2d(pred * pred, window, padding=pad, groups=channels) - mu1_sq).clamp_min(0.0)
    sigma2_sq = (F.conv2d(target * target, window, padding=pad, groups=channels) - mu2_sq).clamp_min(0.0)
    sigma12 = F.conv2d(pred * target, window, padding=pad, groups=channels) - mu1_mu2
    cs = (2.0 * sigma12 + c2) / (sigma1_sq + sigma2_sq + c2)
    ssim = ((2.0 * mu1_mu2 + c1) * (2.0 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    return ssim, cs


def ssim_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """1 - mean SSIM on [0, 1] tensors."""
    val, _ = ssim_map(pred.float(), target.float())
    return 1.0 - val.mean()


def ms_ssim_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """3-scale MS-SSIM loss suitable for 96x96+ training patches."""
    x, y = pred.float(), target.float()
    window = _gaussian_window(11, 1.5, device=x.device, dtype=x.dtype)
    weights = (0.25, 0.35, 0.40)
    losses = []
    for i, w in enumerate(weights):
        s, cs = ssim_map(x, y, window=window)
        losses.append(w * (1.0 - (s.mean() if i == len(weights) - 1 else cs.mean())))
        if i < len(weights) - 1:
            x = F.avg_pool2d(x, 2)
            y = F.avg_pool2d(y, 2)
    return sum(losses)


def sobel_edge_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """Charbonnier Sobel gradient-agreement loss."""
    x, y = pred.float(), target.float()
    kx = _SOBEL_X.to(device=x.device, dtype=x.dtype).view(1, 1, 3, 3)
    ky = _SOBEL_Y.to(device=x.device, dtype=x.dtype).view(1, 1, 3, 3)
    px = F.conv2d(x, kx, padding=1)
    tx = F.conv2d(y, kx, padding=1)
    py = F.conv2d(x, ky, padding=1)
    ty = F.conv2d(y, ky, padding=1)
    lx = torch.sqrt((px - tx).square() + eps * eps).mean()
    ly = torch.sqrt((py - ty).square() + eps * eps).mean()
    return 0.5 * (lx + ly)


def fft_magnitude_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """2D rFFT amplitude spectrum agreement loss (preserves high-frequency line energy)."""
    x, y = pred.float(), target.float()
    fp = torch.fft.rfft2(x, norm="ortho")
    ft = torch.fft.rfft2(y, norm="ortho")
    mp = torch.sqrt(fp.real.square() + fp.imag.square() + eps * eps)
    mt = torch.sqrt(ft.real.square() + ft.imag.square() + eps * eps)
    return torch.sqrt((mp - mt).square() + eps * eps).mean()


def halo_ringing_mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Mean absolute error in the 2-4px halo band immediately surrounding strong Sobel edges."""
    x, y = pred.float(), target.float()
    kx = _SOBEL_X.to(device=x.device, dtype=x.dtype).view(1, 1, 3, 3) / 2.0
    ky = _SOBEL_Y.to(device=x.device, dtype=x.dtype).view(1, 1, 3, 3) / 2.0
    grad = torch.hypot(F.conv2d(y, kx, padding=1), F.conv2d(y, ky, padding=1))
    core_edge = (grad > 0.04).float()
    dilated = F.max_pool2d(core_edge, kernel_size=7, stride=1, padding=3)
    halo_band = (dilated > 0.5) & (core_edge < 0.5)
    if not halo_band.any():
        return 0.0
    return float((x - y).abs()[halo_band].mean().item())


class AlexNetLPIPS(nn.Module):
    """Exact LPIPS v0.1 (AlexNet) in pure PyTorch with zero torchvision dependency.

    Uses standard PyTorch AlexNet feature weights (`alexnet-owt-7be5be79.pth`) and
    official LPIPS v0.1 linear calibration weights (`alex.pth`).
    """

    def __init__(self, alex_weights: str | Path, lpips_weights: str | Path):
        super().__init__()
        self.slice1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
        )
        self.slice2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
        )
        self.slice3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.slice4 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.slice5 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.lins = nn.ModuleList([
            nn.Conv2d(64, 1, 1, bias=False),
            nn.Conv2d(192, 1, 1, bias=False),
            nn.Conv2d(384, 1, 1, bias=False),
            nn.Conv2d(256, 1, 1, bias=False),
            nn.Conv2d(256, 1, 1, bias=False),
        ])
        self.register_buffer("shift", torch.tensor([-0.030, -0.088, -0.188]).view(1, 3, 1, 1))
        self.register_buffer("scale", torch.tensor([0.458, 0.448, 0.450]).view(1, 3, 1, 1))

        alex_sd = torch.load(alex_weights, map_location="cpu", weights_only=True)
        mapping = [
            (self.slice1[0], "features.0"),
            (self.slice2[1], "features.3"),
            (self.slice3[1], "features.6"),
            (self.slice4[0], "features.8"),
            (self.slice5[0], "features.10"),
        ]
        with torch.no_grad():
            for conv, prefix in mapping:
                conv.weight.copy_(alex_sd[f"{prefix}.weight"])
                conv.bias.copy_(alex_sd[f"{prefix}.bias"])
            lpips_sd = torch.load(lpips_weights, map_location="cpu", weights_only=True)
            for i, lin in enumerate(self.lins):
                lin.weight.copy_(lpips_sd[f"lin{i}.model.1.weight"])
        self.eval().requires_grad_(False)

    def _features(self, x: torch.Tensor) -> list[torch.Tensor]:
        if x.shape[1] == 1:
            x = x.expand(-1, 3, -1, -1)
        h = (2.0 * x - 1.0 - self.shift) / self.scale
        f1 = self.slice1(h)
        f2 = self.slice2(f1)
        f3 = self.slice3(f2)
        f4 = self.slice4(f3)
        f5 = self.slice5(f4)
        return [f1, f2, f3, f4, f5]

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Return scalar mean LPIPS distance over batch on [0, 1] tensors."""
        feats_p = self._features(pred.float())
        with torch.no_grad():
            feats_t = self._features(target.float())
        total = 0.0
        for fp, ft, lin in zip(feats_p, feats_t, self.lins, strict=True):
            np_ = fp / (torch.sqrt((fp * fp).sum(dim=1, keepdim=True) + 1e-10))
            nt_ = ft / (torch.sqrt((ft * ft).sum(dim=1, keepdim=True) + 1e-10))
            diff_sq = (np_ - nt_).square()
            total = total + lin(diff_sq).mean()
        return total


class _L2Pooling(nn.Module):
    """Anti-aliased 5x5 Hanning-window L2 pooling used by DISTS (Ding et al., TPAMI 2020)."""

    def __init__(self, channels: int, filter_size: int = 5, stride: int = 2):
        super().__init__()
        self.channels = channels
        self.stride = stride
        self.padding = (filter_size - 2) // 2
        a = np.hanning(filter_size + 2)[1:-1]
        g = torch.tensor(a[:, None] * a[None, :], dtype=torch.float32)
        g = g / g.sum()
        self.register_buffer("filter", g[None, None, :, :].repeat(channels, 1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_sq = x * x
        out = F.conv2d(x_sq, self.filter, stride=self.stride, padding=self.padding, groups=self.channels)
        return torch.sqrt(out + 1e-12)


class DISTS(nn.Module):
    """Exact Deep Image Structure and Texture Similarity (DISTS) in pure PyTorch.

    Combines VGG16 features with anti-aliased L2 pooling and calibrated structure/texture
    weights (`alpha`, `beta`) across 6 scales (3 + 64 + 128 + 256 + 512 + 512 = 1475 channels).
    """

    def __init__(self, vgg_weights: str | Path, dists_weights: str | Path):
        super().__init__()
        self.stage1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.stage2 = nn.Sequential(
            _L2Pooling(64),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.stage3 = nn.Sequential(
            _L2Pooling(128),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.stage4 = nn.Sequential(
            _L2Pooling(256),
            nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.stage5 = nn.Sequential(
            _L2Pooling(512),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.chns = (3, 64, 128, 256, 512, 512)

        vgg_sd = torch.load(vgg_weights, map_location="cpu", weights_only=True)
        mapping = [
            (self.stage1[0], "features.0"), (self.stage1[2], "features.2"),
            (self.stage2[1], "features.5"), (self.stage2[3], "features.7"),
            (self.stage3[1], "features.10"), (self.stage3[3], "features.12"), (self.stage3[5], "features.14"),
            (self.stage4[1], "features.17"), (self.stage4[3], "features.19"), (self.stage4[5], "features.21"),
            (self.stage5[1], "features.24"), (self.stage5[3], "features.26"), (self.stage5[5], "features.28"),
        ]
        with torch.no_grad():
            for conv, prefix in mapping:
                conv.weight.copy_(vgg_sd[f"{prefix}.weight"])
                conv.bias.copy_(vgg_sd[f"{prefix}.bias"])
        dists_sd = torch.load(dists_weights, map_location="cpu", weights_only=True)
        alpha = dists_sd["alpha"].float()
        beta = dists_sd["beta"].float()
        denom = (alpha + beta).sum().clamp_min(1e-8)
        self.register_buffer("alpha", alpha / denom)
        self.register_buffer("beta", beta / denom)
        self.eval().requires_grad_(False)

    def _features(self, x: torch.Tensor) -> list[torch.Tensor]:
        if x.shape[1] == 1:
            x = x.expand(-1, 3, -1, -1)
        h = (x - self.mean) / self.std
        h1 = self.stage1(h)
        h2 = self.stage2(h1)
        h3 = self.stage3(h2)
        h4 = self.stage4(h3)
        h5 = self.stage5(h4)
        return [x, h1, h2, h3, h4, h5]

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Return scalar mean DISTS distance over batch on [0, 1] tensors (0 = identical)."""
        with torch.autocast("cuda", enabled=False):
            feats_p = self._features(pred.float())
            with torch.no_grad():
                feats_t = self._features(target.float())
        alphas = torch.split(self.alpha, self.chns, dim=1)
        betas = torch.split(self.beta, self.chns, dim=1)
        c1, c2 = 1e-6, 1e-6
        dist1 = 0.0
        dist2 = 0.0
        for fp, ft, a, b in zip(feats_p, feats_t, alphas, betas, strict=True):
            mp = fp.mean([2, 3], keepdim=True)
            mt = ft.mean([2, 3], keepdim=True)
            s1 = (2.0 * mp * mt + c1) / (mp * mp + mt * mt + c1)
            dist1 = dist1 + (a * s1).sum(dim=1, keepdim=True)

            vp = ((fp - mp) ** 2).mean([2, 3], keepdim=True)
            vt = ((ft - mt) ** 2).mean([2, 3], keepdim=True)
            cov = (fp * ft).mean([2, 3], keepdim=True) - mp * mt
            s2 = (2.0 * cov + c2) / (vp + vt + c2)
            dist2 = dist2 + (b * s2).sum(dim=1, keepdim=True)
        return (1.0 - (dist1 + dist2)).mean()


def find_lpips_weights() -> Optional[tuple[Path, Path]]:
    """Locate cached AlexNet + LPIPS weights if present on the machine."""
    alex_candidates = [
        Path(os.environ.get("ARTCNN_ALEXNET_WEIGHTS", "")),
        Path.home() / ".local/share/artcnn-pareto/weights/alexnet-owt-7be5be79.pth",
        Path.home() / ".cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth",
    ]
    lpips_candidates = [
        Path(os.environ.get("ARTCNN_LPIPS_WEIGHTS", "")),
        Path.home() / ".local/share/artcnn-pareto/weights/lpips_alex.pth",
        Path.home() / ".local/share/artcnn-pareto/perceptual-exp/lpips_alex.pth",
        Path.home() / ".cache/artcnn/lpips_alex.pth",
    ]
    alex = next((p for p in alex_candidates if p and p.is_file()), None)
    lpips = next((p for p in lpips_candidates if p and p.is_file()), None)
    if alex and lpips:
        return alex, lpips
    return None


def find_dists_weights() -> Optional[tuple[Path, Path]]:
    """Locate cached VGG16 features + DISTS weights if present on the machine."""
    vgg_candidates = [
        Path(os.environ.get("ARTCNN_VGG16_WEIGHTS", "")),
        Path.home() / ".local/share/artcnn-pareto/weights/vgg16_features.pth",
        Path.home() / ".cache/torch/hub/checkpoints/vgg16_features.pth",
        Path.home() / ".cache/torch/hub/checkpoints/vgg16-397923af.pth",
    ]
    dists_candidates = [
        Path(os.environ.get("ARTCNN_DISTS_WEIGHTS", "")),
        Path.home() / ".local/share/artcnn-pareto/weights/dists_weights.pt",
        Path.home() / ".cache/torch/hub/checkpoints/dists_weights.pt",
        Path.home() / ".cache/artcnn/dists_weights.pt",
    ]
    vgg = next((p for p in vgg_candidates if p and p.is_file()), None)
    dists = next((p for p in dists_candidates if p and p.is_file()), None)
    if vgg and dists:
        return vgg, dists
    return None


class FoldableECB(nn.Module):
    """5-branch linear EdgeConvBlock that folds losslessly into a single 3x3 Conv2d.

    Supports exact lossless initialization from an existing 3x3 (weight, bias) so
    fine-tuning starts with zero shock while benefiting from multi-branch edge
    and identity gradient paths.
    """

    def __init__(self, c_in: int, c_out: int, dilation: int = 1):
        super().__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.dilation = dilation
        self.conv3 = nn.Conv2d(c_in, c_out, 3, padding=dilation, dilation=dilation)
        self.conv1 = nn.Conv2d(c_in, c_out, 1, padding=0)
        self.sobel_x_mix = nn.Conv2d(c_in, c_out, 1, padding=0, bias=False)
        self.sobel_y_mix = nn.Conv2d(c_in, c_out, 1, padding=0, bias=False)
        self.register_buffer("sx", _SOBEL_X.view(1, 1, 3, 3).repeat(c_out, 1, 1, 1) * 4.0)
        self.register_buffer("sy", _SOBEL_Y.view(1, 1, 3, 3).repeat(c_out, 1, 1, 1) * 4.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv3(x) + self.conv1(x)
        if self.c_in == self.c_out:
            y = y + x
        y = y + F.conv2d(self.sobel_x_mix(x), self.sx, padding=self.dilation, dilation=self.dilation, groups=self.c_out)
        y = y + F.conv2d(self.sobel_y_mix(x), self.sy, padding=self.dilation, dilation=self.dilation, groups=self.c_out)
        return y

    @torch.no_grad()
    def initialize_from_conv(self, weight: torch.Tensor, bias: torch.Tensor) -> None:
        """Initialize all 5 branches so folded weights equal (weight, bias) exactly."""
        w = weight.clone()
        if self.c_in == self.c_out:
            idx = torch.arange(self.c_out, device=w.device)
            w[idx, idx, 1, 1] -= 1.0
        self.conv3.weight.copy_(w)
        self.conv3.bias.copy_(bias)
        nn.init.zeros_(self.conv1.weight)
        nn.init.zeros_(self.conv1.bias)
        nn.init.zeros_(self.sobel_x_mix.weight)
        nn.init.zeros_(self.sobel_y_mix.weight)

    @torch.no_grad()
    def fold_weights(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Collapse all 5 linear branches into a single (W_eq, b_eq) 3x3 convolution."""
        w = self.conv3.weight.detach().double().clone()
        b = (self.conv3.bias.detach().double() + self.conv1.bias.detach().double()).clone()
        w[:, :, 1, 1] += self.conv1.weight.detach().double().squeeze(-1).squeeze(-1)
        if self.c_in == self.c_out:
            idx = torch.arange(self.c_out, device=w.device)
            w[idx, idx, 1, 1] += 1.0
        sx = self.sx[0, 0].double()
        sy = self.sy[0, 0].double()
        w += self.sobel_x_mix.weight.detach().double() * sx[None, None, :, :]
        w += self.sobel_y_mix.weight.detach().double() * sy[None, None, :, :]
        return w.float(), b.float()


class FoldableGroupedECB(nn.Module):
    """5-branch linear EdgeConvBlock that folds losslessly into a grouped 3x3 Conv2d (no bias)."""

    def __init__(self, channels: int, groups: int):
        super().__init__()
        self.channels = channels
        self.groups = groups
        self.group_ch = channels // groups
        self.conv3 = nn.Conv2d(channels, channels, 3, padding=1, groups=groups, bias=False)
        self.conv1 = nn.Conv2d(channels, channels, 1, padding=0, groups=groups, bias=False)
        self.sobel_x_mix = nn.Conv2d(channels, channels, 1, padding=0, groups=groups, bias=False)
        self.sobel_y_mix = nn.Conv2d(channels, channels, 1, padding=0, groups=groups, bias=False)
        self.register_buffer("sx", _SOBEL_X.view(1, 1, 3, 3).repeat(channels, 1, 1, 1) * 4.0)
        self.register_buffer("sy", _SOBEL_Y.view(1, 1, 3, 3).repeat(channels, 1, 1, 1) * 4.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv3(x) + self.conv1(x) + x
        y = y + F.conv2d(self.sobel_x_mix(x), self.sx, padding=1, groups=self.channels)
        y = y + F.conv2d(self.sobel_y_mix(x), self.sy, padding=1, groups=self.channels)
        return y

    @torch.no_grad()
    def initialize_from_weight(self, weight: torch.Tensor) -> None:
        w = weight.clone()
        for g in range(self.groups):
            base = g * self.group_ch
            for c in range(self.group_ch):
                w[base + c, c, 1, 1] -= 1.0
        self.conv3.weight.copy_(w)
        nn.init.zeros_(self.conv1.weight)
        nn.init.zeros_(self.sobel_x_mix.weight)
        nn.init.zeros_(self.sobel_y_mix.weight)

    @torch.no_grad()
    def fold_weight(self) -> torch.Tensor:
        w = self.conv3.weight.detach().double().clone()
        w[:, :, 1, 1] += self.conv1.weight.detach().double().squeeze(-1).squeeze(-1)
        for g in range(self.groups):
            base = g * self.group_ch
            for c in range(self.group_ch):
                w[base + c, c, 1, 1] += 1.0
        sx = self.sx[0, 0].double()
        sy = self.sy[0, 0].double()
        w += self.sobel_x_mix.weight.detach().double() * sx[None, None, :, :]
        w += self.sobel_y_mix.weight.detach().double() * sy[None, None, :, :]
        return w.float()


def fp16_ste(x: torch.Tensor) -> torch.Tensor:
    """Straight-through estimator simulating float16 rounding for GLSL/WebGPU QAT."""
    return x + (x.to(torch.float16).to(x.dtype) - x).detach()


def ldl_artifact_loss(
    pred: torch.Tensor,
    ema_pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 7,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Locally Discriminative Learning (LDL, Liang et al. CVPR 2022) artifact loss.

    Estimates local patch variance of the residual (pred - target) versus the stable
    EMA reference (ema_pred - target) to penalize GAN/perceptual hallucination artifacts
    in flat regions and halos without damping genuine high-frequency lineart.
    """
    p = pred.float()
    e = ema_pred.detach().float()
    t = target.float()
    r_p = p - t
    r_e = e - t
    pad = window_size // 2
    mean_p = F.avg_pool2d(r_p, window_size, stride=1, padding=pad)
    var_p = (F.avg_pool2d(r_p * r_p, window_size, stride=1, padding=pad) - mean_p * mean_p).clamp_min(0.0)
    mean_e = F.avg_pool2d(r_e, window_size, stride=1, padding=pad)
    var_e = (F.avg_pool2d(r_e * r_e, window_size, stride=1, padding=pad) - mean_e * mean_e).clamp_min(0.0)
    # Artifact weight map: pixels where pred exhibits higher local residual variance than EMA
    # plus a smooth pixel-wise discrepancy gate
    artifact_map = F.relu(var_p - var_e).sqrt()
    artifact_map = artifact_map * (r_p.abs() / (r_e.abs() + 1e-3)).clamp(0.0, 5.0)
    return (artifact_map.detach() * torch.sqrt(r_p * r_p + eps)).mean()


class ClipAnimeIQA(nn.Module):
    """Differentiable CLIP-IQA + semantic fidelity loss for anime perceptual SR.

    Uses a frozen OpenCLIP ViT-B-32 encoder to score outputs against antonym
    anime-quality text prompts (NR-IQA) combined with cosine feature similarity
    to the ground-truth patch (preventing semantic drift/hallucination).
    Strictly training-only (0% inference cost).
    """

    PROMPT_PAIRS = (
        ("a crisp, sharp, high-definition anime frame with clean lineart",
         "a blurry, noisy, compressed anime frame with ringing artifacts"),
        ("clean detailed 2D animation artwork",
         "low resolution pixelated smeared animation"),
        ("sharp high-contrast cel lineart",
         "soft blurry aliased edges with compression blocks"),
    )

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        semantic_weight: float = 0.5,
    ):
        super().__init__()
        import open_clip

        model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        tokenizer = open_clip.get_tokenizer(model_name)
        self.visual = model.visual
        self.visual.eval().requires_grad_(False)
        self.semantic_weight = float(semantic_weight)

        pos_texts = [p[0] for p in self.PROMPT_PAIRS]
        neg_texts = [p[1] for p in self.PROMPT_PAIRS]
        with torch.no_grad():
            pos_emb = F.normalize(model.encode_text(tokenizer(pos_texts)), dim=-1)
            neg_emb = F.normalize(model.encode_text(tokenizer(neg_texts)), dim=-1)
        self.register_buffer("pos_emb", pos_emb.float())
        self.register_buffer("neg_emb", neg_emb.float())
        self.register_buffer("mean", torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1))
        self.eval().requires_grad_(False)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 1:
            x = x.expand(-1, 3, -1, -1)
        if x.shape[-2:] != (224, 224):
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        x = (x - self.mean.to(x.dtype)) / self.std.to(x.dtype)
        feat = self.visual(x)
        return F.normalize(feat.float(), dim=-1)

    @torch.no_grad()
    def quality_score(self, x: torch.Tensor) -> torch.Tensor:
        """Return NR-IQA quality score in [0, 1] (higher = sharper/cleaner anime)."""
        feat = self._encode(x.clamp(0.0, 1.0))
        sim_pos = (feat @ self.pos_emb.T) * 20.0
        sim_neg = (feat @ self.neg_emb.T) * 20.0
        probs = torch.sigmoid(sim_pos - sim_neg)
        return probs.mean(dim=-1)

    def forward(self, pred: torch.Tensor, target: torch.Tensor, max_samples: int = 8) -> torch.Tensor:
        """Return scalar differentiable loss combining NR-IQA (1 - quality) and semantic fidelity."""
        if pred.shape[0] > max_samples:
            pred = pred[:max_samples]
            target = target[:max_samples]
        f_pred = self._encode(pred.clamp(0.0, 1.0))
        with torch.no_grad():
            f_tgt = self._encode(target.clamp(0.0, 1.0))
        sim_pos = (f_pred @ self.pos_emb.T) * 20.0
        sim_neg = (f_pred @ self.neg_emb.T) * 20.0
        nr_loss = (1.0 - torch.sigmoid(sim_pos - sim_neg)).mean()
        sem_loss = (1.0 - (f_pred * f_tgt).sum(dim=-1)).mean()
        return nr_loss + self.semantic_weight * sem_loss
