# SPDX-FileCopyrightText: 2026 perennialtech contributors
# SPDX-License-Identifier: MIT

"""Tests for PyTorch perceptual losses, halo metric, and lossless FoldableECB."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch import nn

from artcnn.perceptual import (
    FoldableECB,
    fft_magnitude_loss,
    halo_ringing_mae,
    ms_ssim_loss,
    sobel_edge_loss,
    ssim_loss,
)


def test_perceptual_losses_zero_on_identity() -> None:
    torch.manual_seed(0)
    x = torch.rand(2, 1, 64, 64, requires_grad=True)
    y = x.detach().clone()
    assert ssim_loss(x, y).item() < 1e-6
    assert ms_ssim_loss(x, y).item() < 1e-5
    assert sobel_edge_loss(x, y).item() < 2e-4
    assert fft_magnitude_loss(x, y).item() < 2e-5
    assert halo_ringing_mae(x, y) == 0.0


def test_perceptual_losses_backward() -> None:
    torch.manual_seed(1)
    pred = torch.rand(2, 1, 64, 64, requires_grad=True)
    target = torch.rand(2, 1, 64, 64)
    loss = ssim_loss(pred, target) + sobel_edge_loss(pred, target) + fft_magnitude_loss(pred, target)
    loss.backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all()
    assert pred.grad.abs().sum().item() > 0


def test_foldable_ecb_lossless_init_and_fold() -> None:
    torch.manual_seed(2)
    conv = nn.Conv2d(16, 16, 3, padding=1)
    ecb = FoldableECB(16, 16)
    ecb.initialize_from_conv(conv.weight, conv.bias)
    x = torch.randn(2, 16, 20, 20)
    diff_init = (conv(x) - ecb(x)).abs().max().item()
    assert diff_init < 1e-5, f"Init mismatch: {diff_init}"

    # Mutate all branches and verify folded 3x3 matches multibranch forward
    with torch.no_grad():
        ecb.conv1.weight.normal_(0, 0.05)
        ecb.conv1.bias.normal_(0, 0.02)
        ecb.sobel_x_mix.weight.normal_(0, 0.03)
        ecb.sobel_y_mix.weight.normal_(0, 0.03)
    w_eq, b_eq = ecb.fold_weights()
    folded = nn.Conv2d(16, 16, 3, padding=1)
    with torch.no_grad():
        folded.weight.copy_(w_eq)
        folded.bias.copy_(b_eq)
    diff_fold = (ecb(x) - folded(x)).abs().max().item()
    assert diff_fold < 1e-5, f"Fold mismatch: {diff_fold}"




def test_discriminators_and_gan_losses() -> None:
    from artcnn.discriminators import (
        SpectralNormPatchGAN,
        SpectralNormUNetDiscriminator,
        feature_matching_loss,
        gan_d_loss,
        gan_g_loss,
    )
    torch.manual_seed(3)
    real = torch.rand(2, 1, 64, 64)
    fake = torch.rand(2, 1, 64, 64, requires_grad=True)
    for disc in (SpectralNormPatchGAN(base_ch=16), SpectralNormUNetDiscriminator(base_ch=16)):
        r_log, r_ft = disc(real)
        f_log, f_ft = disc(fake)
        for kind in ("hinge", "ragan"):
            dl = gan_d_loss(r_log, f_log.detach(), kind)
            gl = gan_g_loss(r_log.detach(), f_log, kind)
            assert torch.isfinite(dl) and torch.isfinite(gl)
        fm = feature_matching_loss(r_ft, f_ft)
        assert torch.isfinite(fm) and fm.item() > 0


if __name__ == "__main__":
    test_perceptual_losses_zero_on_identity()
    test_perceptual_losses_backward()
    test_foldable_ecb_lossless_init_and_fold()
    test_discriminators_and_gan_losses()
    print("ALL PERCEPTUAL TESTS PASSED.")
