"""Perceptual, edge-agreement, frequency-domain, distillation, and temporal-shift
consistency losses with exact analytic gradients in NumPy.

All terms are strictly training-only (0% inference cost).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np


_SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32) / 4.0
_SOBEL_Y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32) / 4.0


def _conv2d_single_kernel(x: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Apply a 3x3 kernel to (N, 1, H, W) with zero padding."""
    xp = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)), mode="constant")
    out = np.zeros_like(x, dtype=np.float32)
    for dy in range(3):
        for dx in range(3):
            w = k[dy, dx]
            if w != 0.0:
                out += w * xp[:, :, dy : dy + x.shape[2], dx : dx + x.shape[3]]
    return out


def _conv2d_single_kernel_adjoint(dout: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Exact transpose (adjoint) of _conv2d_single_kernel."""
    return _conv2d_single_kernel(dout, k[::-1, ::-1])


def charbonnier_loss_and_grad(
    pred: np.ndarray,
    target: np.ndarray,
    eps: float = 1e-4,
) -> Tuple[float, np.ndarray]:
    """Smooth L1 (Charbonnier) pixel reconstruction loss + exact gradient."""
    diff = pred - target
    denom = np.sqrt(diff * diff + eps * eps)
    loss = float(np.mean(denom))
    grad = (diff / denom) / pred.size
    return loss, grad.astype(np.float32)


def l1_loss_and_grad(pred: np.ndarray, target: np.ndarray) -> Tuple[float, np.ndarray]:
    """Exact L1 (MAE) loss + subgradient."""
    diff = pred - target
    loss = float(np.mean(np.abs(diff)))
    grad = np.sign(diff) / pred.size
    return loss, grad.astype(np.float32)


def sobel_edge_loss_and_grad(
    pred: np.ndarray,
    target: np.ndarray,
    eps: float = 1e-4,
) -> Tuple[float, np.ndarray]:
    """Sobel edge-agreement loss: rewards matching reference edge location & strength
    without rewarding arbitrary over-sharpening."""
    px = _conv2d_single_kernel(pred, _SOBEL_X)
    tx = _conv2d_single_kernel(target, _SOBEL_X)
    py = _conv2d_single_kernel(pred, _SOBEL_Y)
    ty = _conv2d_single_kernel(target, _SOBEL_Y)

    lx, dpx = charbonnier_loss_and_grad(px, tx, eps=eps)
    ly, dpy = charbonnier_loss_and_grad(py, ty, eps=eps)
    loss = 0.5 * (lx + ly)
    grad = 0.5 * (
        _conv2d_single_kernel_adjoint(dpx, _SOBEL_X)
        + _conv2d_single_kernel_adjoint(dpy, _SOBEL_Y)
    )
    return float(loss), grad.astype(np.float32)


def fft_magnitude_loss_and_grad(
    pred: np.ndarray,
    target: np.ndarray,
    eps: float = 1e-5,
) -> Tuple[float, np.ndarray]:
    """2D FFT magnitude loss (penalizes losing high-frequency anime line energy)."""
    N, C, H, W = pred.shape
    Fp = np.fft.rfft2(pred, norm="ortho")
    Ft = np.fft.rfft2(target, norm="ortho")
    mp = np.sqrt(Fp.real ** 2 + Fp.imag ** 2 + eps * eps)
    mt = np.sqrt(Ft.real ** 2 + Ft.imag ** 2 + eps * eps)
    diff = mp - mt
    denom = np.sqrt(diff * diff + eps * eps)
    loss = float(np.mean(denom))

    # Gradient w.r.t. real/imag spectrum; handle rfft2 Hermitian symmetry weights
    d_mp = (diff / denom) / mp.size
    dFp = d_mp * (Fp / mp)
    # In rfft2, interior frequency columns appear twice in full DFT; irfft2 divides by 2
    # automatically if we scale non-DC/non-Nyquist columns appropriately.
    # Exact Wirtinger adjoint of rfft2(norm='ortho') for real x:
    # sum_{u,v} Re(conj(g_{u,v}) * F_{u,v}) -> scale interior freqs by 0.5? Wait:
    # For real x, ||x||^2 = sum_{dc,nyq} |F|^2 + 2 * sum_{interior} |F|^2.
    # Because loss sums over rfft2 array once (weight 1 on interior), adjoint is
    # irfft2 with interior bins multiplied by 0.5 so irfft2's factor-of-2 cancels.
    w_sym = np.ones(Fp.shape[-1], dtype=np.float32)
    if W % 2 == 0:
        w_sym[1:-1] = 0.5
    else:
        w_sym[1:] = 0.5
    grad = np.fft.irfft2(dFp * w_sym, s=(H, W), norm="ortho")
    return loss, grad.astype(np.float32)


def shift_image_pixels(x: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Integer pixel shift with edge replication."""
    return np.roll(x, shift=(dy, dx), axis=(-2, -1))


def temporal_shift_consistency_loss_and_grad(
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    shift_lr_y: int = 1,
    shift_lr_x: int = 1,
    scale: int = 2,
    eps: float = 1e-4,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Penalize unjustified shimmer between a frame and its 1-px shifted twin.
    pred_b was computed from input shifted by (shift_lr_y, shift_lr_x), so in HR
    space the overlapping interior is aligned by (shift_lr_y * scale, shift_lr_x * scale).
    Returns (loss, grad_a, grad_b).
    """
    sy = shift_lr_y * scale
    sx = shift_lr_x * scale
    H, W = pred_a.shape[-2], pred_a.shape[-1]
    # Overlap region (avoid wrap-around borders)
    a_slice = (slice(None), slice(None), slice(0, H - sy), slice(0, W - sx))
    b_slice = (slice(None), slice(None), slice(sy, H), slice(sx, W))
    va = pred_a[a_slice]
    vb = pred_b[b_slice]
    loss, d_va = charbonnier_loss_and_grad(va, vb, eps=eps)
    grad_a = np.zeros_like(pred_a, dtype=np.float32)
    grad_b = np.zeros_like(pred_b, dtype=np.float32)
    grad_a[a_slice] = d_va
    grad_b[b_slice] = -d_va
    return loss, grad_a, grad_b


@dataclass
class LossWeights:
    l1: float = 1.0
    sobel: float = 0.0
    fft: float = 0.0
    distill: float = 0.0
    temporal: float = 0.0


def composite_loss_and_grad(
    pred: np.ndarray,
    target: np.ndarray,
    weights: LossWeights,
    teacher_pred: Optional[np.ndarray] = None,
) -> Tuple[float, np.ndarray, Dict[str, float]]:
    """Compute weighted training loss + exact gradient w.r.t. pred."""
    total_loss = 0.0
    total_grad = np.zeros_like(pred, dtype=np.float32)
    metrics: Dict[str, float] = {}

    if weights.l1 > 0:
        l_pix, g_pix = charbonnier_loss_and_grad(pred, target)
        total_loss += weights.l1 * l_pix
        total_grad += weights.l1 * g_pix
        metrics["l1"] = l_pix

    if weights.sobel > 0:
        l_edge, g_edge = sobel_edge_loss_and_grad(pred, target)
        total_loss += weights.sobel * l_edge
        total_grad += weights.sobel * g_edge
        metrics["sobel"] = l_edge

    if weights.fft > 0:
        l_fft, g_fft = fft_magnitude_loss_and_grad(pred, target)
        total_loss += weights.fft * l_fft
        total_grad += weights.fft * g_fft
        metrics["fft"] = l_fft

    if weights.distill > 0 and teacher_pred is not None:
        l_kd, g_kd = charbonnier_loss_and_grad(pred, teacher_pred)
        total_loss += weights.distill * l_kd
        total_grad += weights.distill * g_kd
        metrics["distill"] = l_kd

    metrics["total"] = total_loss
    return total_loss, total_grad, metrics
