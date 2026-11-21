"""Algebraic collapse of consecutive linear 3x3 convolutions into a 5x5 convolution,
plus exact interior-vs-border numerical verification.

In ArtCNN C-series:
  - `mix_global` (3x3, linear) -> `features` (3x3, linear) has no activation between them.
  - Likewise `conv0` (3x3, linear) -> `features` along the skip path is linear.

This module computes the exact equivalent 5x5 kernel and verifies:
  1. Away from image borders (margin >= 2 px), the collapsed 5x5 agrees to float64 precision.
  2. At image borders, naive zero-padded 5x5 diverges from sequential zero-padded 3x3s.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def collapse_two_3x3_to_5x5(
    W1: np.ndarray,
    b1: np.ndarray,
    W2: np.ndarray,
    b2: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Collapse y = Conv3x3(Conv3x3(x, W1, b1), W2, b2) into a single 5x5 conv (W5, b5).

    Shapes:
      W1: (C_mid, C_in, 3, 3), b1: (C_mid,)
      W2: (C_out, C_mid, 3, 3), b2: (C_out,)
    Returns:
      W5: (C_out, C_in, 5, 5), b5: (C_out,)
    """
    W1 = W1.astype(np.float64)
    b1 = b1.astype(np.float64)
    W2 = W2.astype(np.float64)
    b2 = b2.astype(np.float64)

    c_mid, c_in, _, _ = W1.shape
    c_out = W2.shape[0]
    W5 = np.zeros((c_out, c_in, 5, 5), dtype=np.float64)

    # Cross-correlation composition:
    # out[y, x] = sum_{dy2,dx2 in -1..1} W2[dy2+1, dx2+1] * sum_{dy1,dx1 in -1..1} W1[dy1+1, dx1+1] * x[y+dy2+dy1, x+dx2+dx1]
    for ky2 in range(3):
        for kx2 in range(3):
            for ky1 in range(3):
                for kx1 in range(3):
                    W5[:, :, ky1 + ky2, kx1 + kx2] += np.einsum(
                        "om,mi->oi", W2[:, :, ky2, kx2], W1[:, :, ky1, kx1]
                    )

    # Combined bias (exact in interior where all 9 taps of W2 see b1)
    b5 = b2 + np.einsum("omxy,m->o", W2, b1)
    return W5, b5


def _conv_nd_f64(x: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Reference float64 2D cross-correlation with zero padding='same'."""
    N, C, H, W_in = x.shape
    c_out, _, kh, kw = W.shape
    py, px = kh // 2, kw // 2
    xp = np.pad(x, ((0, 0), (0, 0), (py, py), (px, px)), mode="constant")
    out = np.zeros((N, c_out, H, W_in), dtype=np.float64)
    for ky in range(kh):
        for kx in range(kw):
            out += np.einsum("nchw,oc->nohw", xp[:, :, ky : ky + H, kx : kx + W_in], W[:, :, ky, kx])
    out += b[None, :, None, None]
    return out


def verify_collapse_interior_vs_border(
    W1: np.ndarray,
    b1: np.ndarray,
    W2: np.ndarray,
    b2: np.ndarray,
    H: int = 16,
    W: int = 16,
    seed: int = 0,
) -> Tuple[float, float]:
    """Return (max_interior_diff, max_border_diff) between sequential 3x3 and collapsed 5x5."""
    rng = np.random.default_rng(seed)
    c_in = W1.shape[1]
    x = rng.uniform(0.0, 1.0, size=(1, c_in, H, W)).astype(np.float64)

    y_mid = _conv_nd_f64(x, W1.astype(np.float64), b1.astype(np.float64))
    y_seq = _conv_nd_f64(y_mid, W2.astype(np.float64), b2.astype(np.float64))

    W5, b5 = collapse_two_3x3_to_5x5(W1, b1, W2, b2)
    y_5x5 = _conv_nd_f64(x, W5, b5)

    diff = np.abs(y_seq - y_5x5)
    interior_diff = float(np.max(diff[:, :, 2:-2, 2:-2]))
    mask = np.ones((H, W), dtype=bool)
    mask[2:-2, 2:-2] = False
    border_diff = float(np.max(diff[0, :, mask]))
    return interior_diff, border_diff
