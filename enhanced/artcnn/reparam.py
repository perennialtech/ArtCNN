# SPDX-FileCopyrightText: 2026 perennialtech contributors
# SPDX-License-Identifier: MIT

"""Structural re-parameterization (ECB / RepVGG-style foldable edge-conv blocks).

During training, each internal 3x3 layer has 5 parallel linear branches:
  1. Standard 3x3 convolution (W3, b3)
  2. 1x1 pointwise convolution (W1, b1)
  3. Identity shortcut (when c_in == c_out)
  4. Depthwise Sobel-X edge branch with learned per-channel scale + 1x1 projection
  5. Depthwise Sobel-Y edge branch with learned per-channel scale + 1x1 projection

Because all 5 branches are strictly linear, `fold_to_conv3x3()` collapses them
into a single standard 3x3 convolution `(W_eq, b_eq)` with 0% inference overhead.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from artcnn.models_np import ArtCNNC, Conv2D3x3


SOBEL_X_3X3 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
SOBEL_Y_3X3 = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)


def _conv2d_3x3_np(x: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Reference 3x3 conv (padding='same') in float64/float32."""
    N, C, H, W_in = x.shape
    c_out = W.shape[0]
    xp = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)), mode="constant")
    out = np.zeros((N, c_out, H, W_in), dtype=x.dtype)
    for ky in range(3):
        for kx in range(3):
            patch = xp[:, :, ky : ky + H, kx : kx + W_in]  # (N, C, H, W)
            # tensordot over C: (N, H, W, Cout) -> transpose
            out += np.einsum("nchw,oc->nohw", patch, W[:, :, ky, kx])
    out += b[None, :, None, None]
    return out


class FoldableEdgeConvBlock:
    """Multi-branch training block that folds losslessly into a single 3x3 conv."""

    def __init__(self, c_in: int, c_out: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.c_in = c_in
        self.c_out = c_out
        s3 = np.sqrt(2.0 / (c_in * 9))
        s1 = np.sqrt(2.0 / c_in)

        # Branch 1: 3x3 conv
        self.W3 = (rng.standard_normal((c_out, c_in, 3, 3)) * s3).astype(np.float32)
        self.b3 = (rng.standard_normal((c_out,)) * 0.01).astype(np.float32)

        # Branch 2: 1x1 conv
        self.W1 = (rng.standard_normal((c_out, c_in)) * s1 * 0.5).astype(np.float32)
        self.b1 = (rng.standard_normal((c_out,)) * 0.01).astype(np.float32)

        # Branch 3: Identity (active if c_in == c_out)
        self.use_identity = c_in == c_out

        # Branch 4 & 5: Learned 1x1 channel-mix followed by fixed Sobel-X / Sobel-Y
        # (Equivalent to 1x1 -> depthwise 3x3 Sobel; strictly linear!)
        self.Wsx = (rng.standard_normal((c_out, c_in)) * s1 * 0.25).astype(np.float32)
        self.Wsy = (rng.standard_normal((c_out, c_in)) * s1 * 0.25).astype(np.float32)

    def forward_multibranch(self, x: np.ndarray) -> np.ndarray:
        """Execute all 5 branches explicitly (training-time behavior)."""
        # Branch 1: 3x3
        y = _conv2d_3x3_np(x, self.W3, self.b3)

        # Branch 2: 1x1
        y1 = np.einsum("nchw,oc->nohw", x, self.W1) + self.b1[None, :, None, None]
        y = y + y1

        # Branch 3: Identity
        if self.use_identity:
            y = y + x

        # Branch 4: 1x1 -> depthwise Sobel-X (same-padded)
        hx = np.einsum("nchw,oc->nohw", x, self.Wsx)
        sx_kernel = np.zeros((self.c_out, self.c_out, 3, 3), dtype=x.dtype)
        for c in range(self.c_out):
            sx_kernel[c, c] = SOBEL_X_3X3
        y = y + _conv2d_3x3_np(hx, sx_kernel, np.zeros(self.c_out, dtype=x.dtype))

        # Branch 5: 1x1 -> depthwise Sobel-Y (same-padded)
        hy = np.einsum("nchw,oc->nohw", x, self.Wsy)
        sy_kernel = np.zeros((self.c_out, self.c_out, 3, 3), dtype=x.dtype)
        for c in range(self.c_out):
            sy_kernel[c, c] = SOBEL_Y_3X3
        y = y + _conv2d_3x3_np(hy, sy_kernel, np.zeros(self.c_out, dtype=x.dtype))
        return y

    def fold_to_conv3x3(self) -> Tuple[np.ndarray, np.ndarray]:
        """Mathematically collapse all 5 branches into a single (W_eq, b_eq) 3x3 conv."""
        W_eq = self.W3.astype(np.float64).copy()
        b_eq = (self.b3.astype(np.float64) + self.b1.astype(np.float64)).copy()

        # Fold 1x1 into center tap (1, 1)
        W_eq[:, :, 1, 1] += self.W1.astype(np.float64)

        # Fold identity into diagonal center tap
        if self.use_identity:
            for c in range(self.c_out):
                W_eq[c, c, 1, 1] += 1.0

        # Fold (1x1 Wsx -> depthwise Sobel-X) into outer product Wsx[:, :, None, None] * SOBEL_X
        W_eq += self.Wsx.astype(np.float64)[:, :, None, None] * SOBEL_X_3X3.astype(np.float64)[None, None, :, :]
        W_eq += self.Wsy.astype(np.float64)[:, :, None, None] * SOBEL_Y_3X3.astype(np.float64)[None, None, :, :]

        return W_eq.astype(np.float32), b_eq.astype(np.float32)

    def to_conv2d3x3(self) -> Conv2D3x3:
        """Return a standard deployed Conv2D3x3 with folded weights."""
        W_eq, b_eq = self.fold_to_conv3x3()
        layer = Conv2D3x3(self.c_in, self.c_out, np.random.default_rng(0))
        layer.W = W_eq
        layer.b = b_eq
        return layer


def fold_ecb_into_artcnn(model: ArtCNNC, ecb_blocks: list[FoldableEdgeConvBlock]) -> ArtCNNC:
    """Replace internal layers of an ArtCNNC with folded 3x3 weights from ECB blocks."""
    assert len(ecb_blocks) == len(model.internal)
    deployed = model.clone()
    for i, blk in enumerate(ecb_blocks):
        W_eq, b_eq = blk.fold_to_conv3x3()
        deployed.internal[i].W = W_eq
        deployed.internal[i].b = b_eq
    return deployed
