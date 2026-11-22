# SPDX-FileCopyrightText: 2026 perennialtech contributors
# SPDX-License-Identifier: MIT

"""Faithful NumPy implementation of ArtCNN C-series with analytic backprop,
AdamW, EMA, LeakyReLU, and optional direct bilinear input residual skip.

Matches Artoriuz/ArtCNN's C4F16 / C4F32 architecture from ArtCNN_C_Keras.ipynb:
  conv0 (1->F, 3x3, linear)
  -> conv1..conv4 (F->F, 3x3, ReLU or LeakyReLU)
  -> mix_global (F->F, 3x3, linear) + conv0 (long skip)
  -> features (F->4, 3x3, linear)
  -> depth_to_space(2x) [+ optional bilinear input skip] -> clip(0, 1)
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


def im2col_3x3(x: np.ndarray) -> np.ndarray:
    """Extract 3x3 patches with zero-padding.
    x: (N, C, H, W) -> cols: (N, H, W, C*9)
    """
    N, C, H, W = x.shape
    xp = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)), mode="constant")
    # Build strided view: (N, C, 3, 3, H, W)
    sN, sC, sH, sW = xp.strides
    patches = np.lib.stride_tricks.as_strided(
        xp,
        shape=(N, C, 3, 3, H, W),
        strides=(sN, sC, sH, sW, sH, sW),
    )
    return patches.transpose(0, 4, 5, 1, 2, 3).reshape(N * H * W, C * 9)


def col2im_3x3(dcols: np.ndarray, shape: Tuple[int, int, int, int]) -> np.ndarray:
    """Inverse of im2col_3x3 accumulating gradients into padded input."""
    N, C, H, W = shape
    dxp = np.zeros((N, C, H + 2, W + 2), dtype=dcols.dtype)
    patches = dcols.reshape(N, H, W, C, 3, 3).transpose(0, 3, 4, 5, 1, 2)
    for ky in range(3):
        for kx in range(3):
            dxp[:, :, ky : ky + H, kx : kx + W] += patches[:, :, ky, kx, :, :]
    return dxp[:, :, 1:-1, 1:-1]


class Conv2D3x3:
    """3x3 2D convolution with padding='same' and analytic backward pass."""

    def __init__(self, c_in: int, c_out: int, rng: np.random.Generator, zero_init: bool = False):
        self.c_in = c_in
        self.c_out = c_out
        if zero_init:
            self.W = np.zeros((c_out, c_in, 3, 3), dtype=np.float32)
        else:
            scale = np.sqrt(2.0 / (c_in * 9))
            self.W = (rng.standard_normal((c_out, c_in, 3, 3)) * scale).astype(np.float32)
        self.b = np.zeros((c_out,), dtype=np.float32)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self._cache: Optional[Tuple[np.ndarray, Tuple[int, int, int, int]]] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        N, C, H, W = x.shape
        cols = im2col_3x3(x)  # (NHW, C*9)
        w_mat = self.W.reshape(self.c_out, C * 9)  # (Cout, C*9)
        out = cols @ w_mat.T + self.b  # (NHW, Cout)
        self._cache = (cols, x.shape)
        return out.reshape(N, H, W, self.c_out).transpose(0, 3, 1, 2)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        assert self._cache is not None
        cols, x_shape = self._cache
        N, C, H, W = x_shape
        dout_2d = dout.transpose(0, 2, 3, 1).reshape(N * H * W, self.c_out)
        self.dW = (dout_2d.T @ cols).reshape(self.W.shape)
        self.db = dout_2d.sum(axis=0)
        w_mat = self.W.reshape(self.c_out, C * 9)
        dcols = dout_2d @ w_mat
        return col2im_3x3(dcols, x_shape)

    def num_params(self) -> int:
        return int(self.W.size + self.b.size)

    def macs_per_pixel(self) -> int:
        return int(self.c_in * self.c_out * 9)


def depth_to_space_2x(x: np.ndarray) -> np.ndarray:
    """PixelShuffle(2): (N, 4, H, W) -> (N, 1, 2H, 2W)."""
    N, C, H, W = x.shape
    assert C == 4
    # Channel order matches tf.nn.depth_to_space(block_size=2):
    # [0]=(0,0), [1]=(0,1), [2]=(1,0), [3]=(1,1) -> (y_sub, x_sub)
    x = x.reshape(N, 2, 2, H, W)
    x = x.transpose(0, 3, 1, 4, 2)  # (N, H, 2, W, 2)
    return x.reshape(N, 1, H * 2, W * 2)


def space_to_depth_2x(dy: np.ndarray) -> np.ndarray:
    """Exact adjoint of depth_to_space_2x: (N, 1, 2H, 2W) -> (N, 4, H, W)."""
    N, C, H2, W2 = dy.shape
    assert C == 1 and H2 % 2 == 0 and W2 % 2 == 0
    H, W = H2 // 2, W2 // 2
    dy = dy.reshape(N, H, 2, W, 2)
    dy = dy.transpose(0, 2, 4, 1, 3)  # (N, 2, 2, H, W)
    return dy.reshape(N, 4, H, W)


def bilinear_upscale_2x(x: np.ndarray) -> np.ndarray:
    """Bilinear 2x upscale of (N, 1, H, W) -> (N, 1, 2H, 2W)."""
    N, _, H, W = x.shape
    out = np.empty((N, 1, H * 2, W * 2), dtype=np.float32)
    for i in range(N):
        out[i, 0] = cv2.resize(x[i, 0], (W * 2, H * 2), interpolation=cv2.INTER_LINEAR)
    return out


@dataclass
class ArtCNNCConfig:
    filters: int = 16
    internal_layers: int = 4
    activation: str = "relu"  # "relu" or "leaky_relu"
    leaky_slope: float = 0.1
    input_skip: bool = False  # direct bilinear input residual skip


class ArtCNNC:
    """ArtCNN C-series (C4F16 / C4F32) in NumPy with full training support."""

    def __init__(self, config: ArtCNNCConfig, seed: int = 42):
        self.config = config
        rng = np.random.default_rng(seed)
        F = config.filters
        self.conv0 = Conv2D3x3(1, F, rng)
        self.internal: List[Conv2D3x3] = [
            Conv2D3x3(F, F, rng) for _ in range(config.internal_layers)
        ]
        self.mix_global = Conv2D3x3(F, F, rng, zero_init=True)
        self.features = Conv2D3x3(F, 4, rng, zero_init=config.input_skip)
        self._act_caches: List[np.ndarray] = []
        self._last_base: Optional[np.ndarray] = None

    def layers(self) -> List[Tuple[str, Conv2D3x3]]:
        items: List[Tuple[str, Conv2D3x3]] = [("conv2d", self.conv0)]
        for i, layer in enumerate(self.internal, start=1):
            items.append((f"conv2d_{i}", layer))
        items.append((f"conv2d_{self.config.internal_layers + 1}", self.mix_global))
        items.append((f"conv2d_{self.config.internal_layers + 2}", self.features))
        return items

    def _act_forward(self, x: np.ndarray) -> np.ndarray:
        if self.config.activation == "leaky_relu":
            s = self.config.leaky_slope
            return np.where(x > 0.0, x, x * s).astype(np.float32)
        return np.maximum(x, 0.0).astype(np.float32)

    def _act_backward(self, x_pre: np.ndarray, dout: np.ndarray) -> np.ndarray:
        if self.config.activation == "leaky_relu":
            s = self.config.leaky_slope
            return np.where(x_pre > 0.0, dout, dout * s).astype(np.float32)
        return np.where(x_pre > 0.0, dout, 0.0).astype(np.float32)

    def forward(self, x: np.ndarray, clip: bool = False) -> np.ndarray:
        c0 = self.conv0.forward(x)
        h = c0
        self._act_caches = []
        for layer in self.internal:
            pre = layer.forward(h)
            self._act_caches.append(pre)
            h = self._act_forward(pre)
        mg = self.mix_global.forward(h)
        add_g = mg + c0
        feats = self.features.forward(add_g)
        out = depth_to_space_2x(feats)
        if self.config.input_skip:
            base = bilinear_upscale_2x(x)
            self._last_base = base
            out = out + base
        if clip:
            out = np.clip(out, 0.0, 1.0)
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        # Adjoint of depth_to_space_2x (bilinear input skip has zero param grad)
        dfeats = space_to_depth_2x(dout)
        dadd = self.features.backward(dfeats)
        dh = self.mix_global.backward(dadd)
        dc0 = dadd.copy()
        for layer, pre in reversed(list(zip(self.internal, self._act_caches))):
            dpre = self._act_backward(pre, dh)
            dh = layer.backward(dpre)
        dc0 = dc0 + dh
        return self.conv0.backward(dc0)

    def param_count(self) -> int:
        return sum(layer.num_params() for _, layer in self.layers())

    def macs_per_input_pixel(self) -> int:
        return sum(layer.macs_per_pixel() for _, layer in self.layers())

    def state_dict(self) -> Dict[str, np.ndarray]:
        sd: Dict[str, np.ndarray] = {}
        for name, layer in self.layers():
            sd[f"{name}.W"] = layer.W.copy()
            sd[f"{name}.b"] = layer.b.copy()
        return sd

    def load_state_dict(self, sd: Dict[str, np.ndarray]) -> None:
        for name, layer in self.layers():
            layer.W = sd[f"{name}.W"].astype(np.float32).copy()
            layer.b = sd[f"{name}.b"].astype(np.float32).copy()

    def clone(self) -> "ArtCNNC":
        m = ArtCNNC(copy.deepcopy(self.config))
        m.load_state_dict(self.state_dict())
        return m


class AdamWOptimizer:
    """AdamW with optional cosine annealing and EMA shadow weights."""

    def __init__(
        self,
        model: ArtCNNC,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        ema_decay: float = 0.99,
    ):
        self.model = model
        self.base_lr = lr
        self.lr = lr
        self.wd = weight_decay
        self.b1 = beta1
        self.b2 = beta2
        self.eps = eps
        self.ema_decay = ema_decay
        self.t = 0
        self.m: Dict[str, np.ndarray] = {}
        self.v: Dict[str, np.ndarray] = {}
        self.ema: Dict[str, np.ndarray] = model.state_dict()
        for k, val in self.ema.items():
            self.m[k] = np.zeros_like(val)
            self.v[k] = np.zeros_like(val)

    def set_cosine_lr(self, step: int, total_steps: int, min_lr_ratio: float = 0.05) -> float:
        progress = min(1.0, max(0.0, step / max(1, total_steps)))
        factor = min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (1.0 + np.cos(np.pi * progress))
        self.lr = float(self.base_lr * factor)
        return self.lr

    def step(self, grad_clip: float = 1.0) -> None:
        self.t += 1
        bc1 = 1.0 - self.b1 ** self.t
        bc2 = 1.0 - self.b2 ** self.t
        for name, layer in self.model.layers():
            for suffix, param, grad, apply_wd in (
                ("W", layer.W, layer.dW, True),
                ("b", layer.b, layer.db, False),
            ):
                key = f"{name}.{suffix}"
                g = np.clip(grad, -grad_clip, grad_clip)
                self.m[key] = self.b1 * self.m[key] + (1.0 - self.b1) * g
                self.v[key] = self.b2 * self.v[key] + (1.0 - self.b2) * (g * g)
                m_hat = self.m[key] / bc1
                v_hat = self.v[key] / bc2
                if apply_wd:
                    param *= 1.0 - self.lr * self.wd
                param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
                d_ema = min(self.ema_decay, (1.0 + self.t) / (10.0 + self.t))
                self.ema[key] = d_ema * self.ema[key] + (1.0 - d_ema) * param
