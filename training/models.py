# SPDX-FileCopyrightText: 2026 perennialtech contributors
# SPDX-License-Identifier: MIT

"""Clean-still RGB super-resolution models (C-series and R-series) for browser WebGPU."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Dict, List, Literal, Tuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from perceptual import FoldableECB, fp16_ste


@dataclass
class ArtCNNRGBConfig:
    family: Literal["c", "r"] = "c"
    filters: int = 16
    blocks: int = 4  # internal convs for "c"; residual blocks (2 convs each) for "r"
    in_channels: int = 3
    scale: int = 2
    activation: str = "prelu"
    leaky_slope: float = 0.1
    input_skip: bool = True

    def arch_name(self) -> str:
        prefix = "C" if self.family == "c" else "R"
        return f"{prefix}{self.blocks}F{self.filters}"


class _ConvBlock(nn.Module):
    """3x3 conv with optional FoldableECB training reparameterization."""

    def __init__(self, c_in: int, c_out: int, use_ecb: bool = False, zero_init: bool = False):
        super().__init__()
        self.use_ecb = use_ecb
        if use_ecb:
            self.conv = FoldableECB(c_in, c_out, dilation=1)
            if zero_init:
                with torch.no_grad():
                    for p in self.conv.parameters():
                        p.zero_()
        else:
            self.conv = nn.Conv2d(c_in, c_out, 3, padding=1)
            if zero_init:
                nn.init.zeros_(self.conv.weight)
                nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

    @torch.no_grad()
    def folded_wb(self) -> Tuple[np.ndarray, np.ndarray]:
        if isinstance(self.conv, FoldableECB):
            w, b = self.conv.fold_weights()
            return w.detach().cpu().numpy().astype(np.float32), b.detach().cpu().numpy().astype(np.float32)
        return (
            self.conv.weight.detach().cpu().numpy().astype(np.float32),
            self.conv.bias.detach().cpu().numpy().astype(np.float32),
        )


class ArtCNNRGB(nn.Module):
    """2x RGB super-resolution network exportable to browser WebGPU/CPU bundles."""

    def __init__(self, config: ArtCNNRGBConfig, use_ecb: bool = False):
        super().__init__()
        self.config = config
        self.use_ecb = use_ecb
        self.qat_fp16 = False
        F_ch = config.filters
        out_ch = config.in_channels * (config.scale * config.scale)  # 12 for 2x RGB

        self.conv0 = _ConvBlock(config.in_channels, F_ch, use_ecb=False)
        self.internal = nn.ModuleList()
        self.prelus = nn.ParameterList()

        if config.family == "c":
            for _ in range(config.blocks):
                self.internal.append(_ConvBlock(F_ch, F_ch, use_ecb=use_ecb))
                self.prelus.append(nn.Parameter(torch.full((F_ch,), config.leaky_slope)))
        elif config.family == "r":
            for _ in range(config.blocks):
                self.internal.append(_ConvBlock(F_ch, F_ch, use_ecb=use_ecb))
                self.prelus.append(nn.Parameter(torch.full((F_ch,), config.leaky_slope)))
                self.internal.append(_ConvBlock(F_ch, F_ch, use_ecb=use_ecb, zero_init=True))
        else:
            raise ValueError(f"Unsupported family: {config.family}")

        self.mix_global = _ConvBlock(F_ch, F_ch, use_ecb=False, zero_init=True)
        self.features = _ConvBlock(F_ch, out_ch, use_ecb=False, zero_init=config.input_skip)

    def _act(self, h: torch.Tensor, idx: int) -> torch.Tensor:
        if self.config.activation == "prelu":
            slope = self.prelus[idx].view(1, -1, 1, 1)
            return torch.where(h > 0.0, h, h * slope)
        if self.config.activation == "leaky_relu":
            return F.leaky_relu(h, negative_slope=self.config.leaky_slope)
        return F.relu(h)

    def _run_conv(self, block: _ConvBlock, x: torch.Tensor) -> torch.Tensor:
        if self.qat_fp16 and self.training and not block.use_ecb:
            w = fp16_ste(block.conv.weight)
            b = fp16_ste(block.conv.bias)
            return F.conv2d(x, w, b, padding=1)
        out = block(x)
        return fp16_ste(out) if (self.qat_fp16 and self.training) else out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c0 = self._run_conv(self.conv0, x)
        h = c0
        if self.config.family == "c":
            for idx, layer in enumerate(self.internal):
                h = self._act(self._run_conv(layer, h), idx)
        else:
            for b_idx in range(self.config.blocks):
                c1 = self.internal[b_idx * 2]
                c2 = self.internal[b_idx * 2 + 1]
                r = h
                h = self._act(self._run_conv(c1, h), b_idx)
                h = self._run_conv(c2, h) + r
        mg = self._run_conv(self.mix_global, h)
        feats = self._run_conv(self.features, mg + c0)
        out = F.pixel_shuffle(feats, self.config.scale)
        if self.config.input_skip:
            out = out + F.interpolate(x, scale_factor=self.config.scale, mode="bilinear", align_corners=False)
        return out

    @torch.no_grad()
    def export_npz(self, path: Path) -> None:
        sd: Dict[str, np.ndarray] = {"config": np.array(json.dumps(asdict(self.config)))}
        w0, b0 = self.conv0.folded_wb()
        sd["conv0.W"], sd["conv0.b"] = w0, b0
        for i, layer in enumerate(self.internal):
            w, b = layer.folded_wb()
            sd[f"internal.{i}.W"], sd[f"internal.{i}.b"] = w, b
        for i, p in enumerate(self.prelus):
            sd[f"prelu.{i}"] = p.detach().cpu().numpy().astype(np.float32)
        wm, bm = self.mix_global.folded_wb()
        sd["mix_global.W"], sd["mix_global.b"] = wm, bm
        wf, bf = self.features.folded_wb()
        sd["features.W"], sd["features.b"] = wf, bf
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **sd)

    @classmethod
    def from_npz(cls, path: Path) -> "ArtCNNRGB":
        with np.load(path) as data:
            cfg = ArtCNNRGBConfig(**json.loads(str(data["config"].item())))
            model = cls(cfg, use_ecb=False)
            with torch.no_grad():
                model.conv0.conv.weight.copy_(torch.from_numpy(data["conv0.W"]))
                model.conv0.conv.bias.copy_(torch.from_numpy(data["conv0.b"]))
                for i, layer in enumerate(model.internal):
                    layer.conv.weight.copy_(torch.from_numpy(data[f"internal.{i}.W"]))
                    layer.conv.bias.copy_(torch.from_numpy(data[f"internal.{i}.b"]))
                for i, p in enumerate(model.prelus):
                    p.copy_(torch.from_numpy(data[f"prelu.{i}"]))
                model.mix_global.conv.weight.copy_(torch.from_numpy(data["mix_global.W"]))
                model.mix_global.conv.bias.copy_(torch.from_numpy(data["mix_global.b"]))
                model.features.conv.weight.copy_(torch.from_numpy(data["features.W"]))
                model.features.conv.bias.copy_(torch.from_numpy(data["features.b"]))
        return model
