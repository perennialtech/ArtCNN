# SPDX-FileCopyrightText: 2026 perennialtech contributors
# SPDX-License-Identifier: MIT

"""Lightweight luma-guided 4:2:0 -> 4:4:4 chroma upscaler (PyTorch + mpv/libplacebo GLSL).

Takes 2x2 pixel-unshuffled full-resolution LUMA (4 channels) concatenated with native
half-resolution CHROMA (Cb, Cr = 2 channels) -> 6 input channels at CHROMA resolution,
predicts 2x2 chroma residuals added onto bilinear CHROMA_tex, and exports a fused
mpv `//!HOOK CHROMA` compute shader with `//!BIND LUMA` + `//!BIND CHROMA`.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from artcnn.glsl_export import FP16_PREAMBLE, LICENSE_HEADER, _fmt_floats, find_rect, get_tile_off


@dataclass
class ChromaConfig:
    filters: int = 12
    internal_layers: int = 2
    leaky_slope: float = 0.1


class LumaGuidedChromaNet(nn.Module):
    """Compact luma-guided 2x chroma upscaler operating at half-res (CHROMA grid)."""

    def __init__(self, config: ChromaConfig | None = None):
        super().__init__()
        self.config = config or ChromaConfig()
        f = self.config.filters
        self.conv_in = nn.Conv2d(6, f, 3, padding=1)
        self.internal = nn.ModuleList([nn.Conv2d(f, f, 3, padding=1) for _ in range(self.config.internal_layers)])
        self.conv_mix = nn.Conv2d(f, f, 3, padding=1)
        self.conv_out = nn.Conv2d(f, 8, 3, padding=1)
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)

    def forward(self, luma_hr: torch.Tensor, chroma_lr: torch.Tensor) -> torch.Tensor:
        """luma_hr: (N, 1, 2H, 2W), chroma_lr: (N, 2, H, W) -> chroma_hr: (N, 2, 2H, 2W)."""
        # Pixel-unshuffle luma 2x2 -> (N, 4, H, W) ordered (y0x0, y0x1, y1x0, y1x1)
        y00 = luma_hr[:, :, 0::2, 0::2]
        y01 = luma_hr[:, :, 0::2, 1::2]
        y10 = luma_hr[:, :, 1::2, 0::2]
        y11 = luma_hr[:, :, 1::2, 1::2]
        inp = torch.cat([chroma_lr, y00, y01, y10, y11], dim=1)
        skip = self.conv_in(inp)
        h = skip
        for layer in self.internal:
            h = F.leaky_relu(layer(h), self.config.leaky_slope)
        res = self.conv_out(self.conv_mix(h) + skip)
        # Reshape (N, 8, H, W) -> 2 channels x (2x2) pixel shuffle
        cb = F.pixel_shuffle(res[:, 0:4], 2)
        cr = F.pixel_shuffle(res[:, 4:8], 2)
        base = F.interpolate(chroma_lr, scale_factor=2, mode="bicubic", align_corners=False).clamp(0.0, 1.0)
        return base + torch.cat([cb, cr], dim=1)

    def save_npz(self, path: Path) -> None:
        state = {
            "config": json.dumps(asdict(self.config)),
            "conv_in.W": self.conv_in.weight.detach().cpu().numpy(),
            "conv_in.b": self.conv_in.bias.detach().cpu().numpy(),
            "conv_mix.W": self.conv_mix.weight.detach().cpu().numpy(),
            "conv_mix.b": self.conv_mix.bias.detach().cpu().numpy(),
            "conv_out.W": self.conv_out.weight.detach().cpu().numpy(),
            "conv_out.b": self.conv_out.bias.detach().cpu().numpy(),
        }
        for i, layer in enumerate(self.internal):
            state[f"internal_{i}.W"] = layer.weight.detach().cpu().numpy()
            state[f"internal_{i}.b"] = layer.bias.detach().cpu().numpy()
        np.savez(path, **state)

    @classmethod
    def load_npz(cls, path: Path) -> "LumaGuidedChromaNet":
        with np.load(path, allow_pickle=False) as d:
            cfg = ChromaConfig(**json.loads(str(d["config"])))
            net = cls(cfg)
            with torch.no_grad():
                net.conv_in.weight.copy_(torch.from_numpy(d["conv_in.W"]))
                net.conv_in.bias.copy_(torch.from_numpy(d["conv_in.b"]))
                for i, layer in enumerate(net.internal):
                    layer.weight.copy_(torch.from_numpy(d[f"internal_{i}.W"]))
                    layer.bias.copy_(torch.from_numpy(d[f"internal_{i}.b"]))
                net.conv_mix.weight.copy_(torch.from_numpy(d["conv_mix.W"]))
                net.conv_mix.bias.copy_(torch.from_numpy(d["conv_mix.b"]))
                net.conv_out.weight.copy_(torch.from_numpy(d["conv_out.W"]))
                net.conv_out.bias.copy_(torch.from_numpy(d["conv_out.b"]))
        return net


def export_chroma_glsl(net: LumaGuidedChromaNet, label: str = "C2F12_Chroma", workgroup: Tuple[int, int] = (8, 8)) -> str:
    """Export LumaGuidedChromaNet as an mpv/libplacebo `//!HOOK CHROMA` compute shader."""
    cfg = net.config
    f = cfg.filters
    wg_x, wg_y = workgroup
    tile_w, tile_h = find_rect(f // 4)
    passes: list[str] = [LICENSE_HEADER]

    def emit_first() -> str:
        W = net.conv_in.weight.detach().cpu().numpy()
        b = net.conv_in.bias.detach().cpu().numpy()
        n_out = f // 4
        lines = [
            f"//!DESC ArtCNN-Enhanced {label} (conv_in)",
            f"//!COMPUTE {wg_x * tile_w} {wg_y * tile_h} {wg_x} {wg_y}",
            "//!HOOK CHROMA",
            "//!BIND CHROMA",
            "//!BIND LUMA",
            "//!SAVE chroma_in",
            f"//!WIDTH CHROMA.w {float(tile_w)}",
            f"//!HEIGHT CHROMA.h {float(tile_h)}",
            "//!COMPONENTS 4",
            "//!WHEN LUMA.w CHROMA.w / 1.5 > LUMA.h CHROMA.h / 1.5 > *",
            FP16_PREAMBLE.strip(),
            "const ivec2 wg_size = ivec2(gl_WorkGroupSize);",
            "const ivec2 isize = wg_size + ivec2(2, 2);",
            "shared V4 inp0[isize.y][isize.x];",
            "shared V4 inp1[isize.y][isize.x];",
            "void hook() {",
            "    const uvec2 local_xy = gl_LocalInvocationID.xy;",
            "    ivec2 base = ivec2(gl_WorkGroupID) * wg_size;",
            "    for (uint y = local_xy.y; y < isize.y; y += wg_size.y) {",
            "        for (uint x = local_xy.x; x < isize.x; x += wg_size.x) {",
            "            ivec2 cp = base + ivec2(x, y) - ivec2(1, 1);",
            "            vec2 ch = CHROMA_mul * texelFetch(CHROMA_raw, cp, 0).xy;",
            "            ivec2 lp = cp * 2;",
            "            F y00 = F(LUMA_mul * texelFetch(LUMA_raw, lp + ivec2(0, 0), 0).x);",
            "            F y01 = F(LUMA_mul * texelFetch(LUMA_raw, lp + ivec2(1, 0), 0).x);",
            "            F y10 = F(LUMA_mul * texelFetch(LUMA_raw, lp + ivec2(0, 1), 0).x);",
            "            F y11 = F(LUMA_mul * texelFetch(LUMA_raw, lp + ivec2(1, 1), 0).x);",
            "            inp0[y][x] = V4(F(ch.x), F(ch.y), y00, y01);",
            "            inp1[y][x] = V4(y10, y11, F(0.0), F(0.0));",
            "        }",
            "    }",
            "    barrier();",
        ]
        for o in range(n_out):
            lines.append(f"    V4 res{o} = V4({_fmt_floats(b[o*4:(o+1)*4])});")
        for ky in range(3):
            for kx in range(3):
                lines.append(f"    V4 p0_{ky}_{kx} = inp0[local_xy.y + {ky}][local_xy.x + {kx}];")
                lines.append(f"    V4 p1_{ky}_{kx} = inp1[local_xy.y + {ky}][local_xy.x + {kx}];")
                for o in range(n_out):
                    m0 = W[o*4:(o+1)*4, 0:4, ky, kx].T
                    m1 = np.zeros((4, 4), dtype=np.float32)
                    m1[:2, :] = W[o*4:(o+1)*4, 4:6, ky, kx].T
                    lines.append(f"    res{o} = fma(M4({_fmt_floats(m0)}), p0_{ky}_{kx}, res{o});")
                    lines.append(f"    res{o} = fma(M4({_fmt_floats(m1)}), p1_{ky}_{kx}, res{o});")
        for o in range(n_out):
            tx, ty = get_tile_off(o, tile_w)
            lines.append(f"    imageStore(out_image, ivec2(gl_GlobalInvocationID) + ivec2(CHROMA_size) * ivec2({tx}, {ty}), vec4(res{o}));")
        lines.append("}")
        return "\n".join(lines) + "\n"

    def emit_mid(name: str, prev: str, layer: nn.Conv2d, act: bool, add_skip: bool = False) -> str:
        W = layer.weight.detach().cpu().numpy()
        b = layer.bias.detach().cpu().numpy()
        n = f // 4
        lines = [
            f"//!DESC ArtCNN-Enhanced {label} ({name})",
            f"//!COMPUTE {wg_x * tile_w} {wg_y * tile_h} {wg_x} {wg_y}",
            "//!HOOK CHROMA",
            f"//!BIND {prev}",
            *(["//!BIND chroma_in"] if add_skip else []),
            f"//!SAVE {name}",
            f"//!WIDTH CHROMA.w {float(tile_w)}",
            f"//!HEIGHT CHROMA.h {float(tile_h)}",
            "//!COMPONENTS 4",
            "//!WHEN LUMA.w CHROMA.w / 1.5 > LUMA.h CHROMA.h / 1.5 > *",
            FP16_PREAMBLE.strip(),
            "const ivec2 wg_size = ivec2(gl_WorkGroupSize);",
            "const ivec2 isize = wg_size + ivec2(2, 2);",
            f"shared V4 inp[{n}][isize.y][isize.x];",
            "void hook() {",
            "    const uvec2 local_xy = gl_LocalInvocationID.xy;",
            "    ivec2 base = ivec2(gl_WorkGroupID) * wg_size;",
            "    ivec2 tile_sz = ivec2(CHROMA_size);",
            "    for (uint y = local_xy.y; y < isize.y; y += wg_size.y) {",
            "        for (uint x = local_xy.x; x < isize.x; x += wg_size.x) {",
            "            ivec2 cp = clamp(base + ivec2(x, y) - ivec2(1, 1), ivec2(0), tile_sz - ivec2(1));",
        ]
        for i in range(n):
            tx, ty = get_tile_off(i, tile_w)
            lines.append(f"            inp[{i}][y][x] = V4({prev}_mul * texelFetch({prev}_raw, cp + tile_sz * ivec2({tx}, {ty}), 0));")
        lines.extend(["        }", "    }", "    barrier();"])
        for o in range(n):
            lines.append(f"    V4 res{o} = V4({_fmt_floats(b[o*4:(o+1)*4])});")
        for ky in range(3):
            for kx in range(3):
                for i in range(n):
                    lines.append(f"    V4 v{i} = inp[{i}][local_xy.y + {ky}][local_xy.x + {kx}];")
                for o in range(n):
                    for i in range(n):
                        m = W[o*4:(o+1)*4, i*4:(i+1)*4, ky, kx].T
                        lines.append(f"    res{o} = fma(M4({_fmt_floats(m)}), v{i}, res{o});")
        for o in range(n):
            tx, ty = get_tile_off(o, tile_w)
            if act:
                lines.append(f"    res{o} = max(res{o}, res{o} * F({cfg.leaky_slope:.7g}));")
            if add_skip:
                lines.append(f"    res{o} += V4(chroma_in_mul * texelFetch(chroma_in_raw, ivec2(gl_GlobalInvocationID) + tile_sz * ivec2({tx}, {ty}), 0));")
            lines.append(f"    imageStore(out_image, ivec2(gl_GlobalInvocationID) + tile_sz * ivec2({tx}, {ty}), vec4(res{o}));")
        lines.append("}")
        return "\n".join(lines) + "\n"

    def emit_final(prev: str) -> str:
        W = net.conv_out.weight.detach().cpu().numpy()
        b = net.conv_out.bias.detach().cpu().numpy()
        n = f // 4
        lines = [
            f"//!DESC ArtCNN-Enhanced {label} (conv_out-FusedD2S-InputSkip)",
            f"//!COMPUTE {wg_x * 2} {wg_y * 2} {wg_x} {wg_y}",
            "//!HOOK CHROMA",
            "//!BIND CHROMA",
            f"//!BIND {prev}",
            "//!WIDTH LUMA.w",
            "//!HEIGHT LUMA.h",
            "//!COMPONENTS 2",
            "//!WHEN LUMA.w CHROMA.w / 1.5 > LUMA.h CHROMA.h / 1.5 > *",
            FP16_PREAMBLE.strip(),
            "const ivec2 wg_size = ivec2(gl_WorkGroupSize);",
            "const ivec2 isize = wg_size + ivec2(2, 2);",
            f"shared V4 inp[{n}][isize.y][isize.x];",
            "void hook() {",
            "    const uvec2 local_xy = gl_LocalInvocationID.xy;",
            "    ivec2 base = ivec2(gl_WorkGroupID) * wg_size;",
            "    ivec2 tile_sz = ivec2(CHROMA_size);",
            "    for (uint y = local_xy.y; y < isize.y; y += wg_size.y) {",
            "        for (uint x = local_xy.x; x < isize.x; x += wg_size.x) {",
            "            ivec2 cp = clamp(base + ivec2(x, y) - ivec2(1, 1), ivec2(0), tile_sz - ivec2(1));",
        ]
        for i in range(n):
            tx, ty = get_tile_off(i, tile_w)
            lines.append(f"            inp[{i}][y][x] = V4({prev}_mul * texelFetch({prev}_raw, cp + tile_sz * ivec2({tx}, {ty}), 0));")
        lines.extend(["        }", "    }", "    barrier();"])
        lines.append(f"    V4 cb = V4({_fmt_floats(b[0:4])});")
        lines.append(f"    V4 cr = V4({_fmt_floats(b[4:8])});")
        for ky in range(3):
            for kx in range(3):
                for i in range(n):
                    lines.append(f"    V4 v{i} = inp[{i}][local_xy.y + {ky}][local_xy.x + {kx}];")
                for i in range(n):
                    mcb = W[0:4, i*4:(i+1)*4, ky, kx].T
                    mcr = W[4:8, i*4:(i+1)*4, ky, kx].T
                    lines.append(f"    cb = fma(M4({_fmt_floats(mcb)}), v{i}, cb);")
                    lines.append(f"    cr = fma(M4({_fmt_floats(mcr)}), v{i}, cr);")
        # Exact 2x Catmull-Rom bicubic (matching PyTorch F.interpolate(..., mode="bicubic", align_corners=False))
        # For 2x scale, sub-pixel offsets from center pixel p are -0.25 (left/top) and +0.25 (right/bottom).
        # Catmull-Rom weights at t=0.25: [-9/128, 111/128, 29/128, -3/128] (and reversed for t=0.75 / -0.25).
        lines.extend([
            "    ivec2 c_xy = ivec2(gl_GlobalInvocationID);",
            "    ivec2 c_max = tile_sz - ivec2(1);",
            "    // Notice: for subpixel 0 (left/top, -0.25 from c_xy), taps are c_xy + {-2,-1,0,+1};",
            "    // for subpixel 1 (right/bottom, +0.25 from c_xy), taps are c_xy + {-1,0,+1,+2}.",
            "    const float w_neg[4] = float[4](-0.0234375, 0.2265625, 0.8671875, -0.0703125);",
            "    const float w_pos[4] = float[4](-0.0703125, 0.8671875, 0.2265625, -0.0234375);",
            "    vec2 ch5[5][5];",
            "    for (int dy = -2; dy <= 2; ++dy) {",
            "        for (int dx = -2; dx <= 2; ++dx) {",
            "            ivec2 q = clamp(c_xy + ivec2(dx, dy), ivec2(0), c_max);",
            "            ch5[dy + 2][dx + 2] = CHROMA_mul * texelFetch(CHROMA_raw, q, 0).xy;",
            "        }",
            "    }",
            "    vec2 b00 = vec2(0.0); vec2 b01 = vec2(0.0); vec2 b10 = vec2(0.0); vec2 b11 = vec2(0.0);",
            "    for (int j = 0; j < 4; ++j) {",
            "        vec2 r00 = vec2(0.0); vec2 r01 = vec2(0.0); vec2 r10 = vec2(0.0); vec2 r11 = vec2(0.0);",
            "        for (int i = 0; i < 4; ++i) {",
            "            r00 += w_neg[i] * ch5[j][i];",
            "            r01 += w_pos[i] * ch5[j][i + 1];",
            "            r10 += w_neg[i] * ch5[j + 1][i];",
            "            r11 += w_pos[i] * ch5[j + 1][i + 1];",
            "        }",
            "        b00 += w_neg[j] * r00;",
            "        b01 += w_neg[j] * r01;",
            "        b10 += w_pos[j] * r10;",
            "        b11 += w_pos[j] * r11;",
            "    }",
            "    ivec2 out_xy = c_xy * 2;",
            "    imageStore(out_image, out_xy + ivec2(0, 0), vec4(clamp(clamp(b00, 0.0, 1.0) + vec2(cb.x, cr.x), 0.0, 1.0), 0.0, 1.0));",
            "    imageStore(out_image, out_xy + ivec2(1, 0), vec4(clamp(clamp(b01, 0.0, 1.0) + vec2(cb.y, cr.y), 0.0, 1.0), 0.0, 1.0));",
            "    imageStore(out_image, out_xy + ivec2(0, 1), vec4(clamp(clamp(b10, 0.0, 1.0) + vec2(cb.z, cr.z), 0.0, 1.0), 0.0, 1.0));",
            "    imageStore(out_image, out_xy + ivec2(1, 1), vec4(clamp(clamp(b11, 0.0, 1.0) + vec2(cb.w, cr.w), 0.0, 1.0), 0.0, 1.0));",
            "}",
        ])
        return "\n".join(lines) + "\n"

    passes.append(emit_first())
    prev = "chroma_in"
    for idx, layer in enumerate(net.internal):
        name = f"chroma_h{idx}"
        passes.append(emit_mid(name, prev, layer, act=True, add_skip=False))
        prev = name
    passes.append(emit_mid("chroma_mix", prev, net.conv_mix, act=False, add_skip=True))
    passes.append(emit_final("chroma_mix"))
    return "\n".join(passes)
