"""GLSL compute shader exporter for ArtCNN C-series (mpv / libplacebo compatible).

Supports both:
  1. Faithful baseline export (matching Artoriuz/ArtCNN's ArtCNN_C_Keras.ipynb):
     - Separate `conv2d_6` (4-channel intermediate texture) + `depth_to_space` pass
     - Plain ReLU (`max(result, V4(0.0))`)
  2. Enhanced lean runtime export:
     - `fuse_depth_to_space=True`: Fuses the final 2x2 pixel-shuffle directly into
       `conv2d_6` via 4 `imageStore` calls, eliminating an entire shader pass and
       an intermediate 4-channel texture write/read.
     - `activation="leaky_relu"`: Emits `max(result, result * F(slope))` at ~0 GPU cost.
     - `input_skip=True`: Adds bilinear `LUMA_tex` directly onto the 2x2 output pixels
       inside the fused final pass.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np

from artcnn.models_np import ArtCNNC, Conv2D3x3


LICENSE_HEADER = """// MIT License
// Based on ArtCNN (Copyright (c) 2024 Joao Chrisostomo, Kacper Michajlow)
// Enhanced exporter: fused final pass, LeakyReLU, and optional direct input skip.
"""

FP16_PREAMBLE = """#extension GL_EXT_shader_explicit_arithmetic_types_float16 : enable
#ifdef GL_EXT_shader_explicit_arithmetic_types_float16
# define V4 f16vec4
# define M4 f16mat4
# define F float16_t
#else
# define V4 vec4
# define M4 mat4
# define F float
#endif
"""


def find_rect(area: int) -> Tuple[int, int]:
    s = int(math.sqrt(area))
    for w in range(s, 0, -1):
        if area % w == 0:
            h = area // w
            return int(max(w, h)), int(min(w, h))
    return int(area), 1


def get_tile_off(i: int, w: int) -> Tuple[int, int]:
    return i % w, i // w


def _fmt_floats(arr: np.ndarray) -> str:
    return ", ".join(f"{float(v):.7g}" for v in arr.flatten())


def _emit_conv_pass(
    layer_name: str,
    prev_name: str,
    layer: Conv2D3x3,
    filters_label: int,
    has_activation: bool,
    activation: str,
    leaky_slope: float,
    is_add_global: bool = False,
    skip_name: str = "conv2d",
    mix_name: str = "conv2d_5",
    fuse_d2s: bool = False,
    input_skip: bool = False,
) -> str:
    c_in, c_out = layer.c_in, layer.c_out
    threads_w, threads_h = 12, 16
    tile_w, tile_h = find_rect(int(math.ceil(c_out / 4.0)))
    tile_in_w, tile_in_h = find_rect(int(math.ceil(c_in / 4.0)))

    act_tag = ""
    if has_activation:
        act_tag = "-LeakyReLU" if activation == "leaky_relu" else "-ReLU"
    if fuse_d2s:
        act_tag += "-FusedD2S" + ("-InputSkip" if input_skip else "")

    lines = []
    lines.append(f"//!DESC ArtCNN-Enhanced C4F{filters_label} ({layer_name}{act_tag})")
    if fuse_d2s:
        # Each thread computes 1 input pixel -> writes 2x2 output pixels directly
        lines.append(f"//!COMPUTE {threads_w * 2} {threads_h * 2} {threads_w} {threads_h}")
    else:
        lines.append(f"//!COMPUTE {threads_w * tile_w} {threads_h * tile_h} {threads_w} {threads_h}")
    lines.append("//!HOOK LUMA")

    if prev_name == "LUMA":
        lines.append("//!BIND LUMA")
    elif is_add_global:
        lines.append(f"//!BIND {skip_name}")
        lines.append(f"//!BIND {mix_name}")
        if fuse_d2s and input_skip:
            lines.append("//!BIND LUMA")
    else:
        lines.append(f"//!BIND {prev_name}")

    if fuse_d2s:
        lines.append("//!WIDTH LUMA.w 2.0 *")
        lines.append("//!HEIGHT LUMA.h 2.0 *")
    else:
        lines.append(f"//!SAVE {layer_name}")
        lines.append(f"//!WIDTH LUMA.w {float(tile_w)} *")
        lines.append(f"//!HEIGHT LUMA.h {float(tile_h)} *")
        lines.append("//!COMPONENTS 4")

    lines.append("//!WHEN OUTPUT.w LUMA.w / 1.3 > OUTPUT.h LUMA.h / 1.3 > *")
    lines.append(FP16_PREAMBLE.strip())

    storage = "F" if prev_name == "LUMA" else "V4"
    weights_storage = "V4" if prev_name == "LUMA" else "M4"
    load_suffix = ".x" if prev_name == "LUMA" else ""

    lines.append("const ivec2 ksize = ivec2(3, 3);")
    lines.append("const ivec2 offset = ksize / 2;")
    lines.append("const ivec2 wg_size = ivec2(gl_WorkGroupSize);")
    lines.append("const ivec2 isize = wg_size + ksize - 1;")
    lines.append(f"shared {storage} inp[{tile_in_w * tile_in_h}][isize.y][isize.x];")
    lines.append("void hook() {")
    lines.append("    const uvec2 local_xy = gl_LocalInvocationID.xy;")
    lines.append("    ivec2 base = ivec2(gl_WorkGroupID) * wg_size;")
    lines.append("    for (uint y = local_xy.y; y < isize.y; y += wg_size.y) {")
    lines.append("        for (uint x = local_xy.x; x < isize.x; x += wg_size.x) {")
    lines.append(f"            const ivec2 input_base = (base + ivec2(x, y) - offset) * ivec2({tile_in_w}, {tile_in_h});")
    for z in range(tile_in_w * tile_in_h):
        tx, ty = get_tile_off(z, tile_in_w)
        if is_add_global:
            lines.append(
                f"            inp[{z}][y][x] = {storage}({mix_name}_mul * texelFetch({mix_name}_raw, input_base + ivec2({tx}, {ty}), 0) + "
                f"{skip_name}_mul * texelFetch({skip_name}_raw, input_base + ivec2({tx}, {ty}), 0){load_suffix});"
            )
        else:
            lines.append(
                f"            inp[{z}][y][x] = {storage}({prev_name}_mul * texelFetch({prev_name}_raw, input_base + ivec2({tx}, {ty}), 0){load_suffix});"
            )
    lines.append("        }")
    lines.append("    }")
    lines.append("    barrier();")

    # Initialize accumulators with biases
    for p in range(tile_w * tile_h):
        b_slice = layer.b[p * 4 : (p + 1) * 4]
        lines.append(f"    V4 result{p} = V4({_fmt_floats(b_slice)});")

    # Unrolled 3x3 mat-vec accumulate (layer.W shape is [Cout, Cin, 3, 3])
    for z in range(tile_in_w * tile_in_h):
        for ky in range(3):
            for kx in range(3):
                lines.append(f"    const {storage} inp_{z}_{kx}_{ky} = inp[{z}][local_xy.y + {ky}][local_xy.x + {kx}];")
        for p in range(tile_w * tile_h):
            for ky in range(3):
                for kx in range(3):
                    # In GLSL, mat4(col0, col1, col2, col3) * v computes sum_j col_j * v_j
                    # So column j corresponds to input channel j, row i to output channel i.
                    if prev_name == "LUMA":
                        w_vec = layer.W[p * 4 : (p + 1) * 4, 0, ky, kx]
                        lines.append(f"    result{p} += V4({_fmt_floats(w_vec)}) * inp_{z}_{kx}_{ky};")
                    else:
                        w_block = layer.W[p * 4 : (p + 1) * 4, z * 4 : (z + 1) * 4, ky, kx]
                        # Transpose to column-major for GLSL mat4 constructor
                        lines.append(f"    result{p} += {weights_storage}({_fmt_floats(w_block.T)}) * inp_{z}_{kx}_{ky};")

    if fuse_d2s:
        # Directly write 2x2 output block to out_image (eliminates depth_to_space pass!)
        lines.append("    const ivec2 out_base = ivec2(gl_GlobalInvocationID.xy) * 2;")
        sub_coords = [(0, 0, "x"), (1, 0, "y"), (0, 1, "z"), (1, 1, "w")]
        for dx, dy, comp in sub_coords:
            if input_skip:
                lines.append(
                    f"    vec2 uv_{dx}_{dy} = (vec2(out_base + ivec2({dx}, {dy})) + vec2(0.5)) * (LUMA_pt * 0.5);"
                )
                lines.append(
                    f"    float v_{dx}_{dy} = clamp(float(result0.{comp}) + LUMA_tex(uv_{dx}_{dy}).x, 0.0, 1.0);"
                )
            else:
                lines.append(f"    float v_{dx}_{dy} = clamp(float(result0.{comp}), 0.0, 1.0);")
            lines.append(
                f"    imageStore(out_image, out_base + ivec2({dx}, {dy}), vec4(v_{dx}_{dy}, 0.0, 0.0, 1.0));"
            )
    else:
        lines.append(f"    const ivec2 output_base = ivec2(gl_GlobalInvocationID) * ivec2({tile_w}, {tile_h});")
        for p in range(tile_w * tile_h):
            tx, ty = get_tile_off(p, tile_w)
            if has_activation:
                if activation == "leaky_relu":
                    expr = f"max(result{p}, result{p} * F({leaky_slope:.6g}))"
                else:
                    expr = f"max(result{p}, V4(0.0))"
            else:
                expr = f"result{p}"
            lines.append(f"    imageStore(out_image, output_base + ivec2({tx}, {ty}), {expr});")

    lines.append("}\n")
    return "\n".join(lines)


def _emit_separate_d2s_pass(prev_name: str, filters_label: int) -> str:
    return "\n".join([
        f"//!DESC ArtCNN C4F{filters_label} (Depth-To-Space)",
        "//!COMPUTE 24 32 12 16",
        "//!HOOK LUMA",
        f"//!BIND {prev_name}",
        "//!WIDTH LUMA.w 2.0 *",
        "//!HEIGHT LUMA.h 2.0 *",
        "//!WHEN OUTPUT.w LUMA.w / 1.3 > OUTPUT.h LUMA.h / 1.3 > *",
        "void hook() {",
        "    vec4 result = vec4(0.0, 0.0, 0.0, 1.0);",
        f"    vec2 f0 = fract({prev_name}_pos * {prev_name}_size);",
        "    ivec2 i0 = ivec2(f0 * vec2(2.0));",
        f"    result.x = {prev_name}_tex((vec2(0.5) - f0) * {prev_name}_pt + {prev_name}_pos)[i0.y * 2 + i0.x];",
        "    imageStore(out_image, ivec2(gl_GlobalInvocationID), clamp(result, 0.0, 1.0));",
        "}\n",
    ])


def export_artcnn_glsl(
    model: ArtCNNC,
    fuse_depth_to_space: bool = True,
) -> str:
    """Export an ArtCNNC model to an mpv/libplacebo GLSL compute shader."""
    cfg = model.config
    F = cfg.filters
    parts = [LICENSE_HEADER]
    layers = model.layers()

    # conv0 (LUMA -> conv2d)
    parts.append(
        _emit_conv_pass(
            layer_name="conv2d",
            prev_name="LUMA",
            layer=layers[0][1],
            filters_label=F,
            has_activation=False,
            activation=cfg.activation,
            leaky_slope=cfg.leaky_slope,
        )
    )

    # internal conv1..convN (activated)
    for i in range(1, cfg.internal_layers + 1):
        name, layer = layers[i]
        prev_name = layers[i - 1][0]
        parts.append(
            _emit_conv_pass(
                layer_name=name,
                prev_name=prev_name,
                layer=layer,
                filters_label=F,
                has_activation=True,
                activation=cfg.activation,
                leaky_slope=cfg.leaky_slope,
            )
        )

    # mix_global (linear)
    mix_idx = cfg.internal_layers + 1
    mix_name, mix_layer = layers[mix_idx]
    parts.append(
        _emit_conv_pass(
            layer_name=mix_name,
            prev_name=layers[mix_idx - 1][0],
            layer=mix_layer,
            filters_label=F,
            has_activation=False,
            activation=cfg.activation,
            leaky_slope=cfg.leaky_slope,
        )
    )

    # features (add_global -> 4ch, optionally fused with depth_to_space + input_skip)
    feat_idx = cfg.internal_layers + 2
    feat_name, feat_layer = layers[feat_idx]
    parts.append(
        _emit_conv_pass(
            layer_name=feat_name,
            prev_name="add_global",
            layer=feat_layer,
            filters_label=F,
            has_activation=False,
            activation=cfg.activation,
            leaky_slope=cfg.leaky_slope,
            is_add_global=True,
            skip_name="conv2d",
            mix_name=mix_name,
            fuse_d2s=fuse_depth_to_space,
            input_skip=cfg.input_skip,
        )
    )

    if not fuse_depth_to_space:
        parts.append(_emit_separate_d2s_pass(feat_name, F))

    return "\n".join(parts)


def count_shader_passes(glsl_code: str) -> int:
    return glsl_code.count("//!DESC ")


def count_intermediate_textures(glsl_code: str) -> int:
    return glsl_code.count("//!SAVE ")
