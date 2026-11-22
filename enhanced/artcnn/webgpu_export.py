# SPDX-FileCopyrightText: 2026 perennialtech contributors
# SPDX-License-Identifier: MIT

"""Compact WebGPU / browser bundle exporter for ArtCNN C-series models.

Exports an ArtCNNC model (optionally with grouped 4-channel spatial + pointwise
mixing kernels) into browser-friendly files:
  - <name>.json    : architecture metadata, per-layer byte offsets, and SHA-256s
  - <name>.bin     : 16-byte-aligned little-endian float32 weights for core WebGPU & CPU
  - <name>.f16.bin : 8-byte-aligned little-endian float16 weights (half size) for
                     adapters supporting WebGPU's `shader-f16` feature.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from artcnn.models_np import ArtCNNC, ArtCNNCConfig, Conv2D3x3


FORMAT_VERSION = 1


def _to_dtype_bytes(arr: np.ndarray, dtype: str) -> bytes:
    casted = np.ascontiguousarray(arr, dtype=dtype)
    if not np.all(np.isfinite(casted)):
        raise ValueError(f"Non-finite values produced when casting weights to {dtype}")
    return casted.tobytes()


def _pack_vec4_layer_first(layer: Conv2D3x3, dtype: str = "<f4") -> Tuple[bytes, Dict[str, Any]]:
    """Pack conv0 (1 -> F, 3x3): biases [F/4 x vec4], weights [F/4, 3, 3, vec4]."""
    if layer.c_in != 1 or layer.c_out % 4 != 0:
        raise ValueError(f"First layer expects c_in=1 and c_out%4==0, got {layer.c_in}->{layer.c_out}")
    out_v4 = layer.c_out // 4
    w = layer.W.reshape(out_v4, 4, 3, 3).transpose(0, 2, 3, 1)
    blob = _to_dtype_bytes(layer.b, dtype) + _to_dtype_bytes(w, dtype)
    meta = {
        "kind": "conv_in1",
        "c_in": 1,
        "c_out": layer.c_out,
        "bias_floats": int(layer.b.size),
        "weight_floats": int(w.size),
    }
    return blob, meta


def _pack_vec4_layer_dense(layer: Conv2D3x3, dtype: str = "<f4") -> Tuple[bytes, Dict[str, Any]]:
    """Pack dense 3x3 conv (Cin -> Cout, multiples of 4) for column-major mat4x4."""
    if layer.c_in % 4 != 0 or layer.c_out % 4 != 0:
        raise ValueError(f"Dense layer expects multiples of 4, got {layer.c_in}->{layer.c_out}")
    in_v4 = layer.c_in // 4
    out_v4 = layer.c_out // 4
    w = layer.W.reshape(out_v4, 4, in_v4, 4, 3, 3).transpose(0, 2, 4, 5, 3, 1)
    blob = _to_dtype_bytes(layer.b, dtype) + _to_dtype_bytes(w, dtype)
    meta = {
        "kind": "conv_dense",
        "c_in": layer.c_in,
        "c_out": layer.c_out,
        "bias_floats": int(layer.b.size),
        "weight_floats": int(w.size),
    }
    return blob, meta


def _pack_vec4_layer_grouped(
    layer: Conv2D3x3, spatial: np.ndarray, mixing: np.ndarray, dtype: str = "<f4"
) -> Tuple[bytes, Dict[str, Any]]:
    """Pack grouped 4-channel 3x3 spatial + 1x1 mixing conv for column-major mat4x4."""
    c_in, c_out = layer.c_in, layer.c_out
    if c_in != c_out or c_in % 4 != 0:
        raise ValueError(f"Grouped layer expects c_in==c_out and %4==0, got {c_in}->{c_out}")
    if spatial.shape != (c_in, 4, 3, 3) or mixing.shape != (c_out, c_in):
        raise ValueError(f"Unexpected grouped kernel shapes: {spatial.shape}, {mixing.shape}")
    g_v4 = c_in // 4
    sp = spatial.reshape(g_v4, 4, 4, 3, 3).transpose(0, 3, 4, 2, 1)
    mx = mixing.reshape(g_v4, 4, g_v4, 4).transpose(0, 2, 3, 1)
    blob = _to_dtype_bytes(layer.b, dtype) + _to_dtype_bytes(sp, dtype) + _to_dtype_bytes(mx, dtype)
    meta = {
        "kind": "conv_grouped",
        "c_in": c_in,
        "c_out": c_out,
        "bias_floats": int(layer.b.size),
        "spatial_floats": int(sp.size),
        "mixing_floats": int(mx.size),
    }
    return blob, meta


def export_artcnn_webgpu(
    model: ArtCNNC,
    name: str,
    grouped_kernels: Dict[str, Tuple[np.ndarray, np.ndarray]] | None = None,
    weights_filename: str | None = None,
    weights_f16_filename: str | None = None,
) -> Tuple[Dict[str, Any], bytes, bytes]:
    """Serialize an ArtCNNC model to a WebGPU manifest dict plus FP32 and FP16 binaries."""
    cfg = model.config
    grouped_kernels = grouped_kernels or {}
    layers = model.layers()
    chunks_f32: List[bytes] = []
    chunks_f16: List[bytes] = []
    layer_specs: List[Dict[str, Any]] = []
    byte_offset = 0
    byte_offset_f16 = 0

    for idx, (layer_name, layer) in enumerate(layers):
        if idx == 0:
            blob_f32, meta = _pack_vec4_layer_first(layer, "<f4")
            blob_f16, _ = _pack_vec4_layer_first(layer, "<f2")
            role = "entry"
            activation = "none"
        elif idx <= cfg.internal_layers:
            if layer_name in grouped_kernels:
                blob_f32, meta = _pack_vec4_layer_grouped(layer, *grouped_kernels[layer_name], "<f4")
                blob_f16, _ = _pack_vec4_layer_grouped(layer, *grouped_kernels[layer_name], "<f2")
            else:
                blob_f32, meta = _pack_vec4_layer_dense(layer, "<f4")
                blob_f16, _ = _pack_vec4_layer_dense(layer, "<f2")
            role = "internal"
            activation = cfg.activation
        elif idx == cfg.internal_layers + 1:
            if layer_name in grouped_kernels:
                blob_f32, meta = _pack_vec4_layer_grouped(layer, *grouped_kernels[layer_name], "<f4")
                blob_f16, _ = _pack_vec4_layer_grouped(layer, *grouped_kernels[layer_name], "<f2")
            else:
                blob_f32, meta = _pack_vec4_layer_dense(layer, "<f4")
                blob_f16, _ = _pack_vec4_layer_dense(layer, "<f2")
            role = "mix_global"
            activation = "none"
        else:
            blob_f32, meta = _pack_vec4_layer_dense(layer, "<f4")
            blob_f16, _ = _pack_vec4_layer_dense(layer, "<f2")
            role = "features_d2s"
            activation = "none"

        meta.update(
            {
                "name": layer_name,
                "role": role,
                "activation": activation,
                "byte_offset": byte_offset,
                "byte_length": len(blob_f32),
                "byte_offset_f16": byte_offset_f16,
                "byte_length_f16": len(blob_f16),
            }
        )
        chunks_f32.append(blob_f32)
        chunks_f16.append(blob_f16)
        layer_specs.append(meta)
        byte_offset += len(blob_f32)
        byte_offset_f16 += len(blob_f16)

    binary_f32 = b"".join(chunks_f32)
    binary_f16 = b"".join(chunks_f16)
    manifest: Dict[str, Any] = {
        "format": "artcnn-webgpu",
        "version": FORMAT_VERSION,
        "name": name,
        "architecture": f"C{cfg.internal_layers}F{cfg.filters}",
        "scale": 2,
        "color_space": "luma_bt709",
        "dtype": "float32",
        "filters": cfg.filters,
        "internal_layers": cfg.internal_layers,
        "activation": cfg.activation,
        "leaky_slope": float(cfg.leaky_slope),
        "input_skip": bool(cfg.input_skip),
        "weights_file": weights_filename or f"{name}.bin",
        "weights_bytes": len(binary_f32),
        "weights_sha256": hashlib.sha256(binary_f32).hexdigest(),
        "weights_f16_file": weights_f16_filename or f"{name}.f16.bin",
        "weights_f16_bytes": len(binary_f16),
        "weights_f16_sha256": hashlib.sha256(binary_f16).hexdigest(),
        "layers": layer_specs,
    }
    return manifest, binary_f32, binary_f16


def load_checkpoint(path: Path) -> Tuple[ArtCNNC, Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    """Load an enhanced/pareto .npz checkpoint and any grouped kernels."""
    with np.load(path) as weights:
        cfg = ArtCNNCConfig(**json.loads(weights["config"].item()))
        model = ArtCNNC(cfg)
        model.load_state_dict(weights)
        grouped = {
            name: (weights[f"{name}.spatial"].astype(np.float32), weights[f"{name}.mix"].astype(np.float32))
            for name, _ in model.layers()
            if f"{name}.spatial" in weights
        }
    return model, grouped


def write_webgpu_bundle(
    checkpoint_path: Path,
    output_dir: Path,
    name: str | None = None,
) -> Tuple[Path, Path, Path, Dict[str, Any]]:
    """Export a checkpoint .npz to <name>.json, <name>.bin (FP32), and <name>.f16.bin (FP16)."""
    model_name = name or checkpoint_path.stem
    model, grouped = load_checkpoint(checkpoint_path)
    manifest, binary_f32, binary_f16 = export_artcnn_webgpu(
        model,
        name=model_name,
        grouped_kernels=grouped,
        weights_filename=f"{model_name}.bin",
        weights_f16_filename=f"{model_name}.f16.bin",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{model_name}.json"
    bin_path = output_dir / f"{model_name}.bin"
    f16_path = output_dir / f"{model_name}.f16.bin"
    json_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    bin_path.write_bytes(binary_f32)
    f16_path.write_bytes(binary_f16)
    return json_path, bin_path, f16_path, manifest
