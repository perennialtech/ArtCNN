# SPDX-FileCopyrightText: 2026 perennialtech contributors
# SPDX-License-Identifier: MIT

"""Compact WebGPU / browser bundle exporter for ArtCNN clean-still RGB models."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from dataclasses import dataclass

@dataclass
class ArtCNNRGBConfig:
    family: str = "c"
    filters: int = 16
    blocks: int = 4
    in_channels: int = 3
    scale: int = 2
    activation: str = "prelu"
    leaky_slope: float = 0.1
    input_skip: bool = True

    def arch_name(self) -> str:
        prefix = "C" if self.family == "c" else "R"
        return f"{prefix}{self.blocks}F{self.filters}"


FORMAT_VERSION = 2


def _to_dtype_bytes(arr: np.ndarray, dtype: str) -> bytes:
    casted = np.ascontiguousarray(arr, dtype=dtype)
    if not np.all(np.isfinite(casted)):
        raise ValueError(f"Non-finite values produced when casting weights to {dtype}")
    return casted.tobytes()


def _pack_vec4_layer_in3(W: np.ndarray, b: np.ndarray, dtype: str = "<f4") -> Tuple[bytes, Dict[str, Any]]:
    """Pack conv0 (3 -> F, 3x3): biases [F/4 x vec4], weights [F/4, 3, 3, 3, vec4]."""
    c_out, c_in, ky, kx = W.shape
    if c_in != 3 or c_out % 4 != 0 or ky != 3 or kx != 3:
        raise ValueError(f"First layer expects (F%4==0, 3, 3, 3), got {W.shape}")
    out_v4 = c_out // 4
    w = W.reshape(out_v4, 4, 3, 3, 3).transpose(0, 3, 4, 2, 1)  # (out_v4, ky, kx, in_ch, 4)
    blob = _to_dtype_bytes(b, dtype) + _to_dtype_bytes(w, dtype)
    meta = {
        "kind": "conv_in3",
        "c_in": 3,
        "c_out": int(c_out),
        "bias_floats": int(b.size),
        "weight_floats": int(w.size),
    }
    return blob, meta


def _pack_vec4_layer_dense(W: np.ndarray, b: np.ndarray, dtype: str = "<f4") -> Tuple[bytes, Dict[str, Any]]:
    """Pack dense 3x3 conv (Cin -> Cout, multiples of 4) for column-major mat4x4."""
    c_out, c_in, ky, kx = W.shape
    if c_in % 4 != 0 or c_out % 4 != 0 or ky != 3 or kx != 3:
        raise ValueError(f"Dense layer expects multiples of 4, got {W.shape}")
    in_v4 = c_in // 4
    out_v4 = c_out // 4
    w = W.reshape(out_v4, 4, in_v4, 4, 3, 3).transpose(0, 2, 4, 5, 3, 1)
    blob = _to_dtype_bytes(b, dtype) + _to_dtype_bytes(w, dtype)
    meta = {
        "kind": "conv_dense",
        "c_in": int(c_in),
        "c_out": int(c_out),
        "bias_floats": int(b.size),
        "weight_floats": int(w.size),
    }
    return blob, meta


def export_npz_to_webgpu(
    ckpt_path: Path,
    name: str | None = None,
    weights_filename: str | None = None,
    weights_f16_filename: str | None = None,
) -> Tuple[Dict[str, Any], bytes, bytes]:
    """Serialize an ArtCNNRGB .npz checkpoint to a WebGPU manifest + FP32 + FP16 binaries."""
    model_name = name or ckpt_path.stem
    with np.load(ckpt_path) as data:
        cfg = ArtCNNRGBConfig(**json.loads(str(data["config"].item())))
        raw_layers: List[Tuple[str, str, str, np.ndarray, np.ndarray, List[float] | None, bool]] = []

        raw_layers.append(("conv0", "entry", "none", data["conv0.W"], data["conv0.b"], None, False))

        if cfg.family == "c":
            for i in range(cfg.blocks):
                slopes = data[f"prelu.{i}"].astype(np.float32).tolist() if cfg.activation == "prelu" else None
                raw_layers.append(
                    (f"conv{i + 1}", "internal", cfg.activation, data[f"internal.{i}.W"], data[f"internal.{i}.b"], slopes, False)
                )
        else:
            for b_idx in range(cfg.blocks):
                i1 = b_idx * 2
                i2 = b_idx * 2 + 1
                slopes = data[f"prelu.{b_idx}"].astype(np.float32).tolist() if cfg.activation == "prelu" else None
                raw_layers.append(
                    (f"res{b_idx}_conv1", "res_conv1", cfg.activation, data[f"internal.{i1}.W"], data[f"internal.{i1}.b"], slopes, False)
                )
                raw_layers.append(
                    (f"res{b_idx}_conv2", "res_conv2", "none", data[f"internal.{i2}.W"], data[f"internal.{i2}.b"], None, True)
                )

        raw_layers.append(("mix_global", "mix_global", "none", data["mix_global.W"], data["mix_global.b"], None, False))
        raw_layers.append(("features", "features_d2s", "none", data["features.W"], data["features.b"], None, False))

    chunks_f32: List[bytes] = []
    chunks_f16: List[bytes] = []
    layer_specs: List[Dict[str, Any]] = []
    byte_offset = 0
    byte_offset_f16 = 0

    for layer_name, role, activation, W, b, slopes, residual_add in raw_layers:
        if role == "entry":
            blob_f32, meta = _pack_vec4_layer_in3(W, b, "<f4")
            blob_f16, _ = _pack_vec4_layer_in3(W, b, "<f2")
        else:
            blob_f32, meta = _pack_vec4_layer_dense(W, b, "<f4")
            blob_f16, _ = _pack_vec4_layer_dense(W, b, "<f2")

        meta.update(
            {
                "name": layer_name,
                "role": role,
                "activation": activation,
                "residual_add": residual_add,
                "byte_offset": byte_offset,
                "byte_length": len(blob_f32),
                "byte_offset_f16": byte_offset_f16,
                "byte_length_f16": len(blob_f16),
            }
        )
        if slopes is not None:
            meta["prelu_slopes"] = [float(v) for v in slopes]
        chunks_f32.append(blob_f32)
        chunks_f16.append(blob_f16)
        layer_specs.append(meta)
        byte_offset += len(blob_f32)
        byte_offset_f16 += len(blob_f16)

    binary_f32 = b"".join(chunks_f32)
    binary_f16 = b"".join(chunks_f16)
    manifest: Dict[str, Any] = {
        "format": "artcnn-webgpu-rgb",
        "version": FORMAT_VERSION,
        "name": model_name,
        "family": cfg.family,
        "architecture": cfg.arch_name(),
        "scale": cfg.scale,
        "in_channels": cfg.in_channels,
        "color_space": "rgb_srgb",
        "dtype": "float32",
        "filters": cfg.filters,
        "blocks": cfg.blocks,
        "receptive_radius": len(raw_layers),
        "activation": cfg.activation,
        "leaky_slope": float(cfg.leaky_slope),
        "input_skip": bool(cfg.input_skip),
        "weights_file": weights_filename or f"{model_name}.bin",
        "weights_bytes": len(binary_f32),
        "weights_sha256": hashlib.sha256(binary_f32).hexdigest(),
        "weights_f16_file": weights_f16_filename or f"{model_name}.f16.bin",
        "weights_f16_bytes": len(binary_f16),
        "weights_f16_sha256": hashlib.sha256(binary_f16).hexdigest(),
        "layers": layer_specs,
    }
    return manifest, binary_f32, binary_f16


def write_webgpu_bundle(
    checkpoint_path: Path,
    output_dir: Path,
    name: str | None = None,
) -> Tuple[Path, Path, Path, Dict[str, Any]]:
    """Export a checkpoint .npz to <name>.json, <name>.bin (FP32), and <name>.f16.bin (FP16)."""
    model_name = name or checkpoint_path.stem
    manifest, binary_f32, binary_f16 = export_npz_to_webgpu(
        checkpoint_path,
        name=model_name,
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
