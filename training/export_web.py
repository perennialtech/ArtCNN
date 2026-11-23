# SPDX-FileCopyrightText: 2026 perennialtech contributors
# SPDX-License-Identifier: MIT

"""Export clean-still RGB checkpoints to browser WebGPU bundles in Web/models/."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

from webgpu_export import write_webgpu_bundle

DEFAULT_CATALOG: List[Tuple[str, str, str, str]] = [
    ("ArtCNN_C3F8_Clean", "C3F8 Clean", "Ultra-lightweight 2x clean RGB upscaler for low-power GPUs and CPU fallback", "ArtCNN_C3F8_Clean.npz"),
    ("ArtCNN_C4F16_Clean", "C4F16 Clean", "Fast 2x clean RGB upscaler for crisp lineart and everyday stills", "ArtCNN_C4F16_Clean.npz"),
    ("ArtCNN_C4F32_Clean", "C4F32 Clean", "Balanced 32-channel 2x clean RGB upscaler with sharper fine detail", "ArtCNN_C4F32_Clean.npz"),
    ("ArtCNN_R6F48_Clean", "R6F48 Clean", "Maximum-quality 2x clean RGB residual upscaler for still illustrations", "ArtCNN_R6F48_Clean.npz"),
]


def export_catalog(ckpt_dir: Path, output_dir: Path, pareto_json: Path | None = None) -> Path:
    metrics_map = {}
    if pareto_json and pareto_json.exists():
        data = json.loads(pareto_json.read_text(encoding="utf-8"))
        for row in data.get("models", []):
            metrics_map[row["id"]] = row

    entries = []
    for model_id, label, description, ckpt_name in DEFAULT_CATALOG:
        ckpt_path = ckpt_dir / ckpt_name
        if not ckpt_path.exists():
            continue
        json_path, bin_path, f16_path, manifest = write_webgpu_bundle(ckpt_path, output_dir, name=model_id)
        entry = {
            "id": model_id,
            "label": label,
            "description": description,
            "family": manifest["family"],
            "architecture": manifest["architecture"],
            "receptive_radius": manifest["receptive_radius"],
            "manifest": json_path.name,
            "weights": bin_path.name,
            "weights_bytes": manifest["weights_bytes"],
            "weights_sha256": manifest["weights_sha256"],
            "weights_f16": f16_path.name,
            "weights_f16_bytes": manifest["weights_f16_bytes"],
            "weights_f16_sha256": manifest["weights_f16_sha256"],
        }
        if model_id in metrics_map:
            m = metrics_map[model_id]
            entry["pareto"] = {
                "psnr": round(m["mean_psnr"], 3),
                "ssim": round(m["mean_ssim"], 5),
                "lpips": round(m["mean_lpips"], 5) if "mean_lpips" in m else None,
                "dists": round(m["mean_dists"], 5) if "mean_dists" in m else None,
                "pareto_optimal": bool(m.get("pareto_optimal", True)),
            }
        entries.append(entry)

    catalog_path = output_dir / "catalog.json"
    catalog = {"format": "artcnn-webgpu-rgb-catalog", "version": 2, "models": entries}
    catalog_path.write_text(json.dumps(catalog, indent=2) + "\n", encoding="utf-8", newline="\n")
    return catalog_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt-dir", type=Path, default=Path(__file__).resolve().parent / "checkpoints")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent.parent / "Web" / "models")
    parser.add_argument("--pareto-json", type=Path, default=Path(__file__).resolve().parent / "pareto_results.json")
    args = parser.parse_args()
    catalog_path = export_catalog(args.ckpt_dir, args.output_dir, pareto_json=args.pareto_json)
    print(f"Exported browser catalog -> {catalog_path}")


if __name__ == "__main__":
    main()
