# SPDX-FileCopyrightText: 2026 perennialtech contributors
# SPDX-License-Identifier: MIT

"""Export ArtCNN Pareto checkpoints to browser-ready WebGPU bundles in Web/models/."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

from artcnn.webgpu_export import write_webgpu_bundle

DEFAULT_CATALOG: List[Tuple[str, str, str, str]] = [
    ("ArtCNN_C3F8_Fast", "C3F8 Fast", "Lowest GPU cost; ideal for real-time video on integrated GPUs", "ArtCNN_C3F8_Fast.npz"),
    ("ArtCNN_C4F16_Clean", "C4F16 Clean", "Preserves fine detail on clean animation sources", "ArtCNN_C4F16_Clean.npz"),
    ("ArtCNN_C4F16_Balanced", "C4F16 Balanced", "Balanced preset for mixed clean and compressed sources", "ArtCNN_C4F16_Balanced.npz"),
    ("ArtCNN_C4F16_Restore", "C4F16 Restore", "Restores blurred or compressed animation sources", "ArtCNN_C4F16_Restore.npz"),
    ("ArtCNN_C4F16_Grouped", "C4F16 Grouped", "Grouped 16-channel model for lower compute cost", "ArtCNN_C4F16_Grouped.npz"),
    ("ArtCNN_C4F32_Restore", "C4F32 Restore", "Wider 32-channel restoration model for high-end GPUs", "ArtCNN_C4F32_Restore.npz"),
]


def export_catalog(pareto_dir: Path, output_dir: Path) -> Path:
    entries = []
    for model_id, label, description, ckpt_name in DEFAULT_CATALOG:
        ckpt_path = pareto_dir / ckpt_name
        json_path, bin_path, manifest = write_webgpu_bundle(ckpt_path, output_dir, name=model_id)
        entries.append(
            {
                "id": model_id,
                "label": label,
                "description": description,
                "architecture": manifest["architecture"],
                "manifest": json_path.name,
                "weights": bin_path.name,
                "weights_bytes": manifest["weights_bytes"],
                "weights_sha256": manifest["weights_sha256"],
                "input_skip": manifest["input_skip"],
                "grouped": any(layer["kind"] == "conv_grouped" for layer in manifest["layers"]),
            }
        )
    catalog_path = output_dir / "catalog.json"
    catalog = {"format": "artcnn-webgpu-catalog", "version": 1, "models": entries}
    catalog_path.write_text(json.dumps(catalog, indent=2) + "\n", encoding="utf-8", newline="\n")
    return catalog_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pareto-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "pareto",
        help="Directory containing Pareto .npz checkpoints",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "Web" / "models",
        help="Destination directory for browser model bundles",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional single checkpoint .npz to export instead of the default catalog",
    )
    args = parser.parse_args()
    if args.checkpoint is not None:
        json_path, bin_path, manifest = write_webgpu_bundle(args.checkpoint, args.output_dir)
        print(f"Exported {manifest['name']} -> {json_path} ({manifest['weights_bytes']} bytes)")
    else:
        catalog_path = export_catalog(args.pareto_dir, args.output_dir)
        print(f"Exported browser catalog -> {catalog_path}")


if __name__ == "__main__":
    main()
