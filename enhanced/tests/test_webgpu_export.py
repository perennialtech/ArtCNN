# SPDX-FileCopyrightText: 2026 perennialtech contributors
# SPDX-License-Identifier: MIT

"""Tests for the WebGPU / browser bundle exporter (FP32 + FP16) and catalog integrity."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from artcnn.webgpu_export import export_artcnn_webgpu, load_checkpoint, write_webgpu_bundle

REPO_ROOT = Path(__file__).resolve().parents[2]
PARETO_DIR = REPO_ROOT / "enhanced" / "pareto"
WEB_MODELS_DIR = REPO_ROOT / "Web" / "models"


class WebGPUExportTests(unittest.TestCase):
    def test_all_pareto_checkpoints_export_cleanly(self) -> None:
        ckpts = sorted(PARETO_DIR.glob("*.npz"))
        self.assertGreaterEqual(len(ckpts), 6)
        for ckpt in ckpts:
            model, grouped = load_checkpoint(ckpt)
            manifest, binary_f32, binary_f16 = export_artcnn_webgpu(
                model, name=ckpt.stem, grouped_kernels=grouped
            )
            self.assertEqual(manifest["format"], "artcnn-webgpu")
            self.assertEqual(manifest["weights_bytes"], len(binary_f32))
            self.assertEqual(manifest["weights_sha256"], hashlib.sha256(binary_f32).hexdigest())
            self.assertEqual(manifest["weights_f16_bytes"], len(binary_f16))
            self.assertEqual(manifest["weights_f16_sha256"], hashlib.sha256(binary_f16).hexdigest())
            self.assertEqual(len(binary_f16) * 2, len(binary_f32))
            self.assertEqual(len(binary_f32) % 16, 0)
            self.assertEqual(len(binary_f16) % 8, 0)
            offset_f32 = 0
            offset_f16 = 0
            for layer in manifest["layers"]:
                self.assertEqual(layer["byte_offset"], offset_f32)
                self.assertEqual(layer["byte_length"] % 16, 0)
                self.assertEqual(layer["byte_offset_f16"], offset_f16)
                self.assertEqual(layer["byte_length_f16"] * 2, layer["byte_length"])
                self.assertEqual(layer["byte_length_f16"] % 8, 0)
                offset_f32 += layer["byte_length"]
                offset_f16 += layer["byte_length_f16"]
            self.assertEqual(offset_f32, len(binary_f32))
            self.assertEqual(offset_f16, len(binary_f16))
            if grouped:
                self.assertTrue(any(l["kind"] == "conv_grouped" for l in manifest["layers"]))

    def test_committed_web_catalog_matches_checkpoints(self) -> None:
        catalog = json.loads((WEB_MODELS_DIR / "catalog.json").read_text(encoding="utf-8"))
        self.assertEqual(catalog["format"], "artcnn-webgpu-catalog")
        self.assertGreaterEqual(len(catalog["models"]), 6)
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            for entry in catalog["models"]:
                committed_json = json.loads((WEB_MODELS_DIR / entry["manifest"]).read_text(encoding="utf-8"))
                committed_bin = (WEB_MODELS_DIR / entry["weights"]).read_bytes()
                committed_f16 = (WEB_MODELS_DIR / entry["weights_f16"]).read_bytes()
                self.assertEqual(len(committed_bin), entry["weights_bytes"])
                self.assertEqual(hashlib.sha256(committed_bin).hexdigest(), entry["weights_sha256"])
                self.assertEqual(len(committed_f16), entry["weights_f16_bytes"])
                self.assertEqual(hashlib.sha256(committed_f16).hexdigest(), entry["weights_f16_sha256"])

                ckpt = PARETO_DIR / f"{entry['id']}.npz"
                _, fresh_bin, fresh_f16, fresh_manifest = write_webgpu_bundle(ckpt, tmp_dir, name=entry["id"])
                self.assertEqual(committed_json, fresh_manifest)
                self.assertEqual(committed_bin, fresh_bin.read_bytes())
                self.assertEqual(committed_f16, fresh_f16.read_bytes())


if __name__ == "__main__":
    unittest.main()
