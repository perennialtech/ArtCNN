# SPDX-FileCopyrightText: 2026 perennialtech contributors
# SPDX-License-Identifier: MIT

"""Tests for the WebGPU / browser bundle exporter and catalog integrity."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

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
            manifest, binary = export_artcnn_webgpu(model, name=ckpt.stem, grouped_kernels=grouped)
            self.assertEqual(manifest["format"], "artcnn-webgpu")
            self.assertEqual(manifest["weights_bytes"], len(binary))
            self.assertEqual(manifest["weights_sha256"], hashlib.sha256(binary).hexdigest())
            self.assertEqual(len(binary) % 16, 0)
            offset = 0
            for layer in manifest["layers"]:
                self.assertEqual(layer["byte_offset"], offset)
                self.assertEqual(layer["byte_length"] % 16, 0)
                offset += layer["byte_length"]
            self.assertEqual(offset, len(binary))
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
                self.assertEqual(len(committed_bin), entry["weights_bytes"])
                self.assertEqual(hashlib.sha256(committed_bin).hexdigest(), entry["weights_sha256"])

                ckpt = PARETO_DIR / f"{entry['id']}.npz"
                _, fresh_bin_path, fresh_manifest = write_webgpu_bundle(ckpt, tmp_dir, name=entry["id"])
                self.assertEqual(committed_json, fresh_manifest)
                self.assertEqual(committed_bin, fresh_bin_path.read_bytes())


if __name__ == "__main__":
    unittest.main()
