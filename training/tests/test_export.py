# SPDX-FileCopyrightText: 2026 perennialtech contributors
# SPDX-License-Identifier: MIT

"""Tests for clean-still RGB WebGPU bundle export and catalog integrity."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from webgpu_export import export_npz_to_webgpu, write_webgpu_bundle

REPO_ROOT = Path(__file__).resolve().parents[2]
CKPT_DIR = REPO_ROOT / "training" / "checkpoints"
WEB_MODELS_DIR = REPO_ROOT / "Web" / "models"


class WebGPUExportTests(unittest.TestCase):
    def test_committed_catalog_matches_checkpoints(self) -> None:
        catalog = json.loads((WEB_MODELS_DIR / "catalog.json").read_text(encoding="utf-8"))
        self.assertEqual(catalog["format"], "artcnn-webgpu-rgb-catalog")
        self.assertGreaterEqual(len(catalog["models"]), 3)

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
                self.assertEqual(len(committed_f16) * 2, len(committed_bin))
                self.assertEqual(len(committed_bin) % 16, 0)
                self.assertEqual(len(committed_f16) % 8, 0)

                ckpt = CKPT_DIR / f"{entry['id']}.npz"
                self.assertTrue(ckpt.exists(), f"Missing checkpoint {ckpt}")
                _, fresh_bin, fresh_f16, fresh_manifest = write_webgpu_bundle(ckpt, tmp_dir, name=entry["id"])
                self.assertEqual(committed_json, fresh_manifest)
                self.assertEqual(committed_bin, fresh_bin.read_bytes())
                self.assertEqual(committed_f16, fresh_f16.read_bytes())


if __name__ == "__main__":
    unittest.main()
