# SPDX-FileCopyrightText: 2026 perennialtech contributors
# SPDX-License-Identifier: MIT

"""Tests for v4 perceptual optimization features:
  - SPAN local residuals + parameter-free attention + per-channel PReLU + dilation=2
  - FoldableECB(dilation=2) exact lossless weight folding
  - LDL artifact map loss & FP16 straight-through QAT
  - Catmull-Rom bicubic residual LumaGuidedChromaNet & GLSL export
"""
from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from artcnn.chroma import ChromaConfig, LumaGuidedChromaNet, export_chroma_glsl
from artcnn.models_np import ArtCNNCConfig
from artcnn.perceptual import FoldableECB, fp16_ste, ldl_artifact_loss
from train_pareto import Model


class V4PerceptualTests(unittest.TestCase):
    def test_foldable_ecb_dilated_exact_equivalence(self) -> None:
        torch.manual_seed(7)
        ecb = FoldableECB(16, 16, dilation=2)
        x = torch.randn(2, 16, 20, 20)
        with torch.no_grad():
            y_multi = ecb(x)
            w_eq, b_eq = ecb.fold_weights()
            y_folded = torch.nn.functional.conv2d(x, w_eq, b_eq, padding=2, dilation=2)
        self.assertLess((y_multi - y_folded).abs().max().item(), 1e-5)

    def test_span_prelu_dil2_export_and_reload_parity(self) -> None:
        torch.manual_seed(11)
        cfg = ArtCNNCConfig(
            filters=16,
            internal_layers=4,
            activation="prelu",
            input_skip=True,
            dilated_layers=(2,),
            local_residuals=True,
            span_attention=True,
        )
        model = Model(cfg, use_ecb=True)
        with torch.no_grad():
            model.convs[-1].weight.normal_(0.0, 0.05)
            model.prelus[0].fill_(0.15)
            model.prelus[1].fill_(0.08)
        x = torch.rand(1, 1, 24, 24)
        with tempfile.TemporaryDirectory() as td:
            model.export(Path(td), "span", (8, 8))
            glsl = (Path(td) / "span.glsl").read_text(encoding="utf-8")
            self.assertIn("-PReLU-Dil2-SPAN", glsl)
            self.assertIn("//!BIND conv2d", glsl)
            reloaded = Model(cfg, use_ecb=False)
            reloaded.load_npz(Path(td) / "span.npz")
            model.eval()
            reloaded.eval()
            with torch.no_grad():
                diff = (model(x) - reloaded(x)).abs().max().item()
            self.assertLess(diff, 1e-4)

    def test_ldl_artifact_loss_and_fp16_ste(self) -> None:
        torch.manual_seed(19)
        target = torch.rand(2, 1, 48, 48)
        ema_pred = (target + 0.01 * torch.randn_like(target)).clamp(0, 1)
        pred_noisy = (target + 0.05 * torch.randn_like(target)).clamp(0, 1).requires_grad_(True)
        pred_clean = ema_pred.clone().requires_grad_(True)
        loss_noisy = ldl_artifact_loss(pred_noisy, ema_pred, target)
        loss_clean = ldl_artifact_loss(pred_clean, ema_pred, target)
        self.assertGreater(loss_noisy.item(), loss_clean.item())
        loss_noisy.backward()
        self.assertIsNotNone(pred_noisy.grad)
        self.assertTrue(torch.isfinite(pred_noisy.grad).all())

        t = torch.tensor([0.12345678, -0.98765432], requires_grad=True)
        q = fp16_ste(t)
        q.sum().backward()
        self.assertTrue(torch.allclose(t.grad, torch.ones_like(t)))

    def test_catmull_rom_chroma_zero_init_matches_bicubic(self) -> None:
        net = LumaGuidedChromaNet(ChromaConfig(filters=12, internal_layers=2))
        y_hr = torch.rand(1, 1, 32, 32)
        c_lr = torch.rand(1, 2, 16, 16)
        with torch.no_grad():
            out = net(y_hr, c_lr)
            bic = torch.nn.functional.interpolate(c_lr, scale_factor=2, mode="bicubic", align_corners=False).clamp(0, 1)
        self.assertLess((out - bic).abs().max().item(), 1e-6)
        glsl = export_chroma_glsl(net)
        self.assertIn("w_neg", glsl)
        self.assertIn("w_pos", glsl)


if __name__ == "__main__":
    unittest.main()
