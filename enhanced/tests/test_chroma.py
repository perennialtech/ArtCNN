# SPDX-FileCopyrightText: 2026 perennialtech contributors
# SPDX-License-Identifier: MIT

"""Tests for LumaGuidedChromaNet and mpv GLSL chroma shader export."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from artcnn.chroma import ChromaConfig, LumaGuidedChromaNet, export_chroma_glsl


def test_chroma_net_identity_init_and_roundtrip() -> None:
    torch.manual_seed(0)
    net = LumaGuidedChromaNet(ChromaConfig(filters=8, internal_layers=2))
    y = torch.rand(1, 1, 32, 32)
    c = torch.rand(1, 2, 16, 16)
    out = net(y, c)
    assert out.shape == (1, 2, 32, 32)
    # Zero-initialized final conv => exact bilinear identity at init
    bil = torch.nn.functional.interpolate(c, scale_factor=2, mode="bilinear", align_corners=False)
    assert (out - bil).abs().max().item() < 1e-6

    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "chroma.npz"
        net.save_npz(p)
        loaded = LumaGuidedChromaNet.load_npz(p)
        assert (loaded(y, c) - out).abs().max().item() < 1e-6

    glsl = export_chroma_glsl(net, label="C2F8_Chroma")
    assert "//!HOOK CHROMA" in glsl
    assert "//!BIND LUMA" in glsl
    assert "//!WIDTH LUMA.w" in glsl
    assert "//!COMPONENTS 2" in glsl
    assert "imageStore" in glsl


if __name__ == "__main__":
    test_chroma_net_identity_init_and_roundtrip()
    print("ALL CHROMA TESTS PASSED.")
