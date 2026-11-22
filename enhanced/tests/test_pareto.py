"""Check the published shader catalog, fine-tuning, and grouped shader export."""
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import onnx
from onnx.reference import ReferenceEvaluator
import torch

from artcnn.models_np import ArtCNNC, ArtCNNCConfig
from experiments.tune_workgroups import COMPUTE, retile_shader
from train_pareto import Model, load_teacher


def test_release_manifest() -> None:
    root = Path(__file__).resolve().parents[2]
    artifacts = root / "enhanced" / "pareto"
    catalog = root / "GLSL" / "Optimized"
    models = json.loads((artifacts / "models.json").read_text())["models"]
    shaders = []
    for name, model in models.items():
        shader = (artifacts / model["file"]).resolve()
        assert shader.parent == catalog, f"{name} points outside the shader catalog"
        assert hashlib.sha256(shader.read_bytes()).hexdigest() == model["sha256"], name
        if model["checkpoint"]:
            assert (artifacts / model["checkpoint"]).is_file(), name
        shaders.append(shader)
    assert sorted(shaders) == sorted(catalog.glob("*.glsl"))


def test_upstream_import(onnx_file: Path) -> None:
    model = Model(ArtCNNCConfig(filters=16)).eval()
    model.load_onnx(onnx_file)
    pixels = np.random.default_rng(31).random((2, 1, 17, 19), dtype=np.float32)
    upstream = ReferenceEvaluator(onnx.load(onnx_file)).run(None, {"input": pixels})[0]
    with torch.no_grad():
        result = model(torch.from_numpy(pixels)).clamp(0, 1).numpy()
    np.testing.assert_allclose(result, upstream, atol=3e-5, rtol=0)


def test_grouped_export() -> None:
    torch.manual_seed(95)
    config = ArtCNNCConfig(filters=12, internal_layers=3, input_skip=True)
    model = Model(config, grouped=True).eval()
    torch.nn.init.normal_(model.convs[-1].weight, std=0.08)
    rng = np.random.default_rng(66)
    pixels = rng.random((2, 1, 19, 17), dtype=np.float32)
    with TemporaryDirectory() as directory:
        path = Path(directory)
        model.export(path, "model", (8, 8))
        dense = ArtCNNC(config)
        with np.load(path / "model.npz") as weights:
            dense.load_state_dict(weights)
        reloaded = Model(config, grouped=True).eval()
        reloaded.load_npz(path / "model.npz")
        shader = (path / "model.glsl").read_text()
        assert "C3F12" in shader
        assert shader.count("V4 group0 =") == 4
        assert len(COMPUTE.findall(shader)) == 6
        assert ("16", "16", "8", "8") == COMPUTE.findall(shader)[-1]
        assert ("32", "8", "16", "4") == COMPUTE.findall(retile_shader(shader, 16, 4))[-1]
        with torch.no_grad():
            predicted = model(torch.from_numpy(pixels)).numpy()
            restored = reloaded(torch.from_numpy(pixels)).numpy()
        np.testing.assert_array_equal(predicted, restored)
        np.testing.assert_allclose(dense.forward(pixels), predicted, atol=3e-6, rtol=0)


def test_residual_teacher(onnx_file: Path) -> None:
    model = load_teacher(onnx_file).eval()
    pixels = np.random.default_rng(83).random((1, 1, 9, 11), dtype=np.float32)
    expected = ReferenceEvaluator(onnx.load(onnx_file)).run(None, {"input": pixels})[0]
    with torch.no_grad():
        actual = model(torch.from_numpy(pixels)).clamp(0, 1).numpy()
    np.testing.assert_allclose(actual, expected, atol=5e-5, rtol=0)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, default=Path("../ONNX/ArtCNN_C4F16.onnx"))
    parser.add_argument("--teacher-onnx", type=Path, default=Path("../ONNX/ArtCNN_R8F64.onnx"))
    args = parser.parse_args()
    test_release_manifest()
    test_upstream_import(args.onnx)
    test_grouped_export()
    test_residual_teacher(args.teacher_onnx)
    print("Release catalog, upstream import, grouped checkpoint, and workgroup checks passed.")
