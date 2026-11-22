"""Comprehensive verification suite for ArtCNN Enhanced."""

from __future__ import annotations

import numpy as np

from artcnn.benchmark_data import BalancedPatchSampler, generate_dataset
from artcnn.degradation import (
    DegradationConfig,
    apply_dct4x4_compression,
    degrade_hr_to_lr,
)
from artcnn.glsl_export import (
    count_intermediate_textures,
    count_shader_passes,
    export_artcnn_glsl,
)
from artcnn.linear_collapse import verify_collapse_interior_vs_border
from artcnn.losses import (
    charbonnier_loss_and_grad,
    fft_magnitude_loss_and_grad,
    sobel_edge_loss_and_grad,
    temporal_shift_consistency_loss_and_grad,
)
from artcnn.models_np import ArtCNNC, ArtCNNCConfig
from artcnn.reparam import FoldableEdgeConvBlock


def _numerical_grad(fn, x: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    grad = np.zeros_like(x, dtype=np.float64)
    it = np.nditer(x, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        orig = x[idx]
        x[idx] = orig + eps
        lp = fn(x)
        x[idx] = orig - eps
        lm = fn(x)
        x[idx] = orig
        grad[idx] = (lp - lm) / (2.0 * eps)
        it.iternext()
    return grad


def test_architecture_counts():
    m16 = ArtCNNC(ArtCNNCConfig(filters=16))
    m32 = ArtCNNC(ArtCNNCConfig(filters=32))
    assert m16.macs_per_input_pixel() == 12240, m16.macs_per_input_pixel()
    assert m32.macs_per_input_pixel() == 47520, m32.macs_per_input_pixel()
    assert 12000 < m16.param_count() < 12500, m16.param_count()
    assert 47000 < m32.param_count() < 48500, m32.param_count()
    print(
        f"[PASS] Architecture counts: C4F16={m16.param_count()} params / {m16.macs_per_input_pixel()} MACs, "
        f"C4F32={m32.param_count()} params / {m32.macs_per_input_pixel()} MACs"
    )


def test_model_backprop_gradients():
    rng = np.random.default_rng(123)
    for act in ("relu", "leaky_relu"):
        for skip in (False, True):
            model = ArtCNNC(
                ArtCNNCConfig(filters=4, internal_layers=2, activation=act, input_skip=skip),
                seed=7,
            )
            x = rng.uniform(0.1, 0.9, size=(1, 1, 6, 6)).astype(np.float32)
            target = rng.uniform(0.1, 0.9, size=(1, 1, 12, 12)).astype(np.float32)

            pred = model.forward(x)
            _, dout = charbonnier_loss_and_grad(pred, target)
            model.backward(dout)

            for lname, layer in [("conv0", model.conv0), ("internal0", model.internal[0]), ("features", model.features)]:
                W_orig = layer.W.copy()
                analytic = layer.dW[0, 0, 1, 1]
                eps = 1e-3
                layer.W[0, 0, 1, 1] = W_orig[0, 0, 1, 1] + eps
                lp, _ = charbonnier_loss_and_grad(model.forward(x), target)
                layer.W[0, 0, 1, 1] = W_orig[0, 0, 1, 1] - eps
                lm, _ = charbonnier_loss_and_grad(model.forward(x), target)
                layer.W = W_orig
                num = (lp - lm) / (2.0 * eps)
                err = abs(analytic - num)
                assert err < 2e-3, f"{act} skip={skip} {lname}: analytic={analytic}, num={num}, err={err}"
    print("[PASS] Model backprop gradient checks (ReLU, LeakyReLU, input_skip)")


def test_loss_gradients():
    rng = np.random.default_rng(99)
    pred = rng.uniform(0.1, 0.9, size=(1, 1, 8, 8)).astype(np.float32)
    target = rng.uniform(0.1, 0.9, size=(1, 1, 8, 8)).astype(np.float32)

    for name, fn in [
        ("charbonnier", charbonnier_loss_and_grad),
        ("sobel_edge", sobel_edge_loss_and_grad),
        ("fft_magnitude", fft_magnitude_loss_and_grad),
    ]:
        _, g_ana = fn(pred, target)
        g_num = _numerical_grad(lambda p: fn(p.astype(np.float32), target)[0], pred.astype(np.float64), eps=1e-3)
        max_err = float(np.max(np.abs(g_ana - g_num)))
        assert max_err < 1e-3, f"{name} grad mismatch: max_err={max_err}"

    pred_b = rng.uniform(0.1, 0.9, size=(1, 1, 8, 8)).astype(np.float32)
    _, ga, gb = temporal_shift_consistency_loss_and_grad(pred, pred_b)
    ga_num = _numerical_grad(
        lambda p: temporal_shift_consistency_loss_and_grad(p.astype(np.float32), pred_b)[0],
        pred.astype(np.float64),
        eps=1e-3,
    )
    assert float(np.max(np.abs(ga - ga_num))) < 1e-3
    print("[PASS] Loss analytic gradient checks (Charbonnier, Sobel, FFT, Temporal)")


def test_reparam_exact_folding():
    rng = np.random.default_rng(77)
    for c_in, c_out in [(16, 16), (32, 32), (8, 16)]:
        blk = FoldableEdgeConvBlock(c_in, c_out, seed=11)
        x = rng.uniform(-1.0, 1.0, size=(2, c_in, 12, 12)).astype(np.float32)
        y_multi = blk.forward_multibranch(x)
        conv_folded = blk.to_conv2d3x3()
        y_folded = conv_folded.forward(x)
        max_diff = float(np.max(np.abs(y_multi - y_folded)))
        assert max_diff < 1e-5, f"Folding mismatch ({c_in}->{c_out}): {max_diff}"
    print("[PASS] FoldableEdgeConvBlock 5-branch -> 1x Conv3x3 exact equivalence (max_diff < 1e-5)")


def test_degradation_and_sampler():
    imgs, cats = generate_dataset(n_per_category=4, size=64, seed_offset=0)
    sampler = BalancedPatchSampler(imgs, seed=0)
    assert len(sampler.active_buckets) >= 2
    batch = sampler.sample_batch(6)
    assert len(batch) == 6 and batch[0].shape == (64, 64)

    rng = np.random.default_rng(42)
    dct_out = apply_dct4x4_compression(imgs[0], strength=0.02)
    assert dct_out.shape == imgs[0].shape and 0.0 <= dct_out.min() and dct_out.max() <= 1.0

    lr = degrade_hr_to_lr(imgs[0], rng, DegradationConfig(mode="realistic"))
    assert lr.shape == (32, 32)
    print("[PASS] Realistic degradation (4x4 DCT + multi-kernel + halos) & balanced patch sampler")


def test_glsl_export_and_linear_collapse():
    base_model = ArtCNNC(ArtCNNCConfig(filters=16, activation="relu", input_skip=False))
    base_glsl = export_artcnn_glsl(base_model, fuse_depth_to_space=False)
    assert count_shader_passes(base_glsl) == 8
    assert count_intermediate_textures(base_glsl) == 7
    shuffle_pass = base_glsl.split("(Depth-To-Space)\n", 1)[1]
    assert shuffle_pass.splitlines()[0] == "//!COMPUTE 12 16 12 16"

    enh_model = ArtCNNC(ArtCNNCConfig(filters=16, activation="leaky_relu", input_skip=True))
    enh_glsl = export_artcnn_glsl(enh_model, fuse_depth_to_space=True)
    assert count_shader_passes(enh_glsl) == 7
    assert count_intermediate_textures(enh_glsl) == 6
    assert "max(result0, result0 * F(0.1))" in enh_glsl
    assert "LUMA_tex(uv_0_0)" in enh_glsl
    assert "Depth-To-Space" not in enh_glsl

    # Verify consecutive 3x3 -> 5x5 collapse interior precision vs border divergence
    # using non-zero trained-like weights (conv0 -> internal[0] or non-zero mix_global -> features)
    rng = np.random.default_rng(55)
    W1 = (rng.standard_normal((16, 16, 3, 3)) * 0.1).astype(np.float32)
    b1 = (rng.standard_normal((16,)) * 0.05).astype(np.float32)
    W2 = (rng.standard_normal((4, 16, 3, 3)) * 0.1).astype(np.float32)
    b2 = (rng.standard_normal((4,)) * 0.05).astype(np.float32)
    interior_diff, border_diff = verify_collapse_interior_vs_border(W1, b1, W2, b2)
    assert interior_diff < 1e-12, f"Interior diff too large: {interior_diff}"
    assert border_diff > 1e-3, f"Expected non-zero border diff, got: {border_diff}"
    print(
        f"[PASS] GLSL export (baseline: 8 passes/7 tex -> fused: 7 passes/6 tex) & "
        f"3x3->5x5 collapse (interior_diff={interior_diff:.2e}, border_diff={border_diff:.2e})"
    )


if __name__ == "__main__":
    test_architecture_counts()
    test_model_backprop_gradients()
    test_loss_gradients()
    test_reparam_exact_folding()
    test_degradation_and_sampler()
    test_glsl_export_and_linear_collapse()
    print("\nALL TESTS PASSED.")
