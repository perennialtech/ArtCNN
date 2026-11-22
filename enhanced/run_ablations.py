# SPDX-FileCopyrightText: 2026 perennialtech contributors
# SPDX-License-Identifier: MIT

"""Reproducible 3-way local ablation benchmark for ArtCNN C4F16:
  1. Baseline C4F16: pure L1, clean INTER_AREA downscale, ReLU, flat LR, 8-pass GLSL
  2. Zero-Cost C4F16: EXACT SAME baseline architecture/shader, upgraded training recipe
     (realistic multi-kernel + 4x4 DCT + halos, balanced sampler, L1 + Sobel + FFT +
     temporal-shift consistency + C4F32 teacher distillation + cosine LR + EMA)
  3. Enhanced C4F16: Zero-Cost training + LeakyReLU + direct bilinear input skip +
     fused 7-pass GLSL (eliminates separate depth_to_space pass & intermediate texture)
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

from artcnn.benchmark_data import BalancedPatchSampler, generate_dataset
from artcnn.degradation import DegradationConfig, degrade_hr_to_lr
from artcnn.glsl_export import (
    count_intermediate_textures,
    count_shader_passes,
    export_artcnn_glsl,
)
from artcnn.losses import (
    LossWeights,
    composite_loss_and_grad,
    shift_image_pixels,
    sobel_edge_loss_and_grad,
    temporal_shift_consistency_loss_and_grad,
)
from artcnn.models_np import AdamWOptimizer, ArtCNNC, ArtCNNCConfig


def psnr(pred: np.ndarray, target: np.ndarray) -> float:
    mse = float(np.mean((pred - target) ** 2))
    if mse < 1e-12:
        return 100.0
    return float(10.0 * np.log10(1.0 / mse))


def evaluate_model(
    model: ArtCNNC,
    test_imgs: List[np.ndarray],
    test_cats: List[str],
) -> Dict[str, float]:
    rng_clean = np.random.default_rng(1001)
    rng_real = np.random.default_rng(2002)

    clean_psnrs, clean_edges = [], []
    real_psnrs, real_edges = [], []
    flat_stds, shimmer_vals = [], []

    for img, cat in zip(test_imgs, test_cats):
        hr = img[None, None, :, :]

        lr_c = degrade_hr_to_lr(img, rng_clean, DegradationConfig(mode="clean_area"))[None, None, :, :]
        pred_c = model.forward(lr_c, clip=True)
        clean_psnrs.append(psnr(pred_c, hr))
        clean_edges.append(sobel_edge_loss_and_grad(pred_c, hr)[0])

        lr_r = degrade_hr_to_lr(
            img,
            rng_real,
            DegradationConfig(mode="realistic", clean_prob=0.0, dct_prob=1.0, ringing_prob=0.8, blur_prob=0.5),
        )[None, None, :, :]
        pred_r = model.forward(lr_r, clip=True)
        real_psnrs.append(psnr(pred_r, hr))
        real_edges.append(sobel_edge_loss_and_grad(pred_r, hr)[0])

        if cat in ("flat_fills", "gradients"):
            gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
            flat_mask = np.sqrt(gx * gx + gy * gy) < 0.02
            if np.any(flat_mask):
                err = (pred_r[0, 0] - img)[flat_mask]
                flat_stds.append(float(np.std(err)))

        lr_shift = shift_image_pixels(lr_r, 1, 1)
        pred_shift = model.forward(lr_shift, clip=True)
        shimmer, _, _ = temporal_shift_consistency_loss_and_grad(pred_r, pred_shift)
        shimmer_vals.append(shimmer)

    return {
        "clean_psnr_db": float(np.mean(clean_psnrs)),
        "clean_edge_mae": float(np.mean(clean_edges)),
        "realistic_psnr_db": float(np.mean(real_psnrs)),
        "realistic_edge_mae": float(np.mean(real_edges)),
        "flat_fill_err_std": float(np.mean(flat_stds)) if flat_stds else 0.0,
        "temporal_shimmer_mae": float(np.mean(shimmer_vals)),
    }


def train_variant(
    name: str,
    cfg: ArtCNNCConfig,
    deg_cfg: DegradationConfig,
    loss_w: LossWeights,
    train_imgs: List[np.ndarray],
    steps: int = 160,
    batch_size: int = 4,
    lr: float = 3e-3,
    use_cosine_ema: bool = False,
    use_balanced_sampler: bool = False,
    teacher: ArtCNNC | None = None,
    seed: int = 42,
) -> ArtCNNC:
    t0 = time.time()
    model = ArtCNNC(cfg, seed=seed)
    opt = AdamWOptimizer(model, lr=lr, weight_decay=1e-4, ema_decay=0.97)
    rng = np.random.default_rng(seed + 100)
    sampler = BalancedPatchSampler(train_imgs, seed=seed)

    for step in range(steps):
        if use_cosine_ema:
            opt.set_cosine_lr(step, steps, min_lr_ratio=0.05)

        if use_balanced_sampler:
            hr_list = sampler.sample_batch(batch_size)
        else:
            idxs = rng.integers(0, len(train_imgs), size=batch_size)
            hr_list = [train_imgs[int(i)] for i in idxs]

        lr_batch = np.stack([degrade_hr_to_lr(h, rng, deg_cfg) for h in hr_list], axis=0)[:, None, :, :]
        hr_batch = np.stack(hr_list, axis=0)[:, None, :, :]

        pred = model.forward(lr_batch, clip=False)
        t_pred = teacher.forward(lr_batch, clip=True) if (teacher is not None and loss_w.distill > 0) else None
        _, dout, _ = composite_loss_and_grad(pred, hr_batch, loss_w, teacher_pred=t_pred)

        if loss_w.temporal > 0:
            lr_s = shift_image_pixels(lr_batch, 1, 1)
            pred_s = model.forward(lr_s, clip=False)
            _, ga, gb = temporal_shift_consistency_loss_and_grad(pred, pred_s)
            model.backward(loss_w.temporal * gb)
            dW_s = [layer.dW.copy() for _, layer in model.layers()]
            db_s = [layer.db.copy() for _, layer in model.layers()]
            _ = model.forward(lr_batch, clip=False)
            model.backward(dout + loss_w.temporal * ga)
            for (_, layer), dw_add, db_add in zip(model.layers(), dW_s, db_s):
                layer.dW += dw_add
                layer.db += db_add
        else:
            model.backward(dout)

        opt.step(grad_clip=1.0)

    if use_cosine_ema:
        model.load_state_dict(opt.ema)

    print(f"Finished training [{name}] in {time.time() - t0:.1f}s (steps={steps})")
    return model


def main() -> None:
    train_imgs, _ = generate_dataset(n_per_category=8, size=48, seed_offset=0)
    test_imgs, test_cats = generate_dataset(n_per_category=4, size=48, seed_offset=500)

    steps = 160

    baseline = train_variant(
        name="1. Baseline C4F16 (L1 + Clean Area + ReLU)",
        cfg=ArtCNNCConfig(filters=16, activation="relu", input_skip=False),
        deg_cfg=DegradationConfig(mode="clean_area"),
        loss_w=LossWeights(l1=1.0),
        train_imgs=train_imgs,
        steps=steps,
        batch_size=4,
        lr=3.0e-3,
        use_cosine_ema=False,
        use_balanced_sampler=False,
        seed=42,
    )

    teacher = train_variant(
        name="Offline Teacher C4F32",
        cfg=ArtCNNCConfig(filters=32, activation="leaky_relu", input_skip=True),
        deg_cfg=DegradationConfig(mode="realistic"),
        loss_w=LossWeights(l1=1.0, sobel=0.25, fft=0.08),
        train_imgs=train_imgs,
        steps=70,
        batch_size=4,
        lr=3.0e-3,
        use_cosine_ema=True,
        use_balanced_sampler=True,
        seed=7,
    )

    zero_cost = train_variant(
        name="2. Zero-Cost C4F16 (Same Arch, Upgraded Training)",
        cfg=ArtCNNCConfig(filters=16, activation="relu", input_skip=False),
        deg_cfg=DegradationConfig(mode="realistic"),
        loss_w=LossWeights(l1=1.0, sobel=0.25, fft=0.08, distill=0.15, temporal=0.12),
        train_imgs=train_imgs,
        steps=steps,
        batch_size=4,
        lr=3.5e-3,
        use_cosine_ema=True,
        use_balanced_sampler=True,
        teacher=teacher,
        seed=42,
    )

    enhanced = train_variant(
        name="3. Enhanced C4F16 (Training + LeakyReLU + InputSkip + Fused GLSL)",
        cfg=ArtCNNCConfig(filters=16, activation="leaky_relu", input_skip=True),
        deg_cfg=DegradationConfig(mode="realistic"),
        loss_w=LossWeights(l1=1.0, sobel=0.25, fft=0.08, distill=0.15, temporal=0.12),
        train_imgs=train_imgs,
        steps=steps,
        batch_size=4,
        lr=3.5e-3,
        use_cosine_ema=True,
        use_balanced_sampler=True,
        teacher=teacher,
        seed=42,
    )

    m_base = evaluate_model(baseline, test_imgs, test_cats)
    m_zc = evaluate_model(zero_cost, test_imgs, test_cats)
    m_enh = evaluate_model(enhanced, test_imgs, test_cats)

    out_dir = Path("shaders")
    out_dir.mkdir(exist_ok=True)
    base_glsl = export_artcnn_glsl(baseline, fuse_depth_to_space=False)
    zc_glsl = export_artcnn_glsl(zero_cost, fuse_depth_to_space=True)
    enh_glsl = export_artcnn_glsl(enhanced, fuse_depth_to_space=True)
    (out_dir / "ArtCNN_C4F16_Baseline.glsl").write_text(base_glsl, encoding="utf-8", newline="\n")
    (out_dir / "ArtCNN_C4F16_ZeroCost_Fused.glsl").write_text(zc_glsl, encoding="utf-8", newline="\n")
    (out_dir / "ArtCNN_C4F16_Enhanced_Fused.glsl").write_text(enh_glsl, encoding="utf-8", newline="\n")

    summary = {
        "baseline": {
            **m_base,
            "params": baseline.param_count(),
            "macs_per_input_px": baseline.macs_per_input_pixel(),
            "glsl_passes": count_shader_passes(base_glsl),
            "glsl_intermediate_textures": count_intermediate_textures(base_glsl),
        },
        "zero_cost": {
            **m_zc,
            "params": zero_cost.param_count(),
            "macs_per_input_px": zero_cost.macs_per_input_pixel(),
            "glsl_passes": count_shader_passes(zc_glsl),
            "glsl_intermediate_textures": count_intermediate_textures(zc_glsl),
        },
        "enhanced": {
            **m_enh,
            "params": enhanced.param_count(),
            "macs_per_input_px": enhanced.macs_per_input_pixel(),
            "glsl_passes": count_shader_passes(enh_glsl),
            "glsl_intermediate_textures": count_intermediate_textures(enh_glsl),
        },
    }
    Path("ablation_results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md = [
        "# Local 3-Way Ablation Results (Procedural 6-Category Anime Benchmark)",
        "",
        f"Trained from scratch on CPU (pure NumPy, {steps} steps, 48x48 HR patches) to isolate",
        "zero-cost training improvements from low-cost architectural/shader upgrades.",
        "",
        "| Metric | Baseline C4F16 | Zero-Cost C4F16 | Enhanced C4F16 |",
        "| :--- | ---: | ---: | ---: |",
        f"| Clean Downscale PSNR (dB, higher better) | {m_base['clean_psnr_db']:.2f} | {m_zc['clean_psnr_db']:.2f} | {m_enh['clean_psnr_db']:.2f} |",
        f"| Clean Sobel Edge-MAE (lower better) | {m_base['clean_edge_mae']:.4f} | {m_zc['clean_edge_mae']:.4f} | {m_enh['clean_edge_mae']:.4f} |",
        f"| Realistic Compressed+Halo PSNR (dB, higher better) | {m_base['realistic_psnr_db']:.2f} | {m_zc['realistic_psnr_db']:.2f} | {m_enh['realistic_psnr_db']:.2f} |",
        f"| Realistic Compressed+Halo Edge-MAE (lower better) | {m_base['realistic_edge_mae']:.4f} | {m_zc['realistic_edge_mae']:.4f} | {m_enh['realistic_edge_mae']:.4f} |",
        f"| Flat-Fill Error Std (lower better) | {m_base['flat_fill_err_std']:.4f} | {m_zc['flat_fill_err_std']:.4f} | {m_enh['flat_fill_err_std']:.4f} |",
        f"| Temporal Shift Shimmer MAE (lower better) | {m_base['temporal_shimmer_mae']:.4f} | {m_zc['temporal_shimmer_mae']:.4f} | {m_enh['temporal_shimmer_mae']:.4f} |",
        f"| Conv MACs / input pixel | {baseline.macs_per_input_pixel():,} | {zero_cost.macs_per_input_pixel():,} | {enhanced.macs_per_input_pixel():,} |",
        f"| Exported GLSL Passes | {count_shader_passes(base_glsl)} | {count_shader_passes(zc_glsl)} | {count_shader_passes(enh_glsl)} |",
        f"| Intermediate RGBA Textures | {count_intermediate_textures(base_glsl)} | {count_intermediate_textures(zc_glsl)} | {count_intermediate_textures(enh_glsl)} |",
        "",
    ]
    Path("RESULTS.md").write_text("\n".join(md), encoding="utf-8")
    print("\n" + "\n".join(md))


if __name__ == "__main__":
    main()
