# SPDX-FileCopyrightText: 2026 perennialtech contributors
# SPDX-License-Identifier: MIT

"""Evaluate clean-still RGB models on held-out frames and compute the Pareto frontier."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from models import ArtCNNRGB
from perceptual import AlexNetLPIPS, DISTS, find_dists_weights, find_lpips_weights, ssim_map


def _load_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w, _ = rgb.shape
    return rgb[: h - (h % 2), : w - (w % 2)]


@torch.no_grad()
def eval_model(
    model: ArtCNNRGB | None,
    frames: List[np.ndarray],
    device: torch.device,
    lpips_fn: AlexNetLPIPS | None,
    dists_fn: DISTS | None,
    classical_interp: int | None = None,
) -> Dict[str, float]:
    kernels = [("area", cv2.INTER_AREA), ("bicubic", cv2.INTER_CUBIC), ("lanczos4", cv2.INTER_LANCZOS4)]
    per_k: Dict[str, Dict[str, List[float]]] = {k: {"psnr": [], "ssim": [], "lpips": [], "dists": []} for k, _ in kernels}

    for hr in frames:
        h, w, _ = hr.shape
        hr_t = torch.from_numpy(hr.transpose(2, 0, 1).astype(np.float32) / 255.0).unsqueeze(0).to(device)
        for k_name, interp in kernels:
            lr = cv2.resize(hr, (w // 2, h // 2), interpolation=interp)
            if model is not None:
                lr_t = torch.from_numpy(lr.transpose(2, 0, 1).astype(np.float32) / 255.0).unsqueeze(0).to(device)
                pred = model(lr_t).clamp(0.0, 1.0)
            else:
                up = cv2.resize(lr, (w, h), interpolation=classical_interp or cv2.INTER_LINEAR)
                pred = torch.from_numpy(up.transpose(2, 0, 1).astype(np.float32) / 255.0).unsqueeze(0).to(device)

            mse = float(F.mse_loss(pred, hr_t).item())
            per_k[k_name]["psnr"].append(10.0 * math.log10(1.0 / max(mse, 1e-12)))
            per_k[k_name]["ssim"].append(float(ssim_map(pred, hr_t)[0].mean().item()))
            if lpips_fn is not None:
                per_k[k_name]["lpips"].append(float(lpips_fn(pred, hr_t).mean().item()))
            if dists_fn is not None:
                per_k[k_name]["dists"].append(float(dists_fn(pred, hr_t).mean().item()))

    out: Dict[str, float] = {}
    for k_name in per_k:
        out[f"{k_name}_psnr"] = float(np.mean(per_k[k_name]["psnr"]))
        out[f"{k_name}_ssim"] = float(np.mean(per_k[k_name]["ssim"]))
        if per_k[k_name]["lpips"]:
            out[f"{k_name}_lpips"] = float(np.mean(per_k[k_name]["lpips"]))
        if per_k[k_name]["dists"]:
            out[f"{k_name}_dists"] = float(np.mean(per_k[k_name]["dists"]))
    out["mean_psnr"] = float(np.mean([out[f"{k}_psnr"] for k, _ in kernels]))
    out["mean_ssim"] = float(np.mean([out[f"{k}_ssim"] for k, _ in kernels]))
    if lpips_fn is not None:
        out["mean_lpips"] = float(np.mean([out[f"{k}_lpips"] for k, _ in kernels]))
    if dists_fn is not None:
        out["mean_dists"] = float(np.mean([out[f"{k}_dists"] for k, _ in kernels]))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt-dir", type=Path, required=True)
    parser.add_argument("--holdout-manifest", type=Path, required=True)
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    entries = json.loads(args.holdout_manifest.read_text(encoding="utf-8"))
    frames = [_load_rgb(args.frames_dir / e["file"]) for e in entries if (args.frames_dir / e["file"]).exists()]
    print(f"Evaluating on {len(frames)} held-out clean RGB frames...")

    lpips_w = find_lpips_weights()
    lpips_fn = AlexNetLPIPS(*lpips_w).to(device).eval() if lpips_w else None
    dists_w = find_dists_weights()
    dists_fn = DISTS(*dists_w).to(device).eval() if dists_w else None

    results = []
    for name, interp in [("Bilinear", cv2.INTER_LINEAR), ("Bicubic", cv2.INTER_CUBIC), ("Lanczos4", cv2.INTER_LANCZOS4)]:
        m = eval_model(None, frames, device, lpips_fn, dists_fn, classical_interp=interp)
        m["id"] = name
        m["params"] = 0
        m["macs_per_px"] = 0
        results.append(m)

    for ckpt in sorted(args.ckpt_dir.glob("*.npz")):
        model = ArtCNNRGB.from_npz(ckpt).to(device).eval()
        m = eval_model(model, frames, device, lpips_fn, dists_fn)
        params = sum(p.numel() for p in model.parameters())
        # MACs per input pixel = sum(c_in * c_out * 9) across all Conv2d layers
        macs = sum(mod.in_channels * mod.out_channels * 9 for mod in model.modules() if isinstance(mod, torch.nn.Conv2d))
        m["id"] = ckpt.stem
        m["arch"] = model.config.arch_name()
        m["params"] = int(params)
        m["macs_per_px"] = int(macs)
        results.append(m)
        print(f"  {ckpt.stem}: params={params:,} macs/px={macs:,} PSNR={m['mean_psnr']:.3f} SSIM={m['mean_ssim']:.5f}")

    # Mark Pareto optimality on (macs_per_px, mean_psnr) among neural models
    neural = [r for r in results if r["macs_per_px"] > 0]
    for r in neural:
        dominated = any(
            (other["macs_per_px"] <= r["macs_per_px"] and other["mean_psnr"] >= r["mean_psnr"])
            and (other["macs_per_px"] < r["macs_per_px"] or other["mean_psnr"] > r["mean_psnr"])
            for other in neural
        )
        r["pareto_optimal"] = not dominated

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"frames": len(frames), "models": results}, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
