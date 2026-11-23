# SPDX-FileCopyrightText: 2026 perennialtech contributors
# SPDX-License-Identifier: MIT

"""Train a lightweight luma-guided 4:2:0 -> 4:4:4 chroma upscaler on real anime frames."""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from artcnn.chroma import ChromaConfig, LumaGuidedChromaNet, export_chroma_glsl
from artcnn.perceptual import AlexNetLPIPS, DISTS, find_dists_weights, find_lpips_weights, sobel_edge_loss

# BT.709 RGB [0,1] <-> YCbCr [0,1] (Cb, Cr centered at 0.5)
_KR, _KG, _KB = 0.2126, 0.7152, 0.0722


def rgb_to_ycbcr(rgb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
    y = _KR * r + _KG * g + _KB * b
    cb = (b - y) / (2.0 * (1.0 - _KB)) + 0.5
    cr = (r - y) / (2.0 * (1.0 - _KR)) + 0.5
    return y.clamp(0.0, 1.0), torch.cat([cb, cr], dim=1).clamp(0.0, 1.0)


def ycbcr_to_rgb(y: torch.Tensor, cbcr: torch.Tensor) -> torch.Tensor:
    cb = cbcr[:, 0:1] - 0.5
    cr = cbcr[:, 1:2] - 0.5
    r = y + 2.0 * (1.0 - _KR) * cr
    b = y + 2.0 * (1.0 - _KB) * cb
    g = (y - _KR * r - _KB * b) / _KG
    return torch.cat([r, g, b], dim=1).clamp(0.0, 1.0)


class RgbFramesDataset(Dataset):
    def __init__(self, manifest_path: Path, frames_dir: Path, patch: int, count: int, seed: int):
        records = json.loads(manifest_path.read_text())
        self.paths = [frames_dir / r["file"] for r in records if (frames_dir / r["file"]).is_file()]
        if not self.paths:
            raise ValueError(f"No frames found in {frames_dir}")
        self.patch = patch
        self.count = count
        self.seed = seed

    def __len__(self):
        return self.count

    def __getitem__(self, idx):
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, idx]))
        p = self.paths[int(rng.integers(len(self.paths)))]
        with Image.open(p) as im:
            rgb = np.asarray(im.convert("RGB"), dtype=np.uint8)
        h, w, _ = rgb.shape
        y = int(rng.integers((h - self.patch) // 2 + 1)) * 2
        x = int(rng.integers((w - self.patch) // 2 + 1)) * 2
        crop = rgb[y:y + self.patch, x:x + self.patch]
        angle = int(rng.integers(4))
        crop = np.rot90(crop, angle)
        if rng.random() < 0.5:
            crop = crop[:, ::-1]
        return np.ascontiguousarray(crop.transpose(2, 0, 1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--holdout-manifest", type=Path, required=True)
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--filters", type=int, default=12)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--patch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--seed", type=int, default=42001)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda")
    cfg = ChromaConfig(filters=args.filters, internal_layers=args.layers)
    net = LumaGuidedChromaNet(cfg).to(device)
    ema = copy.deepcopy(net).eval().requires_grad_(False)
    lpips_fn = AlexNetLPIPS(*find_lpips_weights()).to(device)
    dists_fn = DISTS(*find_dists_weights()).to(device)

    ds = RgbFramesDataset(args.train_manifest, args.frames_dir, args.patch, args.steps * args.batch, args.seed)
    dl = DataLoader(ds, batch_size=args.batch, num_workers=4, pin_memory=True, drop_last=True)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, fused=True)
    scaler = torch.amp.GradScaler("cuda")
    rng = np.random.default_rng(args.seed)
    t0 = time.monotonic()

    for step, batch in enumerate(dl, start=1):
        rgb = batch.to(device, non_blocking=True).float() / 255.0
        y_hr, c_hr = rgb_to_ycbcr(rgb)
        # Simulate 4:2:0 chroma subsampling (mix area + bicubic + mild quantization)
        if rng.random() < 0.5:
            c_lr = F.avg_pool2d(c_hr, 2)
        else:
            c_lr = F.interpolate(c_hr, scale_factor=0.5, mode="bicubic", align_corners=False, antialias=True).clamp(0, 1)
        c_lr = (c_lr * 224.0).round() / 224.0
        lr_rate = args.lr * (0.08 + 0.92 * (1 + math.cos(math.pi * step / args.steps)) / 2)
        opt.param_groups[0]["lr"] = lr_rate
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            pred_c = net(y_hr, c_lr)
        pred_c_f = pred_c.float().clamp(0.0, 1.0)[:, :, 8:-8, 8:-8]
        tgt_c = c_hr[:, :, 8:-8, 8:-8]
        y_in = y_hr[:, :, 8:-8, 8:-8]
        pred_rgb = ycbcr_to_rgb(y_in, pred_c_f)
        tgt_rgb = rgb[:, :, 8:-8, 8:-8]
        diff = pred_c_f - tgt_c
        loss = (
            torch.sqrt(diff.square() + 1e-6).mean()
            + 0.5 * diff.square().mean()
            + 0.15 * sobel_edge_loss(pred_c_f[:, 0:1], tgt_c[:, 0:1])
            + 0.15 * sobel_edge_loss(pred_c_f[:, 1:2], tgt_c[:, 1:2])
            + 0.04 * lpips_fn(pred_rgb, tgt_rgb)
            + 0.04 * dists_fn(pred_rgb, tgt_rgb)
        )
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        with torch.no_grad():
            torch._foreach_lerp_(tuple(ema.parameters()), tuple(net.parameters()), max(0.01, 2 / (step + 2)))
        if step % 1000 == 0 or step == args.steps:
            print(f"Chroma step {step}/{args.steps} {time.monotonic()-t0:.1f}s loss={loss.item():.5f}", flush=True)

    # Evaluate on holdout 96 frames vs Bilinear and Bicubic chroma
    net = ema
    net.eval()
    h_records = json.loads(args.holdout_manifest.read_text())
    metrics = {k: {"c_psnr": [], "rgb_psnr": [], "rgb_lpips": [], "rgb_dists": []} for k in ("bilinear", "bicubic", "artcnn_chroma")}
    with torch.no_grad():
        for r in h_records:
            p = args.frames_dir / r["file"]
            if not p.is_file():
                continue
            with Image.open(p) as im:
                w, h = im.size
                im = im.convert("RGB").crop((0, 0, w // 4 * 4, h // 4 * 4))
                rgb_t = torch.from_numpy(np.asarray(im, dtype=np.float32).transpose(2, 0, 1))[None].to(device) / 255.0
            y_hr, c_hr = rgb_to_ycbcr(rgb_t)
            c_lr = (F.avg_pool2d(c_hr, 2) * 224.0).round() / 224.0
            preds = {
                "bilinear": F.interpolate(c_lr, scale_factor=2, mode="bilinear", align_corners=False),
                "bicubic": F.interpolate(c_lr, scale_factor=2, mode="bicubic", align_corners=False).clamp(0, 1),
                "artcnn_chroma": net(y_hr, c_lr).clamp(0, 1),
            }
            tc = c_hr[:, :, 8:-8, 8:-8]
            tr = rgb_t[:, :, 8:-8, 8:-8]
            yi = y_hr[:, :, 8:-8, 8:-8]
            for name, pc in preds.items():
                pc_in = pc[:, :, 8:-8, 8:-8]
                pr = ycbcr_to_rgb(yi, pc_in)
                metrics[name]["c_psnr"].append(float((-10.0 * (pc_in - tc).square().mean().clamp_min(1e-10).log10()).item()))
                metrics[name]["rgb_psnr"].append(float((-10.0 * (pr - tr).square().mean().clamp_min(1e-10).log10()).item()))
                metrics[name]["rgb_lpips"].append(float(lpips_fn(pr, tr).item()))
                metrics[name]["rgb_dists"].append(float(dists_fn(pr, tr).item()))

    summary = {k: {m: float(np.mean(v)) for m, v in d.items()} for k, d in metrics.items()}
    print("Holdout 96-frame Chroma/RGB evaluation:")
    for k, d in summary.items():
        print(f"  {k:<15}: Chroma PSNR={d['c_psnr']:.2f} dB | RGB PSNR={d['rgb_psnr']:.2f} dB | RGB LPIPS={d['rgb_lpips']:.4f} | RGB DISTS={d['rgb_dists']:.4f}")

    args.output.mkdir(parents=True, exist_ok=True)
    net.save_npz(args.output / "ArtCNN_Chroma_Fast.npz")
    shader = export_chroma_glsl(net, label=f"C{cfg.internal_layers}F{cfg.filters}_Chroma")
    (args.output / "ArtCNN_Chroma_Fast.glsl").write_text(shader, encoding="utf-8", newline="\n")
    (args.output / "chroma_holdout.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Exported chroma checkpoint & GLSL shader to {args.output}")


if __name__ == "__main__":
    main()
