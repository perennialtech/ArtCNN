# SPDX-FileCopyrightText: 2026 perennialtech contributors
# SPDX-License-Identifier: MIT

"""Train Pareto-frontier clean-still RGB ArtCNN models for browser WebGPU inference."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import time
from typing import Dict, List

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from models import ArtCNNRGB, ArtCNNRGBConfig
from perceptual import (
    DISTS,
    fft_magnitude_loss,
    find_dists_weights,
    sobel_edge_loss,
    ssim_loss,
    ssim_map,
)


def _load_rgb_frame(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Failed to read image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w, _ = rgb.shape
    return rgb[: h - (h % 2), : w - (w % 2)]


_LANCZOS_KERNEL_CACHE: Dict[torch.device, torch.Tensor] = {}


def _lanczos3_2x_downsample(hr: torch.Tensor) -> torch.Tensor:
    """Fast GPU separable Lanczos-3 2x downsampler for (B, 3, H, W)."""
    dev = hr.device
    if dev not in _LANCZOS_KERNEL_CACHE:
        # 1D Lanczos-3 kernel for 2x downsampling centered at +0.5
        t = torch.arange(-5, 7, dtype=torch.float32, device=dev) - 0.5
        x = t * 0.5
        k = torch.where(x.abs() < 1e-6, torch.ones_like(x), torch.sin(math.pi * x) * torch.sin(math.pi * x / 3.0) / ((math.pi * x) ** 2 / 3.0))
        k = torch.where(x.abs() >= 3.0, torch.zeros_like(x), k)
        k = k / k.sum()
        _LANCZOS_KERNEL_CACHE[dev] = k
    k = _LANCZOS_KERNEL_CACHE[dev]
    c = hr.shape[1]
    kh = k.view(1, 1, 1, -1).expand(c, 1, 1, -1)
    kv = k.view(1, 1, -1, 1).expand(c, 1, -1, 1)
    x = F.pad(hr, (5, 6, 0, 0), mode="replicate")
    x = F.conv2d(x, kh, stride=(1, 2), groups=c)
    x = F.pad(x, (0, 0, 5, 6), mode="replicate")
    x = F.conv2d(x, kv, stride=(2, 1), groups=c)
    return x.clamp(0.0, 1.0)


def clean_downscale_batch(hr: torch.Tensor, mode: int) -> torch.Tensor:
    """Clean 2x downscale on GPU tensor (B, 3, H, W) in [0, 1]."""
    if mode == 0:
        return F.avg_pool2d(hr, kernel_size=2, stride=2)
    if mode == 1:
        return F.interpolate(hr, scale_factor=0.5, mode="bicubic", align_corners=False, antialias=True).clamp(0.0, 1.0)
    if mode == 2:
        return F.interpolate(hr, scale_factor=0.5, mode="bilinear", align_corners=False, antialias=True).clamp(0.0, 1.0)
    return _lanczos3_2x_downsample(hr)


class CleanPatchDataset(Dataset):
    """Edge-aware random HR RGB patch sampler from clean stills."""

    def __init__(self, frames: List[np.ndarray], patch: int, steps: int, batch: int, seed: int):
        self.frames = frames
        self.patch = patch
        self.total = steps * batch
        weights = []
        for f in frames:
            sub = f[::8, ::8, 1].astype(np.int16)
            gx = np.abs(np.diff(sub, axis=1)).mean()
            gy = np.abs(np.diff(sub, axis=0)).mean()
            weights.append(float(gx + gy) + 1.0)
        p = np.array(weights, dtype=np.float64)
        self.probs = p / p.sum()

    def __len__(self) -> int:
        return self.total

    def __getitem__(self, index: int) -> torch.Tensor:
        rng = np.random.default_rng(index + 1337)
        idx = int(rng.choice(len(self.frames), p=self.probs))
        img = self.frames[idx]
        h, w, _ = img.shape
        ps = self.patch
        if h < ps or w < ps:
            img = cv2.resize(img, (max(w, ps), max(h, ps)), interpolation=cv2.INTER_CUBIC)
            h, w, _ = img.shape
        best_crop = None
        best_score = -1.0
        trials = 2 if rng.random() < 0.7 else 1
        for _ in range(trials):
            y = int(rng.integers(0, (h - ps) // 2 + 1)) * 2
            x = int(rng.integers(0, (w - ps) // 2 + 1)) * 2
            crop = img[y : y + ps, x : x + ps]
            score = float(np.abs(np.diff(crop.astype(np.float32), axis=1)).mean())
            if score > best_score:
                best_score = score
                best_crop = crop
        assert best_crop is not None
        k = int(rng.integers(0, 4))
        if k:
            best_crop = np.rot90(best_crop, k)
        if rng.random() < 0.5:
            best_crop = np.fliplr(best_crop)
        arr = np.ascontiguousarray(best_crop.transpose(2, 0, 1), dtype=np.float32) / 255.0
        return torch.from_numpy(arr)


@torch.no_grad()
def evaluate_clean_frames(model: ArtCNNRGB, frames: List[np.ndarray], device: torch.device) -> Dict[str, float]:
    model.eval()
    kernels = [cv2.INTER_AREA, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4]
    psnrs, ssims = [], []
    for hr in frames:
        h, w, _ = hr.shape
        hr_t = torch.from_numpy(hr.transpose(2, 0, 1).astype(np.float32) / 255.0).unsqueeze(0).to(device)
        for interp in kernels:
            lr = cv2.resize(hr, (w // 2, h // 2), interpolation=interp)
            lr_t = torch.from_numpy(lr.transpose(2, 0, 1).astype(np.float32) / 255.0).unsqueeze(0).to(device)
            pred = model(lr_t).clamp(0.0, 1.0)
            mse = float(F.mse_loss(pred, hr_t).item())
            psnrs.append(10.0 * math.log10(1.0 / max(mse, 1e-12)))
            ssims.append(float(ssim_map(pred, hr_t)[0].mean().item()))
    return {"psnr": float(np.mean(psnrs)), "ssim": float(np.mean(ssims))}


def load_frames_from_manifest(
    manifest_path: Path, frames_dir: Path, limit: int | None = None, exclude_group: str | None = None
) -> List[np.ndarray]:
    entries = json.loads(manifest_path.read_text(encoding="utf-8"))
    paths = []
    for e in entries:
        if exclude_group and e.get("group") == exclude_group:
            continue
        p = frames_dir / e["file"]
        if p.exists():
            paths.append(p)
    if limit is not None:
        paths = paths[:limit]
    with ThreadPoolExecutor(max_workers=8) as ex:
        return list(ex.map(_load_rgb_frame, paths))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--train-frames", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--val-frames", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--family", choices=["c", "r"], default="c")
    parser.add_argument("--filters", type=int, default=16)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--patch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--ecb", action="store_true")
    parser.add_argument("--qat-frac", type=float, default=0.25)
    parser.add_argument("--eval-every", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading training frames from {args.train_frames}...", flush=True)
    train_frames = load_frames_from_manifest(args.train_manifest, args.train_frames, exclude_group="Hayao_0")
    val_frames = load_frames_from_manifest(args.val_manifest, args.val_frames, limit=12)
    print(f"Loaded {len(train_frames)} train frames, {len(val_frames)} val frames on {device}", flush=True)

    cfg = ArtCNNRGBConfig(family=args.family, filters=args.filters, blocks=args.blocks)
    model = ArtCNNRGB(cfg, use_ecb=args.ecb).to(device)
    ema_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    dists_fn = None
    dists_w = None
    if dists_w:
        dists_fn = DISTS(*dists_w).to(device).eval()

    ds = CleanPatchDataset(train_frames, args.patch, args.steps, args.batch, args.seed)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=0, drop_last=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=args.lr * 0.03)

    best_psnr = -1.0
    best_state = None
    qat_start = int(args.steps * (1.0 - args.qat_frac))
    rng = np.random.default_rng(args.seed)
    t0 = time.time()

    for step, hr in enumerate(loader, start=1):
        model.train()
        model.qat_fp16 = step >= qat_start
        hr = hr.to(device, non_blocking=True)
        mode = int(rng.choice(4, p=[0.40, 0.25, 0.10, 0.25]))
        lr = clean_downscale_batch(hr, mode)

        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            pred = model(lr)
        pred_f = pred.float()
        charb = torch.sqrt((pred_f - hr) ** 2 + 1e-6).mean()
        edge = sobel_edge_loss(pred_f.reshape(-1, 1, pred_f.shape[2], pred_f.shape[3]), hr.reshape(-1, 1, hr.shape[2], hr.shape[3]))
        fft_l = fft_magnitude_loss(pred_f, hr)
        ssim_l = ssim_loss(pred_f.clamp(0.0, 1.0), hr)
        loss = charb + 0.12 * edge + 0.04 * fft_l + 0.04 * ssim_l
        if dists_fn is not None:
            loss = loss + 0.02 * dists_fn(pred_f.clamp(0.0, 1.0), hr).mean()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        with torch.no_grad():
            decay = min(0.995, (1.0 + step) / (10.0 + step))
            for k, v in model.state_dict().items():
                ema_state[k].mul_(decay).add_(v, alpha=1.0 - decay)

        if step % args.eval_every == 0 or step == args.steps:
            saved = {k: v.clone() for k, v in model.state_dict().items()}
            model.load_state_dict(ema_state)
            metrics = evaluate_clean_frames(model, val_frames, device)
            dt = time.time() - t0
            print(
                f"[{cfg.arch_name()}] step {step}/{args.steps} ({dt:.1f}s) "
                f"loss={loss.item():.4f} val_psnr={metrics['psnr']:.3f} val_ssim={metrics['ssim']:.5f}",
                flush=True,
            )
            if metrics["psnr"] > best_psnr:
                best_psnr = metrics["psnr"]
                best_state = {k: v.detach().cpu().clone() for k, v in ema_state.items()}
            model.load_state_dict(saved)

    assert best_state is not None
    model.load_state_dict(best_state)
    model.export_npz(args.output)
    print(f"Saved best {cfg.arch_name()} checkpoint (val_psnr={best_psnr:.3f}) -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
