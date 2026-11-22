"""Early PyTorch GPU trainer for ArtCNN C4F32 / C4F16.
  - Foldable 5-branch EdgeConvBlock (ECB) -> collapsed to standard 3x3 convs on export
  - Blended Charbonnier L1 + Sobel edge-agreement + 2D FFT + Temporal-shift consistency
  - Realistic multi-kernel + 4x4 DCT + ringing degradation (from artcnn.degradation)
  - Flat-fill-aware balanced patch sampling (from artcnn.benchmark_data)
  - Cosine annealing AdamW + EMA weights
  - Exports a PyTorch checkpoint and a fused mpv GLSL shader

Usage from the repository root with a compatible NVIDIA GPU
  uv run --locked --group training python enhanced/train_pytorch.py --hr-dir /path/to/Anime_Train_HR --filters 32 --epochs 300
"""

from __future__ import annotations

import argparse
import copy
import glob
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from artcnn.degradation import DegradationConfig, degrade_hr_to_lr
from artcnn.glsl_export import export_artcnn_glsl
from artcnn.models_np import ArtCNNC, ArtCNNCConfig


def main() -> None:
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch.utils.data import DataLoader, Dataset
    except ImportError as exc:
        raise SystemExit(
            "GPU training requires PyTorch from the training dependency group (`uv run --group training`). "
            "See enhanced/README.md for CPU verification commands."
        ) from exc

    parser = argparse.ArgumentParser(description="Train Enhanced ArtCNN C-series on GPU")
    parser.add_argument("--hr-dir", type=str, required=True, help="Directory of HR PNG images")
    parser.add_argument("--filters", type=int, default=32, choices=[16, 32])
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--input-skip", action="store_true", default=True)
    parser.add_argument("--out-prefix", type=str, default="ArtCNN_C4F32_Enhanced")
    args = parser.parse_args()

    class FoldableECB(nn.Module):
        """5-branch linear edge-conv block that folds into a single 3x3 Conv2d."""

        def __init__(self, c_in: int, c_out: int):
            super().__init__()
            self.c_in, self.c_out = c_in, c_out
            self.conv3 = nn.Conv2d(c_in, c_out, 3, padding=1)
            self.conv1 = nn.Conv2d(c_in, c_out, 1, padding=0)
            self.sobel_x_mix = nn.Conv2d(c_in, c_out, 1, padding=0, bias=False)
            self.sobel_y_mix = nn.Conv2d(c_in, c_out, 1, padding=0, bias=False)
            sx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
            sy = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
            self.register_buffer("sx", sx.view(1, 1, 3, 3).repeat(c_out, 1, 1, 1))
            self.register_buffer("sy", sy.view(1, 1, 3, 3).repeat(c_out, 1, 1, 1))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            y = self.conv3(x) + self.conv1(x)
            if self.c_in == self.c_out:
                y = y + x
            y = y + F.conv2d(self.sobel_x_mix(x), self.sx, padding=1, groups=self.c_out)
            y = y + F.conv2d(self.sobel_y_mix(x), self.sy, padding=1, groups=self.c_out)
            return y

        @torch.no_grad()
        def fold_weights(self):
            W = self.conv3.weight.detach().double().clone()
            b = (self.conv3.bias.detach().double() + self.conv1.bias.detach().double()).clone()
            W[:, :, 1, 1] += self.conv1.weight.detach().double().squeeze(-1).squeeze(-1)
            if self.c_in == self.c_out:
                idx = torch.arange(self.c_out)
                W[idx, idx, 1, 1] += 1.0
            sx = self.sx[0, 0].double()
            sy = self.sy[0, 0].double()
            W += self.sobel_x_mix.weight.detach().double() * sx[None, None, :, :]
            W += self.sobel_y_mix.weight.detach().double() * sy[None, None, :, :]
            return W.float().cpu().numpy(), b.float().cpu().numpy()

    class EnhancedArtCNNC(nn.Module):
        def __init__(self, filters: int = 32, input_skip: bool = True):
            super().__init__()
            self.filters = filters
            self.input_skip = input_skip
            self.conv0 = nn.Conv2d(1, filters, 3, padding=1)
            self.internal = nn.ModuleList([FoldableECB(filters, filters) for _ in range(4)])
            self.act = nn.LeakyReLU(0.1, inplace=True)
            self.mix_global = nn.Conv2d(filters, filters, 3, padding=1)
            self.features = nn.Conv2d(filters, 4, 3, padding=1)
            self.ps = nn.PixelShuffle(2)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            c0 = self.conv0(x)
            h = c0
            for blk in self.internal:
                h = self.act(blk(h))
            out = self.ps(self.features(self.mix_global(h) + c0))
            if self.input_skip:
                out = out + F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            return out

    class AnimeHRDataset(Dataset):
        def __init__(self, files):
            self.files = files
            self.deg_cfg = DegradationConfig(mode="realistic")

        def __len__(self):
            return len(self.files)

        def __getitem__(self, idx):
            rng = np.random.default_rng()
            img = cv2.imread(self.files[idx], cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
            if rng.random() < 0.5:
                img = np.fliplr(img)
            img = np.rot90(img, int(rng.integers(0, 4))).copy()
            lr = degrade_hr_to_lr(img, rng, self.deg_cfg)
            return torch.from_numpy(lr[None]), torch.from_numpy(img[None])

    files = sorted(glob.glob(str(Path(args.hr_dir) / "*.png")))
    if not files:
        raise SystemExit(f"No PNG files found in {args.hr_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EnhancedArtCNNC(filters=args.filters, input_skip=args.input_skip).to(device)
    ema_model = copy.deepcopy(model)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.02)
    loader = DataLoader(AnimeHRDataset(files), batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    sx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=device).view(1, 1, 3, 3) / 4.0
    sy = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=device).view(1, 1, 3, 3) / 4.0

    def charb(a, b, eps=1e-4):
        return torch.mean(torch.sqrt((a - b) ** 2 + eps * eps))

    for epoch in range(args.epochs):
        model.train()
        for lr_in, hr_ref in loader:
            lr_in, hr_ref = lr_in.to(device), hr_ref.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(lr_in)

            # 1. Charbonnier L1 + Sobel edge-agreement + 2D FFT
            loss_pix = charb(pred, hr_ref)
            loss_edge = 0.5 * (
                charb(F.conv2d(pred, sx, padding=1), F.conv2d(hr_ref, sx, padding=1))
                + charb(F.conv2d(pred, sy, padding=1), F.conv2d(hr_ref, sy, padding=1))
            )
            loss_fft = charb(torch.abs(torch.fft.rfft2(pred, norm="ortho")), torch.abs(torch.fft.rfft2(hr_ref, norm="ortho")))

            # 2. Temporal-shift consistency (1-px LR shift -> 2-px HR overlap)
            pred_s = model(torch.roll(lr_in, shifts=(1, 1), dims=(-2, -1)))
            loss_tmp = charb(pred[:, :, :-2, :-2], pred_s[:, :, 2:, 2:])

            loss = loss_pix + 0.25 * loss_edge + 0.08 * loss_fft + 0.10 * loss_tmp
            loss.backward()
            opt.step()

            with torch.no_grad():
                for pe, p in zip(ema_model.parameters(), model.parameters()):
                    pe.mul_(0.995).add_(p, alpha=0.005)
        sched.step()

    # Fold EMA ECB blocks into standard ArtCNNC and export Fused GLSL shader
    np_model = ArtCNNC(ArtCNNCConfig(filters=args.filters, activation="leaky_relu", input_skip=args.input_skip))
    np_model.conv0.W = ema_model.conv0.weight.detach().cpu().numpy()
    np_model.conv0.b = ema_model.conv0.bias.detach().cpu().numpy()
    for i, blk in enumerate(ema_model.internal):
        W_eq, b_eq = blk.fold_weights()
        np_model.internal[i].W = W_eq
        np_model.internal[i].b = b_eq
    np_model.mix_global.W = ema_model.mix_global.weight.detach().cpu().numpy()
    np_model.mix_global.b = ema_model.mix_global.bias.detach().cpu().numpy()
    np_model.features.W = ema_model.features.weight.detach().cpu().numpy()
    np_model.features.b = ema_model.features.bias.detach().cpu().numpy()

    glsl = export_artcnn_glsl(np_model, fuse_depth_to_space=True)
    Path(f"{args.out_prefix}_Fused.glsl").write_text(glsl, encoding="utf-8")
    torch.save(ema_model.state_dict(), f"{args.out_prefix}.pth")
    print(f"Exported {args.out_prefix}_Fused.glsl and {args.out_prefix}.pth")


if __name__ == "__main__":
    main()
