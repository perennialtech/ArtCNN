# SPDX-FileCopyrightText: 2026 perennialtech contributors
# SPDX-License-Identifier: MIT

"""Fine-tune or distill ArtCNN models on the pinned real-frame training set."""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import json
import math
from pathlib import Path
import time

import numpy as np
import onnx
import torch
from onnx import numpy_helper
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from artcnn.discriminators import (
    SpectralNormPatchGAN,
    SpectralNormUNetDiscriminator,
    feature_matching_loss,
    gan_d_loss,
    gan_g_loss,
    subpixel_shimmer_loss,
)
from artcnn.glsl_export import export_artcnn_glsl
from artcnn.models_np import ArtCNNC, ArtCNNCConfig
from artcnn.perceptual import (
    AlexNetLPIPS,
    FoldableECB,
    fft_magnitude_loss,
    find_lpips_weights,
    halo_ringing_mae,
    sobel_edge_loss,
    ssim_loss,
    ssim_map,
)
from experiments.tune_workgroups import retile_shader

RECIPES = {
    "clean": (0.90, 0.08, 0.02),
    "balanced": (0.48, 0.24, 0.28),
    "restoration": (0.12, 0.38, 0.50),
}


class GroupedMix(nn.Module):
    def __init__(self, filters: int):
        super().__init__()
        self.spatial = nn.Conv2d(filters, filters, 3, padding=1, groups=filters // 4, bias=False)
        self.mix = nn.Conv2d(filters, filters, 1)

    def forward(self, x):
        return self.mix(self.spatial(x))

    @torch.no_grad()
    def initialize(self, weight, bias):
        for start in range(0, len(weight), 4):
            channels = slice(start, start + 4)
            u, s, vh = np.linalg.svd(weight[:, channels].reshape(len(weight), -1), full_matrices=False)
            scale = np.sqrt(s[:4])
            self.spatial.weight[channels].copy_(torch.from_numpy((scale[:, None] * vh[:4]).reshape(4, 4, 3, 3)))
            mixing = torch.from_numpy(u[:, :4] * scale[None, :])
            self.mix.weight[:, channels, 0, 0].copy_(mixing)
        self.mix.bias.copy_(torch.from_numpy(bias.copy()))


class Model(nn.Module):
    def __init__(self, config: ArtCNNCConfig, grouped: bool = False, use_ecb: bool = False):
        super().__init__()
        self.config = config
        self.grouped = grouped
        self.use_ecb = use_ecb and not grouped
        filters = config.filters
        middle = []
        for i in range(config.internal_layers + 1):
            if grouped:
                middle.append(GroupedMix(filters))
            elif self.use_ecb and i < config.internal_layers:
                middle.append(FoldableECB(filters, filters))
            else:
                middle.append(nn.Conv2d(filters, filters, 3, padding=1))
        self.convs = nn.ModuleList([nn.Conv2d(1, filters, 3, padding=1), *middle, nn.Conv2d(filters, 4, 3, padding=1)])
        if config.input_skip:
            nn.init.zeros_(self.convs[-1].weight)
            nn.init.zeros_(self.convs[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = self.convs[0](x)
        h = skip
        for layer in self.convs[1:-2]:
            h = layer(h)
            h = F.leaky_relu(h, self.config.leaky_slope) if self.config.activation == "leaky_relu" else F.relu(h)
        out = F.pixel_shuffle(self.convs[-1](self.convs[-2](h) + skip), 2)
        if self.config.input_skip:
            out = out + F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return out

    @torch.no_grad()
    def _load_conv_weights(self, layer: nn.Module, weight: np.ndarray, bias: np.ndarray) -> None:
        if isinstance(layer, GroupedMix):
            layer.initialize(weight, bias)
        elif isinstance(layer, FoldableECB):
            layer.initialize_from_conv(torch.from_numpy(weight.copy()), torch.from_numpy(bias.copy()))
        else:
            if tuple(weight.shape) != tuple(layer.weight.shape):
                raise ValueError(f"Unexpected weight shape {weight.shape} != {tuple(layer.weight.shape)}")
            layer.weight.copy_(torch.from_numpy(weight.copy()))
            layer.bias.copy_(torch.from_numpy(bias.copy()))

    @torch.no_grad()
    def load_onnx(self, path: Path):
        model = onnx.load(path)
        nodes = list(model.graph.node)
        pattern = ["Conv"] + ["Conv", "Relu"] * self.config.internal_layers + [
            "Conv", "Add", "Conv", "DepthToSpace", "Clip",
        ]
        if self.config.input_skip or self.config.activation != "relu" or [n.op_type for n in nodes] != pattern:
            raise ValueError(f"Incompatible upstream ArtCNN graph: {path}")
        values = {x.name: x for x in model.graph.initializer}
        for layer, node in zip(self.convs, (n for n in nodes if n.op_type == "Conv"), strict=True):
            weight, bias = (numpy_helper.to_array(values[n]) for n in node.input[1:])
            self._load_conv_weights(layer, weight, bias)

    @torch.no_grad()
    def load_npz(self, path: Path):
        with np.load(path) as data:
            if json.loads(data["config"].item()) != asdict(self.config):
                raise ValueError(f"Checkpoint configuration does not match: {path}")
            for (name, _), layer in zip(ArtCNNC(self.config).layers(), self.convs, strict=True):
                if isinstance(layer, GroupedMix):
                    layer.spatial.weight.copy_(torch.from_numpy(data[f"{name}.spatial"]))
                    layer.mix.weight.copy_(torch.from_numpy(data[f"{name}.mix"][:, :, None, None]))
                    layer.mix.bias.copy_(torch.from_numpy(data[f"{name}.b"]))
                else:
                    self._load_conv_weights(layer, data[f"{name}.W"], data[f"{name}.b"])

    def export(self, directory: Path, stem: str, workgroup: tuple[int, int]):
        portable = ArtCNNC(self.config)
        kernels, extra = {}, {}
        for (name, dst), src in zip(portable.layers(), self.convs, strict=True):
            if isinstance(src, GroupedMix):
                spatial = src.spatial.weight.detach().cpu().numpy().copy()
                mixing = src.mix.weight.detach().cpu().numpy()[:, :, 0, 0].copy()
                for start in range(0, len(spatial), 4):
                    channels = slice(start, start + 4)
                    dst.W[:, channels] = np.einsum("oi,ijxy->ojxy", mixing[:, channels], spatial[channels])
                dst.b = src.mix.bias.detach().cpu().numpy().copy()
                kernels[name] = spatial, mixing
                extra[f"{name}.spatial"] = spatial
                extra[f"{name}.mix"] = mixing
            elif isinstance(src, FoldableECB):
                w, b = src.fold_weights()
                dst.W = w.cpu().numpy().copy()
                dst.b = b.cpu().numpy().copy()
            else:
                dst.W = src.weight.detach().cpu().numpy().copy()
                dst.b = src.bias.detach().cpu().numpy().copy()
        np.savez(directory / f"{stem}.npz", **portable.state_dict(), **extra, config=json.dumps(asdict(self.config)))
        shader = export_artcnn_glsl(portable, fuse_depth_to_space=True, grouped_kernels=kernels)
        (directory / f"{stem}.glsl").write_text(retile_shader(shader, *workgroup), encoding="utf-8", newline="\n")


class ResidualTeacher(nn.Module):
    """Upstream R-series inference graph used during distillation."""

    def __init__(self, graph):
        super().__init__()
        nodes = [node for node in graph.node if node.op_type == "Conv"]
        if len(nodes) < 6 or (len(nodes) - 3) % 3:
            raise ValueError("Expected the upstream R-series residual graph")
        self.layers = nn.ModuleList()
        values = {x.name: x for x in graph.initializer}
        with torch.no_grad():
            for node in nodes:
                weight, bias = (numpy_helper.to_array(values[n]) for n in node.input[1:])
                layer = nn.Conv2d(weight.shape[1], weight.shape[0], 3, padding=1)
                layer.weight.copy_(torch.from_numpy(weight.copy()))
                layer.bias.copy_(torch.from_numpy(bias.copy()))
                self.layers.append(layer)

    def forward(self, x):
        skip = self.layers[0](x)
        state = skip
        for i in range(1, len(self.layers) - 2, 3):
            first = F.silu(self.layers[i](state))
            second = F.silu(self.layers[i+1](first))
            state = state + self.layers[i+2](second)
        return F.pixel_shuffle(self.layers[-1](self.layers[-2](state) + skip), 2)


def load_teacher(path):
    graph = onnx.load(path).graph
    if any(node.op_type == "Sigmoid" for node in graph.node):
        return ResidualTeacher(graph)
    first = next(node for node in graph.node if node.op_type == "Conv")
    width = next(x.dims[0] for x in graph.initializer if x.name == first.input[1])
    model = Model(ArtCNNCConfig(filters=width))
    model.load_onnx(path)
    return model


class Pairs:
    def __init__(self, path: Path):
        self.sources = []
        index = json.loads((path / "index.json").read_text())
        for group in index:
            stem = group["stem"]
            levels = [np.load(path / f"{stem}-{kind}.npy", mmap_mode="r") for kind in ("hr", "qp24", "qp32")]
            self.sources.extend(zip(*levels, strict=True))
        if not self.sources:
            raise ValueError(f"Empty paired dataset: {path}")

    def crop(self, index, y, x, size, qp_index):
        ref, *compressed = self.sources[index]
        hr = ref[y:y + size, x:x + size]
        lr = compressed[qp_index][y // 2:(y + size) // 2, x // 2:(x + size) // 2]
        return hr, lr


class Patches(Dataset):
    def __init__(self, source, patch, steps, batch, seed):
        self.source = source
        self.patch = patch
        self.count = steps * batch
        self.seed = seed

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, index]))
        i = int(rng.integers(len(self.source.sources)))
        h, w = self.source.sources[i][0].shape
        level = int(rng.integers(2))
        search = 3 if rng.random() < 0.55 else 1
        for _ in range(search):
            y = int(rng.integers((h - self.patch) // 2 + 1)) * 2
            x = int(rng.integers((w - self.patch) // 2 + 1)) * 2
            hr, lr = self.source.crop(i, y, x, self.patch, level)
            if hr[::4, ::4].std() > 15:
                break
        angle = int(rng.integers(4))
        hr, lr = np.rot90(hr, angle), np.rot90(lr, angle)
        if rng.random() < 0.5:
            hr, lr = hr[:, ::-1], lr[:, ::-1]
        return np.ascontiguousarray(hr[None]), np.ascontiguousarray(lr[None])


def gaussian(inputs, sigma):
    axis = torch.arange(-3, 4, device=inputs.device, dtype=inputs.dtype)
    kernel = torch.exp(-0.5 * (axis / sigma).square())
    kernel = kernel / kernel.sum()
    weights = (kernel[:, None] * kernel[None, :])[None, None]
    return F.conv2d(F.pad(inputs, (3, 3, 3, 3), mode="reflect"), weights)


def unsharp_ringing(inputs, amount, sigma=0.8):
    blurred = gaussian(inputs, sigma)
    return (inputs + amount * (inputs - blurred)).clamp(0.0, 1.0)


def quantize(x):
    return (219 * x).round().clamp_(0, 219) / 219


def degradation(hr, compressed, mode, sigma=0.8, rng=None, extended=False):
    if mode == 2:
        if extended and rng is not None and rng.random() < 0.30:
            # Blend slight bicubic-sharpening/ringing or mild blur into compressed regime
            if rng.random() < 0.5:
                return quantize(unsharp_ringing(compressed, float(rng.uniform(0.15, 0.35)), 0.75))
            bic = F.interpolate(hr, scale_factor=0.5, mode="bicubic", align_corners=False, antialias=True).clamp(0, 1)
            # Add compression residual from x264 onto bicubic downscale to simulate bicubic+codec
            area = quantize(F.avg_pool2d(hr, 2))
            return quantize((bic + (compressed - area)).clamp(0, 1))
        return compressed
    if extended and rng is not None and mode == 0 and rng.random() < 0.25:
        return quantize(F.interpolate(hr, scale_factor=0.5, mode="bicubic", align_corners=False, antialias=True).clamp(0, 1))
    filtered = gaussian(hr, sigma) if mode == 1 else hr
    if extended and rng is not None and mode == 1 and rng.random() < 0.25:
        out = F.interpolate(filtered, scale_factor=0.5, mode="bicubic", align_corners=False, antialias=True).clamp(0, 1)
        return quantize(out)
    return quantize(F.avg_pool2d(filtered, 2))


def tensors(hr, lr, device):
    return hr.to(device, non_blocking=True).float() / 255, (lr.to(device, non_blocking=True).float() - 16) / 219


@torch.no_grad()
def validate(model, samples, device, lpips_fn=None):
    model.eval()
    psnr_report, ssim_report, lpips_report, edge_report, halo_report = [], [], [], [], []
    sx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], device=device).view(1, 1, 3, 3) / 8.0
    sy = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], device=device).view(1, 1, 3, 3) / 8.0
    for mode in range(3):
        psnrs, ssims, lps, edges, halos = [], [], [], [], []
        for hr, lr in samples:
            hr, lr = tensors(hr, lr, device)
            pred = model(degradation(hr, lr, mode).to(memory_format=torch.channels_last)).clamp_(0, 1)
            p_in = pred[:, :, 16:-16, 16:-16]
            t_in = hr[:, :, 16:-16, 16:-16]
            residual = p_in - t_in
            psnrs.extend((-10 * residual.square().mean(dim=(1, 2, 3)).clamp_min(1e-10).log10()).cpu().tolist())
            ssims.append(float(ssim_map(p_in, t_in)[0].mean().item()))
            if lpips_fn is not None:
                lps.append(float(lpips_fn(p_in, t_in).item()))
            gx_t, gy_t = F.conv2d(t_in, sx, padding=1), F.conv2d(t_in, sy, padding=1)
            gx_p, gy_p = F.conv2d(p_in, sx, padding=1), F.conv2d(p_in, sy, padding=1)
            mask = torch.hypot(gx_t, gy_t) > 0.04
            if mask.any():
                edges.append(float(torch.hypot(gx_p - gx_t, gy_p - gy_t)[mask].mean().item()))
            halos.append(halo_ringing_mae(p_in, t_in))
        psnr_report.append(float(np.mean(psnrs)))
        ssim_report.append(float(np.mean(ssims)))
        lpips_report.append(float(np.mean(lps)) if lps else 0.0)
        edge_report.append(float(np.mean(edges)) if edges else 0.0)
        halo_report.append(float(np.mean(halos)))
    return {
        "psnr": psnr_report,
        "ssim": ssim_report,
        "lpips": lpips_report,
        "edge_mae": edge_report,
        "halo_mae": halo_report,
    }


def validation_samples(path):
    source = Pairs(path)
    hr, lr = [], []
    patch = 192
    for i, (image, _, _) in enumerate(source.sources):
        height, width = image.shape
        side = min(patch, height, width) // 4 * 4
        if side != patch:
            continue
        for factor in (0.32, 0.63):
            y = int((height - side) * factor) // 2 * 2
            x = int((width - side) * (1 - factor)) // 2 * 2
            hi, lo = source.crop(i, y, x, side, 1)
            hr.append(hi[None])
            lr.append(lo[None])
    if not hr:
        raise ValueError("No validation patches large enough")
    h, l = torch.tensor(np.stack(hr)), torch.tensor(np.stack(lr))
    return list(zip(h.split(8), l.split(8), strict=True))


def pixel_loss(
    pred,
    target,
    teacher=None,
    imitation_weight=0.20,
    *,
    mse_weight=3.0,
    ssim_weight=0.0,
    sobel_weight=0.0,
    fft_weight=0.0,
    lpips_fn=None,
    lpips_weight=0.0,
):
    p = pred.float()[:, :, 12:-12, 12:-12]
    t = target[:, :, 12:-12, 12:-12]
    error = p - t
    objective = (error.square() + 1e-6).sqrt().mean()
    if mse_weight > 0:
        objective = objective + mse_weight * error.square().mean()
    if sobel_weight > 0:
        objective = objective + sobel_weight * sobel_edge_loss(p, t)
    else:
        y = error[:, :, 1:, :] - error[:, :, :-1, :]
        x = error[:, :, :, 1:] - error[:, :, :, :-1]
        objective = objective + 0.10 * (x.abs().mean() + y.abs().mean())
    if ssim_weight > 0:
        objective = objective + ssim_weight * ssim_loss(p.clamp(0.0, 1.0), t)
    if fft_weight > 0:
        objective = objective + fft_weight * fft_magnitude_loss(p, t)
    if lpips_fn is not None and lpips_weight > 0:
        objective = objective + lpips_weight * lpips_fn(p.clamp(0.0, 1.0), t)
    if teacher is not None:
        imitation = (pred.float().clamp(0, 1) - teacher.float())[:, :, 12:-12, 12:-12]
        objective = objective + imitation_weight * imitation.abs().mean()
    return objective


def selection_score(metrics: dict, recipe: np.ndarray, mode: str) -> float:
    psnr = float(np.dot(metrics["psnr"], recipe))
    if mode == "psnr":
        return psnr
    # Perceptual composite score: rewards higher PSNR & SSIM and sharply lower LPIPS & Edge-MAE
    lpips = float(np.dot(metrics["lpips"], recipe))
    ssim = float(np.dot(metrics["ssim"], recipe))
    edge = float(np.dot(metrics["edge_mae"], recipe))
    return psnr + 80.0 * ssim - 60.0 * lpips - 80.0 * edge


def train(args):
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = True
    cuda = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this training run")
    config = ArtCNNCConfig(
        filters=args.filters,
        internal_layers=args.layers,
        activation="relu" if args.upstream else "leaky_relu",
        input_skip=not args.upstream,
    )
    if args.initialize:
        with np.load(args.initialize) as data:
            config = ArtCNNCConfig(**json.loads(data["config"].item()))
    model = Model(config, grouped=args.grouped, use_ecb=args.ecb)
    if args.upstream:
        model.load_onnx(args.upstream)
    if args.initialize:
        model.load_npz(args.initialize)
    model = model.to(device=cuda, memory_format=torch.channels_last)

    teacher = None
    if args.teacher:
        teacher = load_teacher(args.teacher).to(device=cuda, memory_format=torch.channels_last)
        if args.teacher_pt:
            teacher.load_state_dict(torch.load(args.teacher_pt, map_location=cuda, weights_only=True))
        teacher.eval().requires_grad_(False)

    lpips_fn = None
    if args.lpips_weight > 0 or args.select_metric == "perceptual":
        found = find_lpips_weights()
        if found is not None:
            lpips_fn = AlexNetLPIPS(*found).to(cuda)
        elif args.lpips_weight > 0:
            raise RuntimeError("Requested --lpips-weight > 0 but AlexNet/LPIPS weights were not found")

    ema = copy.deepcopy(model).eval().requires_grad_(False)
    paired = Pairs(args.train)
    patches = Patches(paired, args.patch, args.steps, args.batch, args.seed)
    batches = DataLoader(patches, batch_size=args.batch, num_workers=args.workers, pin_memory=True, drop_last=True)
    val = validation_samples(args.dev)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, fused=True, weight_decay=0)
    scaler = torch.amp.GradScaler("cuda", init_scale=256)
    disc = None
    disc_optim = None
    disc_scaler = None
    if args.gan != "none" and (args.gan_weight > 0 or args.fm_weight > 0):
        disc = (
            SpectralNormPatchGAN() if args.gan == "patchgan" else SpectralNormUNetDiscriminator()
        ).to(device=cuda, memory_format=torch.channels_last)
        disc_optim = torch.optim.AdamW(disc.parameters(), lr=args.d_lr, betas=(0.5, 0.999), fused=True, weight_decay=0)
        disc_scaler = torch.amp.GradScaler("cuda", init_scale=256)
    rng = np.random.default_rng(args.seed)
    recipe = np.array(RECIPES[args.recipe])
    args.output.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    metrics = validate(ema, val, cuda, lpips_fn=lpips_fn)
    best = selection_score(metrics, recipe, args.select_metric)
    print(
        f"Initial PSNR {[round(x, 3) for x in metrics['psnr']]} "
        f"LPIPS {[round(x, 4) for x in metrics['lpips']]} "
        f"Edge {[round(x, 4) for x in metrics['edge_mae']]} score={best:.3f}",
        flush=True,
    )
    if args.upstream or args.initialize:
        ema.export(args.output, "best", tuple(args.workgroup))
    history = [{"step": 0, **metrics, "score": best}]
    for step, (hr, compressed) in enumerate(batches, start=1):
        hr, compressed = tensors(hr, compressed, cuda)
        mode = int(rng.choice(3, p=recipe))
        sigma_step = float(rng.uniform(0.55, 1.1))
        lr = degradation(
            hr, compressed, mode, sigma_step, rng=rng, extended=args.extended_degradation
        ).to(memory_format=torch.channels_last)
        rate = args.lr * (0.12 + 0.88 * (1 + math.cos(math.pi * step / args.steps)) / 2)
        optim.param_groups[0]["lr"] = rate
        optim.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            expected = teacher(lr).detach().clamp_(0, 1) if teacher is not None else None
            pred = model(lr)
        value = pixel_loss(
            pred,
            hr,
            expected,
            args.imitation,
            mse_weight=args.mse_weight,
            ssim_weight=args.ssim_weight,
            sobel_weight=args.sobel_weight,
            fft_weight=args.fft_weight,
            lpips_fn=lpips_fn,
            lpips_weight=args.lpips_weight,
        )
        if args.shimmer_weight > 0 and mode in (0, 1):
            with torch.autocast("cuda", dtype=torch.float16):
                def _down(h_img):
                    f_img = gaussian(h_img, sigma_step) if mode == 1 else h_img
                    return quantize(F.avg_pool2d(f_img, 2))
                value = value + args.shimmer_weight * subpixel_shimmer_loss(model, hr, pred, _down)
        if disc is not None:
            disc.requires_grad_(False)
            p_crop = pred.float().clamp(0.0, 1.0)[:, :, 12:-12, 12:-12]
            t_crop = hr[:, :, 12:-12, 12:-12]
            with torch.autocast("cuda", dtype=torch.float16):
                fake_logits, fake_feats = disc(p_crop)
                with torch.no_grad():
                    real_logits, real_feats = disc(t_crop)
            if args.gan_weight > 0:
                value = value + args.gan_weight * gan_g_loss(real_logits.float(), fake_logits.float(), args.gan_loss)
            if args.fm_weight > 0:
                value = value + args.fm_weight * feature_matching_loss(
                    [f.float() for f in real_feats], [f.float() for f in fake_feats]
                )
        scaler.scale(value).backward()
        scaler.unscale_(optim)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optim)
        scaler.update()

        if disc is not None:
            disc.requires_grad_(True)
            disc_optim.zero_grad(set_to_none=True)
            p_det = pred.detach().float().clamp(0.0, 1.0)[:, :, 12:-12, 12:-12]
            t_crop = hr[:, :, 12:-12, 12:-12]
            with torch.autocast("cuda", dtype=torch.float16):
                r_log, _ = disc(t_crop)
                f_log, _ = disc(p_det)
            d_loss = gan_d_loss(r_log.float(), f_log.float(), args.gan_loss)
            disc_scaler.scale(d_loss).backward()
            disc_scaler.unscale_(disc_optim)
            torch.nn.utils.clip_grad_norm_(disc.parameters(), 1.0)
            disc_scaler.step(disc_optim)
            disc_scaler.update()
        with torch.no_grad():
            torch._foreach_lerp_(tuple(ema.parameters()), tuple(model.parameters()), max(0.01, 2 / (step + 2)))
        if step % args.evaluate_every == 0 or step == args.steps:
            metrics = validate(ema, val, cuda, lpips_fn=lpips_fn)
            scalar = selection_score(metrics, recipe, args.select_metric)
            history.append({"step": step, **metrics, "score": scalar, "loss": float(value.detach())})
            (args.output / "training.json").write_text(json.dumps(history, indent=2) + "\n")
            if scalar > best or not (args.output / "best.npz").exists():
                best = scalar
                ema.export(args.output, "best", tuple(args.workgroup))
            print(
                f"Step {step}/{args.steps} {time.monotonic()-start:.1f}s "
                f"PSNR {[round(x, 3) for x in metrics['psnr']]} "
                f"LPIPS {[round(x, 4) for x in metrics['lpips']]} "
                f"Edge {[round(x, 4) for x in metrics['edge_mae']]} score={scalar:.3f}",
                flush=True,
            )
    ema.export(args.output, "last", tuple(args.workgroup))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--upstream", type=Path)
    parser.add_argument("--initialize", type=Path)
    parser.add_argument("--teacher", type=Path)
    parser.add_argument("--filters", type=int, default=16)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--grouped", action="store_true")
    parser.add_argument("--ecb", action="store_true", help="Train with lossless 5-branch FoldableECB blocks")
    parser.add_argument("--recipe", choices=sorted(RECIPES), default="balanced")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--patch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--imitation", type=float, default=0.20)
    parser.add_argument("--mse-weight", type=float, default=3.0)
    parser.add_argument("--ssim-weight", type=float, default=0.0)
    parser.add_argument("--sobel-weight", type=float, default=0.0)
    parser.add_argument("--fft-weight", type=float, default=0.0)
    parser.add_argument("--lpips-weight", type=float, default=0.0)
    parser.add_argument("--gan", choices=["none", "patchgan", "unet"], default="none")
    parser.add_argument("--gan-loss", choices=["hinge", "ragan"], default="hinge")
    parser.add_argument("--gan-weight", type=float, default=0.0)
    parser.add_argument("--fm-weight", type=float, default=0.0)
    parser.add_argument("--d-lr", type=float, default=1e-4)
    parser.add_argument("--shimmer-weight", type=float, default=0.0)
    parser.add_argument("--teacher-pt", type=Path)
    parser.add_argument("--extended-degradation", action="store_true")
    parser.add_argument("--select-metric", choices=["psnr", "perceptual"], default="psnr")
    parser.add_argument("--evaluate-every", type=int, default=500)
    parser.add_argument("--workgroup", type=int, nargs=2, default=[8, 8])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=41001)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
