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

from artcnn.glsl_export import export_artcnn_glsl
from artcnn.models_np import ArtCNNC, ArtCNNCConfig
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
    def __init__(self, config: ArtCNNCConfig, grouped: bool = False):
        super().__init__()
        self.config = config
        filters = config.filters
        middle = [GroupedMix(filters) if grouped else nn.Conv2d(filters, filters, 3, padding=1)
                  for _ in range(config.internal_layers + 1)]
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
            if isinstance(layer, GroupedMix):
                layer.initialize(weight, bias)
            else:
                if tuple(weight.shape) != tuple(layer.weight.shape):
                    raise ValueError(f"Unexpected weight shape in {path} for {node.output[0]}")
                layer.weight.copy_(torch.from_numpy(weight.copy()))
                layer.bias.copy_(torch.from_numpy(bias.copy()))

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
                    layer.weight.copy_(torch.from_numpy(data[f"{name}.W"]))
                    layer.bias.copy_(torch.from_numpy(data[f"{name}.b"]))

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
            else:
                dst.W = src.weight.detach().cpu().numpy().copy()
                dst.b = src.bias.detach().cpu().numpy().copy()
        np.savez(directory / f"{stem}.npz", **portable.state_dict(), **extra, config=json.dumps(asdict(self.config)))
        shader = export_artcnn_glsl(portable, fuse_depth_to_space=True, grouped_kernels=kernels)
        (directory / f"{stem}.glsl").write_text(retile_shader(shader, *workgroup))


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


def quantize(x):
    return (219 * x).round().clamp_(0, 219) / 219


def degradation(hr, compressed, mode, sigma=0.8):
    if mode == 2:
        return compressed
    filtered = gaussian(hr, sigma) if mode == 1 else hr
    return quantize(F.avg_pool2d(filtered, 2))


def tensors(hr, lr, device):
    return hr.to(device, non_blocking=True).float() / 255, (lr.to(device, non_blocking=True).float() - 16) / 219


@torch.no_grad()
def validate(model, samples, device):
    model.eval()
    report = []
    for mode in range(3):
        scores = []
        for hr, lr in samples:
            hr, lr = tensors(hr, lr, device)
            pred = model(degradation(hr, lr, mode).to(memory_format=torch.channels_last)).clamp_(0, 1)
            residual = (pred - hr)[:, :, 16:-16, 16:-16]
            scores.extend((-10 * residual.square().mean(dim=(1, 2, 3)).clamp_min(1e-10).log10()).cpu().tolist())
        report.append(float(np.mean(scores)))
    return report


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


def pixel_loss(pred, target, teacher=None, imitation_weight=0.20):
    error = (pred.float() - target)[:, :, 12:-12, 12:-12]
    y = error[:, :, 1:, :] - error[:, :, :-1, :]
    x = error[:, :, :, 1:] - error[:, :, :, :-1]
    objective = (error.square() + 1e-6).sqrt().mean() + 3 * error.square().mean()
    objective = objective + 0.10 * (x.abs().mean() + y.abs().mean())
    if teacher is not None:
        imitation = (pred.float().clamp(0, 1) - teacher.float())[:, :, 12:-12, 12:-12]
        objective = objective + imitation_weight * imitation.abs().mean()
    return objective


def train(args):
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = True
    cuda = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this training run")
    config = ArtCNNCConfig(filters=args.filters, internal_layers=args.layers,
                           activation="relu" if args.upstream else "leaky_relu", input_skip=not args.upstream)
    if args.initialize:
        with np.load(args.initialize) as data:
            config = ArtCNNCConfig(**json.loads(data["config"].item()))
    model = Model(config, grouped=args.grouped)
    if args.upstream:
        model.load_onnx(args.upstream)
    if args.initialize:
        model.load_npz(args.initialize)
    model = model.to(device=cuda, memory_format=torch.channels_last)
    teacher = None
    if args.teacher:
        teacher = load_teacher(args.teacher).to(device=cuda, memory_format=torch.channels_last)
        teacher.eval().requires_grad_(False)

    ema = copy.deepcopy(model).eval().requires_grad_(False)
    paired = Pairs(args.train)
    patches = Patches(paired, args.patch, args.steps, args.batch, args.seed)
    batches = DataLoader(patches, batch_size=args.batch, num_workers=args.workers, pin_memory=True, drop_last=True)
    val = validation_samples(args.dev)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, fused=True, weight_decay=0)
    scaler = torch.amp.GradScaler("cuda", init_scale=256)
    rng = np.random.default_rng(args.seed)
    recipe = np.array(RECIPES[args.recipe])
    args.output.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    scores = validate(ema, val, cuda)
    print(f"Initial validation PSNR: {scores}", flush=True)
    best = float(np.dot(scores, recipe))
    if args.upstream or args.initialize:
        ema.export(args.output, "best", tuple(args.workgroup))
    history = [{"step": 0, "psnr": scores, "score": best}]
    for step, (hr, compressed) in enumerate(batches, start=1):
        hr, compressed = tensors(hr, compressed, cuda)
        mode = int(rng.choice(3, p=recipe))
        lr = degradation(hr, compressed, mode, float(rng.uniform(0.55, 1.1))).to(memory_format=torch.channels_last)
        rate = args.lr * (0.12 + 0.88 * (1 + math.cos(math.pi * step / args.steps)) / 2)
        optim.param_groups[0]["lr"] = rate
        optim.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            expected = teacher(lr).detach().clamp_(0, 1) if teacher is not None else None
            pred = model(lr)
        value = pixel_loss(pred, hr, expected, args.imitation)
        scaler.scale(value).backward()
        scaler.unscale_(optim)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optim)
        scaler.update()
        with torch.no_grad():
            torch._foreach_lerp_(tuple(ema.parameters()), tuple(model.parameters()), max(0.01, 2 / (step + 2)))
        if step % args.evaluate_every == 0 or step == args.steps:
            scores = validate(ema, val, cuda)
            scalar = float(np.dot(scores, recipe))
            history.append({"step": step, "psnr": scores, "score": scalar, "loss": float(value.detach())})
            (args.output / "training.json").write_text(json.dumps(history, indent=2) + "\n")
            if scalar > best or not (args.output / "best.npz").exists():
                best = scalar
                ema.export(args.output, "best", tuple(args.workgroup))
            print(f"Step {step}/{args.steps} {time.monotonic()-start:.1f}s PSNR "
                  f"{[round(x, 3) for x in scores]} weighted {scalar:.3f}", flush=True)
    ema.export(args.output, "last", tuple(args.workgroup))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    seed_model = parser.add_mutually_exclusive_group()
    seed_model.add_argument("--upstream", type=Path)
    seed_model.add_argument("--initialize", type=Path)
    parser.add_argument("--teacher", type=Path)
    parser.add_argument("--imitation", type=float, default=0.20)
    parser.add_argument("--grouped", action="store_true")
    parser.add_argument("--recipe", choices=RECIPES, default="balanced")
    parser.add_argument("--filters", type=int, choices=(8, 12, 16, 24, 32), default=16)
    parser.add_argument("--layers", type=int, choices=(2, 3, 4, 5), default=4)
    parser.add_argument("--workgroup", nargs=2, type=int, default=(16, 8))
    parser.add_argument("--steps", type=int, default=4500)
    parser.add_argument("--evaluate-every", type=int, default=600)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--patch", type=int, default=96)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=31041)
    args = parser.parse_args()
    if args.patch % 4 or not 32 <= args.patch <= 128:
        parser.error("--patch must be a multiple of four from 32 to 128")
    train(args)


if __name__ == "__main__":
    main()
