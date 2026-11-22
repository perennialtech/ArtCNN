# SPDX-FileCopyrightText: 2026 perennialtech contributors
# SPDX-License-Identifier: MIT

"""Reconstruction quality of stock and candidate CPU ONNX models on held-out frames."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, maximum_filter, minimum_filter, sobel
from skimage.metrics import structural_similarity



def encode(lows, codec):
    height, width = lows[0].shape
    def pack(low):
        codes = np.rint(16 + 219 * low).clip(16, 235).astype(np.uint8)
        return codes.tobytes() + bytes([128]) * (low.size // 2)
    raw = b"".join(map(pack, lows))
    command = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-threads", "1"]
    options = (["-qp", "32", "-x264-params", "aq-mode=0"] if codec == "h264"
               else ["-x265-params", "qp=34:pools=none:frame-threads=1:log-level=error"])
    compressed = subprocess.run([
        *command, "-f", "rawvideo", "-pix_fmt", "yuv420p", "-s", f"{width}x{height}", "-r", "24",
        "-i", "pipe:", "-an", "-c:v", f"libx{264 if codec == 'h264' else 265}", "-preset",
        "veryfast", "-threads", "1", "-g", "1", *options, "-f", codec, "pipe:",
    ], input=raw, stdout=subprocess.PIPE, check=True).stdout
    decoded = subprocess.run([
        *command, "-f", codec, "-i", "pipe:", "-an", "-pix_fmt", "yuv420p", "-f", "rawvideo", "pipe:",
    ], input=compressed, stdout=subprocess.PIPE, check=True).stdout
    pixels = np.frombuffer(decoded, np.uint8).reshape(len(lows), height * width * 3 // 2)
    return [((row[:height*width].reshape(height, width).astype(np.float32) - 16) / 219).clip(0, 1)
            for row in pixels]


def prepare(manifest, *, full_size=None, limit=None):
    records = json.loads(manifest.read_text())
    if limit is not None and limit < len(records):
        records = [records[(2 * i + 1) * len(records) // (2 * limit)] for i in range(limit)]
    refs = []
    matrix = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    width, height = full_size or (512, 288)
    for row in records:
        src = manifest.parent / "frames" / row["file"]
        raw = src.read_bytes()
        if hashlib.sha256(raw).hexdigest() != row["sha256"]:
            raise ValueError(f"Input hash mismatch: {src}")
        with Image.open(src) as image:
            if full_size is not None:
                reference = image.convert("RGB").resize(full_size, Image.Resampling.LANCZOS)
            else:
                rng = np.random.default_rng(int(row["sha256"][:16], 16))
                x = rng.integers(0, image.width - width + 1)
                y = rng.integers(0, image.height - height + 1)
                reference = image.crop((x, y, x+width, y+height)).convert("RGB")
            refs.append((np.asarray(reference, dtype=np.float32) / 255) @ matrix)
    def area(image):
        return image.reshape(height//2, 2, width//2, 2).mean(axis=(1, 3))
    def codes(image):
        return np.rint(219 * image).clip(0, 219) / 219
    area_inputs = [area(ref) for ref in refs]
    bicubic = [np.asarray(Image.fromarray(ref).resize((width//2, height//2), Image.Resampling.BICUBIC))
               for ref in refs]
    return records, refs, {
        "area": list(map(codes, area_inputs)),
        "blur-area": [codes(area(gaussian_filter(ref, 0.8))) for ref in refs],
        "h264-qp32": encode(area_inputs, "h264"),
        "bicubic-hevc": encode(bicubic, "hevc"),
    }


def reference_metrics(ref):
    dx = sobel(ref, axis=1) / 8
    dy = sobel(ref, axis=0) / 8
    edge = np.hypot(dx, dy) > 0.04
    halo = (maximum_filter(edge.astype(np.uint8), size=7) > 0) & (~edge)
    flat = maximum_filter(ref, size=7) - minimum_filter(ref, size=7) < 0.025
    return ref, dx, dy, edge, halo, flat


def score(ref_data, pred):
    ref, dx, dy, edge, halo, flat = ref_data
    interior = np.s_[16:-16, 16:-16]
    error = pred - ref
    e = error[interior]
    edge, halo, flat = edge[interior], halo[interior], flat[interior]
    ref_inner, pred_inner = ref[interior], pred[interior]
    grad_err = np.hypot(sobel(pred, axis=1) / 8 - dx, sobel(pred, axis=0) / 8 - dy)[interior]
    return {
        "psnr": float(-10 * np.log10(max(float(np.mean(e**2)), 1e-16))),
        "full_psnr": float(-10 * np.log10(max(float(np.mean(error**2)), 1e-16))),
        "ssim": float(structural_similarity(ref_inner, pred_inner, data_range=1.0, gaussian_weights=True,
                                           sigma=1.5, use_sample_covariance=False)),
        "edge_mae": float(np.mean(grad_err[edge])) if edge.any() else None,
        "halo_mae": float(np.mean(np.abs(e[halo]))) if halo.any() else None,
        "flat_mae": float(np.mean(np.abs(e[flat]))) if flat.any() else None,
        "bias": float(np.mean(e)),
        "black_fraction": float(np.mean(pred < 1 / 255)),
    }


def evaluate(args):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from engine import Engine
    records, refs, inputs = prepare(args.manifest, full_size=args.full_size, limit=args.limit)
    work = list(zip(records, map(reference_metrics, refs), strict=True))
    with args.output.open("w") as out:
        for spec in args.models:
            label, model_file = spec.split("=", 1) if "=" in spec else (Path(spec).stem, spec)
            if label == "bicubic":
                def run(frame):
                    h, w = frame.shape[2:]
                    im = Image.fromarray(frame[0, 0]).resize((w*2, h*2), Image.Resampling.BICUBIC)
                    return np.asarray(im).clip(0, 1)[None, None]
            else:
                backend = "onnxruntime" if args.backend == "ort" else "openvino"
                run = Engine(model_file, cpu_backend=backend, cpu_threads=args.threads).run
            for regime, lows in inputs.items():
                measurements = []
                for (row, ref), low in zip(work, lows, strict=True):
                    if label == "bicubic":
                        pred = run(low[None, None])[0, 0]
                    else:
                        pred = run(low)[..., 0]
                    if pred.shape != refs[0].shape or not np.isfinite(pred).all():
                        raise ValueError(f"Invalid prediction from {label}")
                    metrics = score(ref, pred)
                    out.write(json.dumps({"model": label, "group": row["group"], "id": row["id"],
                                          "regime": regime, **metrics}) + "\n")
                    measurements.append(metrics["psnr"])
                print(label, regime, len(measurements), round(float(np.mean(measurements)), 4), flush=True)
                out.flush()
            if label != "bicubic":
                del run


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--backend", choices=["ort", "openvino"], default="openvino")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--limit", type=int)
    p.add_argument("--full-size", nargs=2, type=int, metavar=("WIDTH", "HEIGHT"))
    p.add_argument("--fetch", action="store_true", help="Download missing pinned input frames")
    args = p.parse_args()
    if args.full_size is not None:
        args.full_size = tuple(args.full_size)
        if any(size < 64 or size % 4 for size in args.full_size):
            p.error("Full-frame dimensions must be multiples of four and at least 64.")
    if args.limit is not None and args.limit < 1:
        p.error("Frame limit must be positive.")
    if args.fetch:
        from data import fetch_frames

        fetch_frames(args.manifest)
    evaluate(args)


if __name__ == "__main__":
    main()
