# SPDX-FileCopyrightText: 2026 perennialtech contributors
# SPDX-License-Identifier: MIT

"""Prepare luma and aligned H.264 / Bicubic-HEVC inputs for real-frame ArtCNN training."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np
from PIL import Image

BT709 = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def source_luma(path: Path, record: dict) -> np.ndarray:
    if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
        raise ValueError(f"Frame hash mismatch: {path}")
    with Image.open(path) as image:
        width, height = image.size
        cropped = image.convert("RGB").crop((0, 0, width // 4 * 4, height // 4 * 4))
        return np.rint(np.asarray(cropped, dtype=np.float32) @ BT709).clip(0, 255).astype("u1")


def codec_luma(hr: np.ndarray, qp: int, temporary: Path) -> np.ndarray:
    n, h, w = hr.shape
    width, height = w // 2, h // 2
    frames = hr.reshape(n, height, 2, width, 2)
    clean = frames.mean(axis=(2, 4), dtype=np.float32)
    codes = np.rint(16 + clean * (219 / 255)).astype("u1")
    neutral_chroma = bytes([128]) * (width * height // 2)
    source, video = temporary / "input.yuv", temporary / "encoded.h264"
    with source.open("wb") as stream:
        for plane in codes:
            stream.write(plane.tobytes())
            stream.write(neutral_chroma)
    base = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-xerror", "-y"]
    subprocess.run([
        *base, "-f", "rawvideo", "-pixel_format", "yuv420p", "-video_size", f"{width}x{height}",
        "-framerate", "24", "-i", str(source), "-frames:v", str(n),
        "-c:v", "libx264", "-preset", "fast", "-g", "1", "-qp", str(qp),
        "-x264-params", "aq-mode=0", "-threads", "2", "-f", "h264", str(video),
    ], check=True)
    decode = subprocess.run([
        *base, "-threads", "2", "-i", str(video), "-frames:v", str(n),
        "-pix_fmt", "yuv420p", "-f", "rawvideo", "pipe:1",
    ], check=True, stdout=subprocess.PIPE)
    buf = np.frombuffer(decode.stdout, dtype="u1")
    if buf.size != n * width * height * 3 // 2:
        raise RuntimeError(f"Unexpected decoder output size for {width}x{height}")
    return buf.reshape(n, -1)[:, :width * height].reshape(n, height, width).copy()


def hevc_bicubic_luma(hr: np.ndarray, crf: int, temporary: Path) -> np.ndarray:
    """Bicubic-downscale HR luma and encode with real libx265 (inter-frame GOP, realistic AQ)."""
    n, h, w = hr.shape
    width, height = w // 2, h // 2
    codes = np.rint(16 + hr.astype(np.float32) * (219 / 255)).astype("u1")
    neutral_hr_chroma = bytes([128]) * (w * h // 2)
    source, video = temporary / "input_hr.yuv", temporary / "encoded.hevc"
    with source.open("wb") as stream:
        for plane in codes:
            stream.write(plane.tobytes())
            stream.write(neutral_hr_chroma)
    base = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-xerror", "-y"]
    subprocess.run([
        *base, "-f", "rawvideo", "-pixel_format", "yuv420p", "-video_size", f"{w}x{h}",
        "-framerate", "24", "-i", str(source), "-frames:v", str(n),
        "-vf", f"scale={width}:{height}:flags=bicubic",
        "-c:v", "libx265", "-preset", "fast", "-g", "24", "-crf", str(crf),
        "-x265-params", "log-level=error:aq-mode=2", "-threads", "4", "-f", "hevc", str(video),
    ], check=True)
    decode = subprocess.run([
        *base, "-threads", "4", "-i", str(video), "-frames:v", str(n),
        "-pix_fmt", "yuv420p", "-f", "rawvideo", "pipe:1",
    ], check=True, stdout=subprocess.PIPE)
    buf = np.frombuffer(decode.stdout, dtype="u1")
    if buf.size != n * width * height * 3 // 2:
        raise RuntimeError(f"Unexpected HEVC decoder output size for {width}x{height}")
    return buf.reshape(n, -1)[:, :width * height].reshape(n, height, width).copy()


def prepare(manifests: list[Path], output: Path, limit: int = 0, with_hevc: bool = True) -> None:
    inputs = []
    for manifest in manifests:
        records = json.loads(manifest.read_text())
        if limit:
            count = min(limit, len(records))
            records = [records[(2 * i + 1) * len(records) // (2 * count)] for i in range(count)]
        inputs.extend((manifest.parent / "frames" / item["file"], item) for item in records)
    groups = defaultdict(list)
    for path, row in inputs:
        width, height = (v // 4 * 4 for v in row["size"])
        if min(width, height) >= 128:
            groups[(height, width)].append((path, row))
    output.mkdir(parents=True, exist_ok=True)
    batches = []
    with tempfile.TemporaryDirectory(dir=output) as temporary:
        for group_id, ((height, width), group) in enumerate(sorted(groups.items())):
            for first in range(0, len(group), 64):
                chunk = group[first:first + 64]
                hr = np.stack([source_luma(path, row) for path, row in chunk])
                stem = f"{group_id}-{first}"
                np.save(output / f"{stem}-hr.npy", hr)
                for qp in (24, 32):
                    np.save(output / f"{stem}-qp{qp}.npy", codec_luma(hr, qp, Path(temporary)))
                if with_hevc:
                    np.save(output / f"{stem}-hevc.npy", hevc_bicubic_luma(hr, 28, Path(temporary)))
                batches.append({"stem": stem, "size": [width, height], "samples": [row for _, row in chunk]})
                print(f"Prepared {sum(len(b['samples']) for b in batches)}/{len(inputs)} frames", flush=True)
    (output / "index.json").write_text(json.dumps(batches, indent=2) + "\n")


def add_hevc(directory: Path, crf: int = 28) -> None:
    """Append real bicubic + libx265 HEVC pairs to an existing prepared pairs directory."""
    index = json.loads((directory / "index.json").read_text())
    total = sum(len(b["samples"]) for b in index)
    done = 0
    with tempfile.TemporaryDirectory(dir=directory) as temporary:
        for group in index:
            stem = group["stem"]
            out_path = directory / f"{stem}-hevc.npy"
            if not out_path.exists():
                hr = np.load(directory / f"{stem}-hr.npy")
                np.save(out_path, hevc_bicubic_luma(hr, crf, Path(temporary)))
            done += len(group["samples"])
            print(f"HEVC {done}/{total} frames ({stem})", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("manifests", nargs="*", type=Path)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--add-hevc", action="store_true", help="Add bicubic-HEVC arrays to existing pairs dir")
    parser.add_argument("--hevc-crf", type=int, default=28)
    parser.add_argument("--no-hevc", action="store_true")
    args = parser.parse_args()
    if args.add_hevc:
        add_hevc(args.output, crf=args.hevc_crf)
    else:
        if not args.manifests:
            parser.error("manifests are required unless --add-hevc is set")
        prepare(args.manifests, args.output, args.limit, with_hevc=not args.no_hevc)


if __name__ == "__main__":
    main()
