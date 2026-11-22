"""Benchmark ArtCNN mpv shaders through FFmpeg's Vulkan libplacebo filter."""
from collections import defaultdict
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import random
import re
import subprocess
import time

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, maximum_filter, minimum_filter, sobel
from skimage.metrics import structural_similarity

MODELS = {
    "bilinear": None,
    "spline36": None,
    "upstream-f16": "ArtCNN_C4F16.glsl",
    "upstream-dn": "ArtCNN_C4F16_DN.glsl",
    "upstream-ds": "ArtCNN_C4F16_DS.glsl",
    "upstream-f32": "ArtCNN_C4F32.glsl",
    "fork-baseline": "ArtCNN_C4F16_Baseline.glsl",
    "fork-zero": "ArtCNN_C4F16_ZeroCost_Fused.glsl",
    "fork-enhanced": "ArtCNN_C4F16_Enhanced_Fused.glsl",
}
REGIMES = ("area", "blur-area", "h264-qp32")
STRESS = "bicubic-hevc"
WARMUP = 12
MATRIX = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
COLOR = [
    "-colorspace", "bt709", "-color_trc", "bt709", "-color_primaries", "bt709", "-color_range", "tv",
]
FFMPEG = ["ffmpeg", "-nostdin", "-hide_banner", "-xerror"]


def raw_input(path, width, height, rate):
    return [
        "-f", "rawvideo", "-pixel_format", "yuv420p", "-video_size", f"{width}x{height}",
        "-framerate", str(rate), *COLOR, "-i", str(path),
    ]


def render_setup(model, width, height, pixel_format, download):
    shader = MODELS[model]
    sampling = "bilinear" if model == "bilinear" else "spline36"
    options = [
        f"w={width}", f"h={height}", f"format={pixel_format}", "colorspace=bt709",
        "color_primaries=bt709", "color_trc=bt709", "range=limited", "sigmoid=false",
        "disable_linear=true", "dithering=none", f"upscaler={sampling}",
    ]
    if shader:
        options.append(f"custom_shader_path=shaders/{shader}")
    if os.environ.get("ARTCNN_INTERNAL_VULKAN") == "1":
        if not download:
            raise ValueError("GPU-resident timing requires an FFmpeg Vulkan device")
        return ["-filter_threads", "1"], "libplacebo=" + ":".join(options)
    graph = "hwupload,libplacebo=" + ":".join(options)
    if download:
        graph += f",hwdownload,format={pixel_format}"
    init = ["-init_hw_device", f"vulkan=vk:{os.environ.get('ARTCNN_VULKAN_DEVICE', 'NVIDIA')}", "-filter_hw_device", "vk", "-filter_threads", "1"]
    return init, graph


def validate_log(path, model):
    log = path.read_text(errors="replace")
    device = os.environ.get("ARTCNN_VULKAN_DEVICE", "NVIDIA")
    if not any(device in line and ("Using device:" in line or "Device Name:" in line) for line in log.splitlines()):
        raise RuntimeError(f"Expected Vulkan device {device!r} in {path}")
    if re.search(r"(shader.*(?:error|fail)|(?:error|fail).*shader|disabling.*hook)", log, re.I):
        raise RuntimeError(f"Shader error in {path}")
    if MODELS[model]:
        specs = Path("shaders", MODELS[model]).read_text().splitlines()
        for desc in (line.removeprefix("//!DESC ") for line in specs if line.startswith("//!DESC ")):
            if f"Registering hook pass: {desc}" not in log:
                raise RuntimeError(f"Missing hook {desc!r} in {path}")


@contextmanager
def frame_stream(command, log_path):
    with log_path.open("wb") as log, subprocess.Popen(command, stdout=subprocess.PIPE, stderr=log) as process:
        try:
            yield process.stdout
            if process.stdout.read(1) or process.wait(timeout=10):
                raise RuntimeError(f"Unexpected FFmpeg output: {log_path}")
        finally:
            if process.poll() is None:
                process.kill()


def read_rgb(sample):
    path = Path("frames", sample["file"])
    if hashlib.sha256(path.read_bytes()).hexdigest() != sample["sha256"]:
        raise ValueError(f"Input hash mismatch for {path}")
    with Image.open(path) as image:
        width, height = image.size
        image = image.convert("RGB")
        image = image.crop((0, 0, width // 4 * 4, height // 4 * 4))
        return np.asarray(image, dtype=np.float32) / 255


def down2(image):
    height, width = image.shape[:2]
    return image.reshape(height // 2, 2, width // 2, 2, *image.shape[2:]).mean(axis=(1, 3))


def yuv420(y):
    y_codes = np.rint(16 + 219 * y).clip(16, 235).astype("u1")
    return y_codes.tobytes() + bytes([128]) * (y_codes.size // 2)


def encode_video(raw, compressed, width, height, count, hevc=False):
    format_name = "hevc" if hevc else "h264"
    encoded = raw.with_suffix(f".{format_name}")
    if hevc:
        encoder = ["-c:v", "libx265", "-preset", "fast", "-g", "1",
                   "-x265-params", "qp=32:keyint=1:pools=none:frame-threads=2:log-level=error"]
    else:
        encoder = ["-c:v", "libx264", "-g", "1", "-qp", "32", "-x264-params", "aq-mode=0", "-threads", "2"]
    subprocess.run([
        *FFMPEG, "-loglevel", "error", "-y", *raw_input(raw, width, height, 24),
        "-frames:v", str(count), *encoder, "-f", format_name, str(encoded),
    ], check=True)
    subprocess.run([
        *FFMPEG, "-loglevel", "error", "-y", "-threads", "2", "-i", str(encoded),
        "-frames:v", str(count), "-pix_fmt", "yuv420p", "-f", "rawvideo", str(compressed),
    ], check=True)
    if compressed.stat().st_size != count * width * height * 3 // 2:
        raise RuntimeError(f"Unexpected decoded size: {compressed}")
    encoded.unlink()


def reference_metrics(ref):
    dx = sobel(ref, axis=1) / 8
    dy = sobel(ref, axis=0) / 8
    edge = np.hypot(dx, dy) > 0.04
    flat = maximum_filter(ref, size=7) - minimum_filter(ref, size=7) < 0.025
    return ref, dx, dy, edge, flat


def score(ref_data, pred):
    ref, dx, dy, edge, flat = ref_data
    interior = np.s_[16:-16, 16:-16]
    error = pred - ref
    e = error[interior]
    edge, flat = edge[interior], flat[interior]
    ref_inner, pred_inner = ref[interior], pred[interior]
    grad_err = np.hypot(sobel(pred, axis=1) / 8 - dx, sobel(pred, axis=0) / 8 - dy)[interior]
    return {
        "psnr": float(-10 * np.log10(max(float(np.mean(e**2)), 1e-16))),
        "full_psnr": float(-10 * np.log10(max(float(np.mean(error**2)), 1e-16))),
        "ssim": float(structural_similarity(ref_inner, pred_inner, data_range=1.0, gaussian_weights=True,
                                           sigma=1.5, use_sample_covariance=False)),
        "edge_mae": float(np.mean(grad_err[edge])) if edge.any() else None,
        "flat_mae": float(np.mean(np.abs(e[flat]))) if flat.any() else None,
        "bias": float(np.mean(e)),
        "black_fraction": float(np.mean(pred < 1 / 255)),
    }


def write_gray(y, path):
    Image.fromarray(np.rint(255 * y).clip(0, 255).astype("u1")).save(path)


def save_controls(ref, lr, regime, sample, visual):
    directory = visual / sample["id"]
    directory.mkdir(parents=True, exist_ok=True)
    write_gray(ref, directory / "ref.png")
    write_gray(lr, directory / f"{regime}-input.png")


def check_bilinear(input_frame, pred):
    """Use CPU interpolation to detect reordered frames or color-range mistakes."""
    lr_height, lr_width = (dimension // 2 for dimension in pred.shape)
    y = np.frombuffer(input_frame, dtype="u1", count=lr_height * lr_width).reshape(lr_height, lr_width)
    y = (y.astype(np.float32) - 16) / 219
    expected = np.asarray(Image.fromarray(y).resize(pred.shape[::-1], Image.Resampling.BILINEAR))
    error = np.abs(expected[4:-4, 4:-4] - pred[4:-4, 4:-4])
    if float(error.mean()) > 0.001 or float(np.quantile(error, 0.99)) > 0.003:
        raise RuntimeError(f"Bilinear control disagrees with CPU: mean={error.mean():.5f}, p99={np.quantile(error, 0.99):.5f}")


def run_quality_batch(samples, output, label, model_names, identical, stress):
    work = output / label
    work.mkdir(exist_ok=True)
    refs = [read_rgb(sample) @ MATRIX for sample in samples]
    height, width = refs[0].shape
    if any(ref.shape != (height, width) for ref in refs):
        raise ValueError(f"Mixed dimensions for {label}")
    lw, lh = width // 2, height // 2
    sample_bytes = lw * lh * 3 // 2
    regimes = (*REGIMES, STRESS) if stress else REGIMES
    paths = {regime: work / f"{regime}.yuv" for regime in regimes}
    for regime in ("area", "blur-area"):
        with paths[regime].open("wb") as fp:
            for ref in refs:
                filtered = ref if regime == "area" else gaussian_filter(ref, sigma=0.8, mode="reflect")
                fp.write(yuv420(down2(filtered)))
    encode_video(paths["area"], paths["h264-qp32"], lw, lh, len(samples))
    if stress:
        bicubic = work / "bicubic.yuv"
        with bicubic.open("wb") as fp:
            for ref in refs:
                scaled = Image.fromarray(ref).resize((lw, lh), Image.Resampling.BICUBIC)
                fp.write(yuv420(np.asarray(scaled)))
        encode_video(bicubic, paths[STRESS], lw, lh, len(samples), hevc=True)
        bicubic.unlink()

    batch = work / "render-input.yuv"
    input_hashes = {}
    input_change = {}
    gallery = {group: vals[len(vals) // 2]["id"] for group, vals in groupby(samples, "group").items()}
    clean = paths["area"].read_bytes()
    with batch.open("wb") as combined:
        combined.write(clean[:sample_bytes] * WARMUP)
        for regime, path in paths.items():
            raw = path.read_bytes()
            input_hashes[regime] = hashlib.sha256(raw).hexdigest()
            per_image = []
            for i, sample in enumerate(samples):
                offset = i * sample_bytes
                close = offset + lw * lh
                y = np.frombuffer(raw[offset:close], dtype="u1").reshape(lh, lw).astype(np.float32)
                y_clean = np.frombuffer(clean[offset:close], dtype="u1").reshape(lh, lw).astype(np.float32)
                per_image.append(float(np.mean(np.abs(y - y_clean))))
                if sample["id"] == gallery[sample["group"]]:
                    save_controls(refs[i], (y - 16) / 219, regime, sample, output / "gallery")
            input_change[regime] = per_image
            combined.write(raw)

    ref_data = [reference_metrics(ref) for ref in refs]
    frame_bytes = 3 * width * height * 2
    output_hashes = {}
    stats = work / "per-image.jsonl"
    with stats.open("w") as summary:
        for model in model_names:
            init, graph = render_setup(model, width, height, "yuv444p16le", download=True)
            log_path = work / f"{model}.log"
            cmd = [
                *FFMPEG, "-loglevel", "verbose", *init, *raw_input(batch, lw, lh, 24),
                "-vf", graph, "-pix_fmt", "yuv444p16le", "-threads", "1", "-f", "rawvideo", "pipe:1",
            ]
            digest = hashlib.sha256()
            start = time.monotonic()
            with frame_stream(cmd, log_path) as stream:
                for warmup in range(WARMUP):
                    raw = stream.read(frame_bytes)
                    if len(raw) != frame_bytes:
                        raise RuntimeError(f"Incomplete warm-up frame {warmup}: {log_path}")
                warm_y = raw[:width * height * 2]
                for regime in regimes:
                    control = paths[regime].read_bytes() if model == "bilinear" else None
                    for i, sample in enumerate(samples):
                        raw = stream.read(frame_bytes)
                        if len(raw) != frame_bytes:
                            raise RuntimeError(f"Incomplete frame {i}, {regime}, {model}. See {log_path}")
                        pixels = np.frombuffer(raw, dtype="<u2", count=width * height).reshape(height, width)
                        digest.update(memoryview(raw)[:width * height * 2])
                        if i == 0 and regime == "area" and model != "fork-baseline":
                            if warm_y != raw[:width * height * 2]:
                                raise RuntimeError(f"First scored frame disagrees with repeated warm-up input: {log_path}")
                        pred = (pixels.astype(np.float32) - 4096) / 56064
                        if control is not None:
                            check_bilinear(control[i * sample_bytes:(i + 1) * sample_bytes], pred)
                        row = {
                            "dataset": sample["dataset"], "group": sample["group"], "id": sample["id"],
                            "model": model, "regime": regime, "width": width, "height": height,
                            "lr_delta_codes": input_change[regime][i], **score(ref_data[i], pred),
                        }
                        summary.write(json.dumps(row, allow_nan=False) + "\n")
                        if sample["id"] == gallery[sample["group"]]:
                            write_gray(pred, output / "gallery" / sample["id"] / f"{regime}-{model}.png")
            validate_log(log_path, model)
            output_hashes[model] = digest.hexdigest()
            print(f"{label}: {model} rendered and scored {len(samples) * len(regimes)} frames in {time.monotonic()-start:.1f}s")
    expected = set()
    for name, anchor in identical.items():
        if output_hashes[name] != output_hashes[anchor]:
            raise RuntimeError(f"{name} differs from {anchor} on {label}")
        expected.add(frozenset((name, anchor)))
    seen = set()
    for i, (name, digest) in enumerate(output_hashes.items()):
        for other, other_digest in list(output_hashes.items())[:i]:
            if digest == other_digest:
                seen.add(frozenset((name, other)))
    if seen != expected:
        raise RuntimeError(f"Unexpected identical luma streams in {label}: {seen ^ expected}")
    (work / "hashes.json").write_text(json.dumps({"input": input_hashes, "output": output_hashes}, indent=2) + "\n")
    for path in [*paths.values(), batch]:
        path.unlink()


def groupby(rows, field):
    groups = defaultdict(list)
    for row in rows:
        groups[row[field]].append(row)
    return groups


def quality(paths, output, model_names, identical, stress=False):
    for path in paths:
        samples = json.loads(path.read_text())
        batches = defaultdict(list)
        for sample in samples:
            width, height = (d // 4 * 4 for d in sample["size"])
            batches[(width, height)].append(sample)
        for (width, height), batch in sorted(batches.items()):
            run_quality_batch(batch, output, f"{path.stem}-{width}x{height}", model_names, identical, stress)


def timed_render(model, raw, width, height, frames, log_path):
    init, graph = render_setup(model, width * 2, height * 2, "yuv420p", download=False)
    cmd = [
        *FFMPEG, "-loglevel", "verbose", "-nostats", "-stats_period", "0.10",
        "-progress", "pipe:1", *init, "-stream_loop", "-1", *raw_input(raw, width, height, 24),
        "-frames:v", str(frames), "-vf", graph, "-f", "null", "-",
    ]
    points = []
    with log_path.open("wb") as log, subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=log,
                                                      text=True, bufsize=1) as ff:
        for line in ff.stdout:
            key, _, value = line.strip().partition("=")
            if key == "frame":
                points.append((int(value), time.monotonic()))
        if ff.wait(timeout=20):
            raise RuntimeError(f"Timing failed: {log_path}")
    validate_log(log_path, model)
    if not points or points[-1][0] != frames:
        raise RuntimeError(f"Frame count mismatch: {log_path}")
    if not re.search(r"Video: wrapped_avframe, [^\n]*\bvulkan\(", log_path.read_text()):
        raise RuntimeError(f"Expected GPU-resident output: {log_path}")
    warm = [(frame, stamp) for frame, stamp in points if 30 <= frame <= frames - 10]
    if len(warm) < 3:
        return None
    indexes, times = map(np.array, zip(*warm))
    seconds = times - times[0]
    spf, offset = np.polyfit(indexes, seconds, 1)
    max_resid = float(np.max(np.abs(seconds - (spf * indexes + offset))))
    return {"fps": float(1 / spf), "ms": float(spf * 1000), "points": len(warm),
            "last_frame": points[-1][0], "max_regression_residual_s": max_resid,
            "trace": [[int(f), float(t - points[0][1])] for f, t in points]}


def speed(manifest, output, model_names, repeats, sizes, frames):
    samples = json.loads(manifest.read_text())
    samples = samples[2:10] or samples
    if not samples:
        raise ValueError(f"No timing frames in {manifest}")
    with (output / "speed.jsonl").open("w") as summary:
        for width, height in sizes:
            raw = output / "speed-input.yuv"
            with raw.open("wb") as fp:
                for sample in samples:
                    rgb = read_rgb(sample)
                    luma = rgb @ MATRIX
                    if luma.shape != (height, width):
                        luma = np.asarray(Image.fromarray(luma).resize((width, height), Image.Resampling.LANCZOS))
                    fp.write(yuv420(luma))
            models = list(model_names)
            for repeat in range(repeats):
                random.Random(20261122 + repeat).shuffle(models)
                for model in models:
                    stem = f"speed-{width}x{height}-{model}-{repeat+1}"
                    result = timed_render(model, raw, width, height, frames, output / f"{stem}.log")
                    if result is None:
                        result = timed_render(model, raw, width, height, frames * 4, output / f"{stem}-retry.log")
                    if result is None:
                        raise RuntimeError(f"Insufficient timing samples: {stem}")
                    row = {"model": model, "width": width, "height": height, "repeat": repeat + 1, **result}
                    summary.write(json.dumps(row, allow_nan=False) + "\n")
                    summary.flush()
                    print(f"{stem}: {result['fps']:.1f} FPS")
            raw.unlink()
