# SPDX-FileCopyrightText: 2026 perennialtech contributors
# SPDX-License-Identifier: MIT

"""Repeated warm CPU timings with real, SHA-verified, held-out animation frames."""
import argparse
import hashlib
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import random
from statistics import median
import sys
from time import perf_counter_ns

import numpy as np
from PIL import Image

def frames(manifest, dims, count):
    records = json.loads(manifest.read_text())
    samples = [records[(2 * i + 1) * len(records) // (2 * count)] for i in range(count)]
    images = []
    for sample in samples:
        path = manifest.parent / "frames" / sample["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != sample["sha256"]:
            raise ValueError(f"Source hash mismatch for {path}")
        with Image.open(path) as im:
            image = im.convert("L").resize(dims, Image.Resampling.LANCZOS)
            images.append(np.asarray(image, dtype=np.float32)[..., None] / 255)
    return images


def get_engine(engine_type, model, backend, upstream=None):
    if backend == "ort" and upstream is not None:
        return upstream.Engine(str(model), providers=["CPUExecutionProvider"]).run
    if backend.startswith("tile"):
        tile, workers = map(int, backend.removeprefix("tile").split("x"))
        return engine_type(model, cpu_tile=tile, cpu_threads=workers).run
    threads = int(backend.split("-")[1]) if "-" in backend else None
    provider = "openvino" if backend.startswith("ov") else "onnxruntime"
    return engine_type(model, cpu_backend=provider, cpu_threads=threads).run


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--engine-dir", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--upstream-engine-dir", type=Path, help="Use the pinned upstream Engine for the ort control")
    p.add_argument("--sizes", required=True, nargs="+")
    p.add_argument("--models", required=True, nargs="+", help="Pass path:backend,backend")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--samples", type=int, default=6)
    p.add_argument("--rounds", type=int, default=4)
    p.add_argument("--fetch", action="store_true", help="Download missing pinned input frames")
    args = p.parse_args()
    if args.fetch:
        from data import fetch_frames

        fetch_frames(args.manifest)
    sys.path.insert(0, str(args.engine_dir))
    from engine import Engine
    upstream = None
    if args.upstream_engine_dir:
        spec = spec_from_file_location("upstream_engine", args.upstream_engine_dir / "engine.py")
        upstream = module_from_spec(spec)
        spec.loader.exec_module(upstream)
    with args.output.open("w") as stream:
        for size in args.sizes:
            wh = tuple(map(int, size.split("x")))
            images = frames(args.manifest, wh, args.samples)
            cases = []
            for spec in args.models:
                name, backends = spec.rsplit(":", 1)
                cases.extend((Path(name), backend) for backend in backends.split(","))
            for trial in range(args.rounds):
                random.Random(43117 + trial).shuffle(cases)
                for model, backend in cases:
                    infer = get_engine(Engine, model, backend, upstream)
                    expected = infer(images[(trial + 1) % len(images)]).shape
                    if expected != (wh[1]*2, wh[0]*2, 1):
                        raise ValueError(f"Bad result shape for {model} under {backend}: {expected}")
                    durations, means = [], []
                    for i in range(len(images)):
                        start = perf_counter_ns()
                        output = infer(images[(trial + i) % len(images)])
                        durations.append((perf_counter_ns() - start) / 1e6)
                        means.append(float(output.mean()))
                    row = {"model": model.stem, "size": size, "backend": backend, "trial": trial,
                           "milliseconds": durations, "median_ms": median(durations), "output_means": means}
                    stream.write(json.dumps(row) + "\n")
                    stream.flush()
                    print(size, model.stem, backend, trial, round(row["median_ms"], 2), flush=True)
                    del infer


if __name__ == "__main__":
    main()
