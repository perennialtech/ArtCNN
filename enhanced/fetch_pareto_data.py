# SPDX-FileCopyrightText: 2026 perennialtech contributors
# SPDX-License-Identifier: MIT

"""Fetch pinned, deterministic ATD-12K training and held-out frame subsets."""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import time
from urllib.request import Request, urlopen

from PIL import Image

MIRROR = "ProgramYourFace/ATD-12k"
REVISION = "bc613323fa3721a36ffec39886a04eb6fff6b6f2"
ROOT = f"https://huggingface.co/datasets/{MIRROR}/resolve/{REVISION}"
FILMS = {
    "morevna": (
        "https://upload.wikimedia.org/wikipedia/commons/a/a7/Morevna_Episode_3.0.2_%28English%29.webm",
        range(45, 586, 30), 138495285,
    ),
    "sol-levante": (
        "https://upload.wikimedia.org/wikipedia/commons/transcoded/4/44/Sol_Levante.webm/Sol_Levante.webm.1080p.vp9.webm",
        range(15, 206, 10), 90812847,
    ),
}


def fetch(url: str, size: int | None = None) -> bytes:
    headers = {"User-Agent": "ArtCNNBench/1.0 (https://github.com/perennialtech/ArtCNN)"}
    if size is not None:
        headers["Range"] = f"bytes=0-{size - 1}"
    request = Request(url, headers=headers)
    for attempt in range(4):
        try:
            with urlopen(request, timeout=30) as response:
                payload = response.read()
            if size is not None and len(payload) != size:
                raise OSError(f"Unexpected response length for {url}")
            return payload
        except OSError:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)
    raise AssertionError("Unreachable")


def even_sample(paths: list[str], count: int) -> list[str]:
    count = min(len(paths), count)
    order = sorted(paths)
    return [order[(2 * i + 1) * len(order) // (2 * count)] for i in range(count)]


def select(paths: list[str], split: str) -> list[str]:
    groups = defaultdict(list)
    for path in paths:
        pieces = path.split("/")
        if len(pieces) != 4:
            continue
        frame, name, subset = pieces[-1], pieces[-2], pieces[-3]
        group = name.rsplit("_", 2)[0]
        if split == "train" and subset == "train_10k" and frame == "frame2.jpg" and group.startswith("Japan_"):
            groups[group].append(path)
        elif split == "holdout" and (subset, frame, group) == ("train_10k", "frame2.jpg", "Hayao_0"):
            groups[group].append(path)
        elif split in {"dev", "holdout"} and subset == "test_2k_540p" and frame == "frame2.png" and group.startswith("Japan_v2_"):
            groups[group].append(path)
    if split == "train":
        return [item for _, files in sorted(groups.items()) for item in even_sample(files, 220)]
    if split == "dev":
        return [item for _, files in sorted(groups.items()) for item in even_sample(files, 32)]
    result = []
    for group, files in sorted(groups.items()):
        if group == "Hayao_0":
            result.extend(even_sample(files, 32))
        else:
            # The development set used 32 evenly spaced frames per Japanese test group.
            # Keep neighboring triplet indices separate where the sampling permits it.
            order = sorted(files)
            dev = {(2 * i + 1) * len(order) // 64 for i in range(32)}
            unused = [p for i, p in enumerate(order) if all(i + offset not in dev for offset in (-1, 0, 1))]
            result.extend(even_sample(unused, 16))
    return result


def download(path: str, frames: Path, split: str) -> dict:
    name = path.split("/")[-2]
    target = frames / (name + Path(path).suffix)
    url = f"{ROOT}/{path}"
    if not target.exists():
        payload = fetch(url)
        temporary = target.with_suffix(".part")
        temporary.write_bytes(payload)
        with Image.open(temporary) as image:
            image.verify()
        temporary.replace(target)
    with Image.open(target) as image:
        size = list(image.size)
    return {"id": name, "dataset": split, "group": name.rsplit("_", 2)[0], "size": size, "url": url,
            "sha256": hashlib.sha256(target.read_bytes()).hexdigest(), "file": target.name}


def film_frames(name: str, frames: Path) -> list[dict]:
    url, timestamps, length = FILMS[name]
    movie = frames.parent / f"{name}.webm"
    if not movie.exists():
        temporary = movie.with_suffix(".part")
        temporary.write_bytes(fetch(url, length))
        temporary.replace(movie)
    if movie.stat().st_size != length:
        raise ValueError(f"Unexpected movie length: {movie}")
    with movie.open("rb") as stream:
        source_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    result = []
    for timestamp in timestamps:
        sample = f"{name}-{timestamp:03}"
        image = frames / f"{sample}.png"
        if not image.exists():
            subprocess.run([
                "ffmpeg", "-nostdin", "-v", "error", "-ss", str(timestamp), "-i", str(movie), "-map", "0:v:0",
                "-frames:v", "1", "-threads", "1", "-update", "1", str(image),
            ], check=True)
        with Image.open(image) as data:
            size = list(data.size)
        result.append({
            "id": sample, "dataset": name, "group": name, "file": image.name, "size": size, "url": url,
            "timestamp_s": timestamp, "video_sha256": source_hash, "sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
        })
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("split", choices=["train", "dev", "holdout", *FILMS])
    parser.add_argument("output", type=Path)
    parser.add_argument("--jobs", type=int, default=12)
    args = parser.parse_args()
    frames = args.output / "frames"
    frames.mkdir(parents=True, exist_ok=True)
    if args.split in FILMS:
        samples = film_frames(args.split, frames)
    else:
        index = json.loads(fetch(f"https://huggingface.co/api/datasets/{MIRROR}/revision/{REVISION}"))
        if index["sha"] != REVISION:
            raise ValueError("Dataset revision mismatch")
        subset = select([entry["rfilename"] for entry in index["siblings"]], args.split)
        if not subset:
            raise ValueError(f"No inputs found for {args.split}")
        print(f"Fetching {len(subset)} {args.split} frames from ATD-12K", flush=True)
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            samples = []
            for record in pool.map(lambda p: download(p, frames, args.split), subset):
                samples.append(record)
                if len(samples) % 100 == 0:
                    print(f"Fetched {len(samples)}/{len(subset)}", flush=True)
    if len({r["id"] for r in samples}) != len(samples):
        raise ValueError("Duplicate frame IDs")
    manifest = args.output / f"{args.split}-manifest.json"
    manifest.write_text(json.dumps(samples, indent=2) + "\n")
    print(f"Wrote {len(samples)} records to {manifest}", flush=True)


if __name__ == "__main__":
    main()
