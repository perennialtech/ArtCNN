# SPDX-FileCopyrightText: 2026 perennialtech contributors
# SPDX-License-Identifier: MIT

"""Score candidate shaders with the same inputs and renderer as the stock controls."""
import argparse
import json
from pathlib import Path
import benchmark_real as benchmark


def configure_models(specs):
    for entry in specs:
        name, path = entry.split("=", 1)
        if not name or not path or not name.replace("-", "").replace("_", "").isalnum():
            raise ValueError(f"Expected name=shader-path: {entry}")
        shader = Path(path).resolve(strict=True)
        dest = Path("shaders", f"bench-{name}.glsl")
        if dest.is_symlink() and dest.resolve() != shader:
            dest.unlink()
        if not dest.exists():
            dest.symlink_to(shader)
        benchmark.MODELS[name] = dest.name


def stage_manifests(paths, output, limit):
    frame_root = Path("frames")
    frame_root.mkdir(exist_ok=True)
    staged = []
    for path in paths:
        records = json.loads(path.read_text())
        if limit:
            count = min(limit, len(records))
            records = [records[(2 * i + 1) * len(records) // (2 * count)] for i in range(count)]
        for record in records:
            source = (path.parent / "frames" / record["file"]).resolve(strict=True)
            target = frame_root / record["file"]
            if target.exists() and target.resolve() != source:
                raise ValueError(f"Frame staging collision: {target}")
            if not target.exists():
                target.symlink_to(source)
        dest = output / path.name
        dest.write_text(json.dumps(records, indent=2) + "\n")
        staged.append(dest)
    return staged


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["quality", "speed", "speed-all"])
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--register", nargs="*", default=[])
    parser.add_argument("--identical", nargs="*", default=[], metavar="MODEL=ANCHOR",
                        help="Require bit-identical luma from these model pairs")
    parser.add_argument("--stress", action="store_true", help="Add bicubic downscale and HEVC compression")
    parser.add_argument("--manifests", nargs="+", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    Path("shaders").mkdir(exist_ok=True)
    configure_models(args.register)
    args.output.mkdir(parents=True, exist_ok=True)
    for model in args.models:
        if model not in benchmark.MODELS:
            parser.error(f"Unknown model {model!r}")
        filename = benchmark.MODELS[model]
        if filename and not (Path("shaders") / filename).is_file():
            parser.error(f"Shader for {model!r} is missing. Register it with --register.")
    if args.mode == "quality":
        staged = stage_manifests(args.manifests, args.output, args.limit)
        identical = dict(pair.split("=", 1) for pair in args.identical)
        if (set(identical) | set(identical.values())) - set(args.models):
            parser.error("Both sides of --identical must appear in --models")
        benchmark.quality(staged, args.output, args.models, identical, args.stress)
    else:
        staged = stage_manifests(args.manifests[:1], args.output, None)
        sizes = [(960, 540), (1280, 720), (1920, 1080)] if args.mode == "speed-all" else [(1920, 1080)]
        benchmark.speed(staged[0], args.output, args.models, args.repeats, sizes, 388)


if __name__ == "__main__":
    main()
