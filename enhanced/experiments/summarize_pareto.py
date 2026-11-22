# SPDX-FileCopyrightText: 2026 perennialtech contributors
# SPDX-License-Identifier: MIT

"""Summarize paired reconstruction and repeated renderer timings."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import mean, median

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

REGIMES = {
    "area": "Clean area downscale", "blur-area": "Blur plus area downscale",
    "h264-qp32": "H.264 compression", "bicubic-hevc": "Bicubic plus HEVC",
}
UPSTREAM = {"upstream-f16", "upstream-dn", "upstream-ds", "upstream-f32", "upstream-f32-dn", "upstream-f32-ds", "spline36"}
ANCHOR = "upstream-f16"
LABELS = {
    "upstream-f16": "Stock F16", "upstream-dn": "Stock F16 DN",
    "upstream-ds": "Stock F16 DS", "upstream-f32": "Stock F32",
    "upstream-f32-dn": "Stock F32 DN", "upstream-f32-ds": "Stock F32 DS",
    "spline36": "Spline36", "f8-fast": "Fast 8", "f12-fast": "Fast 12",
    "g16-fast": "Grouped 16", "g32-fast": "Grouped 32",
    "f16-fast": "F16 fused FMA", "f16-exact": "F16 exact",
    "f16-restore": "Restore 16", "f16-clean": "Clean 16", "f16-balanced": "Balanced 16",
    "f32-fast": "F32 fused FMA", "f32-exact": "F32 exact",
    "f32-restore": "Restore 32", "f32-clean": "Clean 32", "c6f16-restore": "Restore C6F16",
    "f16-dn-fast": "F16 DN fused FMA", "f16-ds-fast": "F16 DS fused FMA",
    "f32-dn-fast": "F32 DN fused FMA", "f32-ds-fast": "F32 DS fused FMA",
    "f16-ds-exact": "F16 DS exact", "f32-ds-exact": "F32 DS exact",
}


def read_jsonl(file):
    return [json.loads(line) for line in file.read_text().splitlines()]


def verified_rows(quality, speed):
    quality_rows = [r for path in sorted(quality.glob("*/per-image.jsonl")) for r in read_jsonl(path)]
    grouped = defaultdict(dict)
    for row in quality_rows:
        key = row["id"], row["regime"]
        if row["model"] in grouped[key]:
            raise ValueError(f"Duplicate quality result for {key} and {row['model']}")
        grouped[key][row["model"]] = row
    speed_rows = read_jsonl(speed / "speed.jsonl")
    timings = defaultdict(dict)
    for row in speed_rows:
        if (row["width"], row["height"]) != (1920, 1080):
            continue
        if row["repeat"] in timings[row["model"]] or row["fps"] <= 0:
            raise ValueError("Duplicate or invalid timing")
        timings[row["model"]][row["repeat"]] = row["fps"]
    names = set(timings)
    cases = set(grouped)
    regimes = {r for _, r in grouped}
    ids = {i for i, _ in grouped}
    if cases != {(i, r) for i in ids for r in regimes}:
        raise ValueError("Incomplete frame by regime matrix")
    output = {}
    for key, models in grouped.items():
        if models.keys() != names:
            raise ValueError(f"Incomplete frame by model matrix at {key}: {models.keys() ^ names}")
    repeats = {len(v) for v in timings.values()}
    if len(repeats) != 1 or min(repeats) < 3:
        raise ValueError("Expected at least three repeats for every model")
    for name, runs in sorted(timings.items()):
        centers = list(runs.values())
        anchor_ratios = [fps / timings[ANCHOR][r] for r, fps in runs.items()]
        output[name] = {
            "median_fps": median(centers), "min_fps": min(centers), "max_fps": max(centers),
            "median_paired_speedup_pct": (median(anchor_ratios) - 1) * 100, "repeats": len(centers),
            "regimes": {},
        }
        for regime in REGIMES:
            subset = [values for (_, key), values in grouped.items() if key == regime]
            if not subset:
                continue
            own = [values[name] for values in subset]
            delta = [values[name]["psnr"] - values[ANCHOR]["psnr"] for values in subset]
            by_group = defaultdict(list)
            for row, diff in zip(own, delta):
                by_group[row["group"]].append(diff)
            output[name]["regimes"][regime] = {
                "mean_psnr": mean(r["psnr"] for r in own), "mean_ssim": mean(r["ssim"] for r in own),
                "mean_full_psnr": mean(r["full_psnr"] for r in own),
                "mean_edge_mae": mean(r["edge_mae"] for r in own if r["edge_mae"] is not None),
                "mean_delta_db": mean(delta), "paired_wins": sum(d > 0 for d in delta), "count": len(own),
                "group_delta_db": {g: mean(ds) for g, ds in sorted(by_group.items())},
            }
    return output, len(ids)


def frontier(rows, regime, candidates):
    pairs = {key: (rows[key]["median_fps"], rows[key]["regimes"][regime]["mean_psnr"]) for key in candidates}
    return sorted((key for key, (x, y) in pairs.items() if
                   not any(ox >= x and oy >= y and (ox > x or oy > y) for ox, oy in pairs.values())),
                  key=lambda key: pairs[key][0])


def make_plot(stats, subset, count, path):
    layouts = list(REGIMES.items())
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.0))
    highlights = {
        "area": {"f8-fast", "g16-fast", "f16-clean", "f32-clean", "f32-fast"},
        "blur-area": {"f8-fast", "f12-fast", "g16-fast", "f16-restore", "c6f16-restore", "f32-restore"},
        "h264-qp32": {"f8-fast", "g16-fast", "f16-restore", "c6f16-restore", "f32-restore"},
        "bicubic-hevc": {"f8-fast", "g16-fast", "f16-restore", "c6f16-restore", "f32-restore", "f32-ds-fast"},
    }
    for ax, (regime, title) in zip(axes.ravel(), layouts):
        for key, row in stats.items():
            x, y = row["median_fps"], row["regimes"][regime]["mean_psnr"]
            upstream = key in UPSTREAM
            color = "#366183" if upstream else "#b45325"
            ax.scatter(x, y, marker="o" if upstream else "D", c=color,
                       edgecolors="white", linewidths=.65, s=48, zorder=3)
            if key in highlights[regime]:
                ax.annotate(LABELS.get(key, key), (x, y), xytext=(-4, 7), ha="right",
                            textcoords="offset points", fontsize=8)
        for keys, color, ls in (
            (subset[regime]["upstream"], "#366183", "--"),
            (subset[regime]["combined"], "#b45325", "-"),
        ):
            ax.plot([stats[k]["median_fps"] for k in keys],
                    [stats[k]["regimes"][regime]["mean_psnr"] for k in keys], ls=ls, color=color, alpha=.85)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xscale("log")
        ax.set_xticks([40, 60, 100, 160, 250, 400, 600], ["40","60","100","160","250","400","600"])
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlim(33, 760)
        ax.margins(y=0.10)
        ax.grid(alpha=.20)
        ax.spines[["right","top"]].set_visible(False)
    fig.suptitle(f"ArtCNN quality–speed frontier · {count} held-out frames", x=.055, y=.985, ha="left", fontsize=17, weight="bold")
    fig.text(.056, .935, "Blue shows upstream and Spline36. Orange shows the expanded frontier.", fontsize=10)
    fig.supxlabel("RTX 3080 1080p → 2160p throughput (FPS, log scale)", y=.02)
    fig.supylabel("Mean luma PSNR (dB)", x=.012)
    fig.subplots_adjust(bottom=.105, top=.865, wspace=.26, hspace=.40)
    fig.savefig(path, dpi=155, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("quality", type=Path)
    parser.add_argument("speed", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    stats, count = verified_rows(args.quality, args.speed)
    frontiers = {}
    for key in REGIMES:
        if key in stats[ANCHOR]["regimes"]:
            frontiers[key] = {
                "upstream": frontier(stats, key, UPSTREAM & stats.keys()),
                "combined": frontier(stats, key, stats),
            }
    args.output.mkdir(exist_ok=True, parents=True)
    data = {"frames": count, "models": stats, "frontiers": frontiers}
    (args.output / "frontier.json").write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    if len(frontiers) == len(REGIMES):
        make_plot(stats, frontiers, count, args.output / "frontier.png")
    for key, val in frontiers.items():
        print(key, val)


if __name__ == "__main__":
    main()
