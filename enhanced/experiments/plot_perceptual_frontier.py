# SPDX-FileCopyrightText: 2026 perennialtech contributors
# SPDX-License-Identifier: MIT

"""Plot unified perceptual quality-speed Pareto frontier (FPS vs. AlexNet LPIPS & DISTS)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

REGIMES = [
    ("area", "Clean area downscale"),
    ("blur-area", "Blur plus area downscale"),
    ("h264-qp32", "H.264 compression"),
    ("bicubic-hevc", "Bicubic plus HEVC"),
]

# Map unified perceptual model names -> frontier.json speed keys
FPS_KEY = {
    "C3F8_Fast": "f8-fast",
    "C3F12_Fast": "f12-fast",
    "C4F16_Grouped": "g16-fast",
    "C4F16_Clean": "f16-clean",
    "C4F16_Balanced": "f16-balanced",
    "C4F16_Restore": "f16-restore",
    "C6F16_Restore": "c6f16-restore",
    "C4F32_Grouped": "g32-fast",
    "C4F32_Clean": "f32-clean",
    "C4F32_Restore": "f32-restore",
    "Upstream_C4F16": "upstream-f16",
    "Upstream_C4F16_DS": "upstream-ds",
    "Upstream_C4F32": "upstream-f32",
    "Upstream_C4F32_DS": "upstream-f32-ds",
}

LABELS = {
    "C3F8_Fast": "Fast 8",
    "C3F12_Fast": "Fast 12",
    "C4F16_Grouped": "Grouped 16",
    "C4F16_Clean": "Clean 16",
    "C4F16_Balanced": "Balanced 16",
    "C4F16_Restore": "Restore 16",
    "C6F16_Restore": "Restore C6F16",
    "C4F32_Grouped": "Grouped 32",
    "C4F32_Clean": "Clean 32",
    "C4F32_Restore": "Restore 32",
    "Upstream_C4F16": "Stock F16",
    "Upstream_C4F16_DS": "Stock F16 DS",
    "Upstream_C4F32": "Stock F32",
    "Upstream_C4F32_DS": "Stock F32 DS",
}


def pareto_lower_is_better(points: dict[str, tuple[float, float]]) -> list[str]:
    """Return non-dominated keys where higher FPS (x) and lower distance (y) are better."""
    keep = []
    for k, (x, y) in points.items():
        if not any(ox >= x and oy <= y and (ox > x or oy < y) for ox, oy in points.values()):
            keep.append(k)
    return sorted(keep, key=lambda k: points[k][0])


def plot_metric(perceptual: dict, frontier_data: dict, metric: str, ylabel: str, out_path: Path) -> None:
    models_fps = frontier_data["models"]
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.2))
    for ax, (regime, title) in zip(axes.ravel(), REGIMES):
        pts_all = {}
        pts_up = {}
        for name, regimes in perceptual.items():
            fk = FPS_KEY.get(name)
            if not fk or fk not in models_fps or regime not in regimes:
                continue
            fps = models_fps[fk]["median_fps"]
            val = regimes[regime][metric]
            pts_all[name] = (fps, val)
            if name.startswith("Upstream_"):
                pts_up[name] = (fps, val)

        front_all = pareto_lower_is_better(pts_all)
        front_up = pareto_lower_is_better(pts_up)

        for name, (x, y) in pts_all.items():
            is_up = name.startswith("Upstream_")
            color = "#366183" if is_up else "#b45325"
            ax.scatter(x, y, marker="o" if is_up else "D", c=color, edgecolors="white", linewidths=0.65, s=50, zorder=3)
            if name in front_all or name in ("Upstream_C4F16", "Upstream_C4F16_DS", "Upstream_C4F32_DS"):
                dy = 7 if is_up else -11
                ax.annotate(LABELS.get(name, name), (x, y), xytext=(-4, dy), ha="right", textcoords="offset points", fontsize=8)

        for keys, color, ls in ((front_up, "#366183", "--"), (front_all, "#b45325", "-")):
            if keys:
                ax.plot([pts_all[k][0] for k in keys], [pts_all[k][1] for k in keys], ls=ls, color=color, alpha=0.85)

        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xscale("log")
        ax.set_xticks([40, 60, 100, 160, 250, 400, 600], ["40", "60", "100", "160", "250", "400", "600"])
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlim(33, 680)
        ax.invert_yaxis()  # Higher on chart = lower perceptual distance = better
        ax.margins(y=0.14)
        ax.grid(alpha=0.20)
        ax.spines[["right", "top"]].set_visible(False)

    fig.suptitle(f"ArtCNN perceptual frontier ({ylabel}) · 96 held-out frames", x=0.055, y=0.985, ha="left", fontsize=16, weight="bold")
    fig.text(0.056, 0.935, "Blue = upstream stock shaders. Orange = perceptual Pareto frontier (higher on chart = lower distance = better).", fontsize=9.5)
    fig.supxlabel("RTX 3080 1080p → 2160p throughput (FPS, log scale)", y=0.02)
    fig.supylabel(f"{ylabel} (lower is better; inverted axis)", x=0.012)
    fig.subplots_adjust(bottom=0.105, top=0.865, wspace=0.26, hspace=0.40)
    fig.savefig(out_path, dpi=155, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("perceptual_json", type=Path)
    parser.add_argument("frontier_json", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    perceptual = json.loads(args.perceptual_json.read_text())
    frontier_data = json.loads(args.frontier_json.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_metric(perceptual, frontier_data, "lpips", "AlexNet LPIPS", args.output_dir / "perceptual_lpips_frontier.png")
    plot_metric(perceptual, frontier_data, "dists", "DISTS (VGG16)", args.output_dir / "perceptual_dists_frontier.png")
    print("Saved perceptual frontier charts to", args.output_dir)


if __name__ == "__main__":
    main()
