# SPDX-FileCopyrightText: 2026 perennialtech contributors
# SPDX-License-Identifier: MIT

"""Fuse upstream C-series pixel shuffle and optionally use explicit FMA."""
from __future__ import annotations

import argparse
from pathlib import Path
import re

from fused_accumulation import explicit_fma
from tune_workgroups import retile_shader


def replace_once(text, source, target):
    if text.count(source) != 1:
        raise ValueError(f"Expected exactly one occurrence of {source!r}")
    return text.replace(source, target, 1)


def fuse_upstream_final(shader: str) -> str:
    parts = shader.split("//!DESC ")
    if len(parts) != 9 or not parts[-1].splitlines()[0].endswith("(Depth-To-Space)"):
        raise ValueError("Expected an eight-pass upstream C-series shader")
    final = parts[-2]
    if not final.splitlines()[0].endswith("(Conv2D-6)") or "//!BIND conv2d_6\n" not in parts[-1]:
        raise ValueError("Unexpected upstream final layers")
    final = replace_once(final, "(Conv2D-6)\n", "(Conv2D-6 and pixel shuffle)\n")
    final = replace_once(final, "//!COMPUTE 12 16 12 16", "//!COMPUTE 24 32 12 16")
    final = replace_once(final, "//!SAVE conv2d_6\n", "")
    final = replace_once(final, "//!WIDTH LUMA.w 1.0 *\n//!HEIGHT LUMA.h 1.0 *",
                         "//!WIDTH LUMA.w 2.0 *\n//!HEIGHT LUMA.h 2.0 *")
    original = re.search(r"(?m)^    const ivec2 output_base = [^\n]+;\n    imageStore\(out_image, [^\n]+, result0\);$", final)
    if original is None:
        raise ValueError("Unexpected final convolution store")
    direct = ["    const ivec2 p = ivec2(gl_GlobalInvocationID.xy) * 2;"]
    for x, y, component in ((0, 0, "x"), (1, 0, "y"), (0, 1, "z"), (1, 1, "w")):
        direct.append(f"    imageStore(out_image, p + ivec2({x}, {y}), "
                      f"vec4(clamp(float(result0.{component}), 0.0, 1.0), 0.0, 0.0, 1.0));")
    shader = "//!DESC ".join([*parts[:-2], replace_once(final, original[0], "\n".join(direct))])
    return shader.rstrip() + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--workgroup", type=int, nargs=2, default=(8, 8))
    parser.add_argument("--fma", action="store_true")
    args = parser.parse_args()
    shader = retile_shader(fuse_upstream_final(args.source.read_text(encoding="utf-8")), *args.workgroup)
    if args.fma:
        shader = explicit_fma(shader)
    title = "// MIT License\n"
    if shader.startswith(title):
        identifier = "// SPDX-License-Identifier: MIT\n"
        if not shader.startswith(title + identifier):
            shader = title + identifier + shader[len(title):]
        notice = "// Copyright (c) 2026 perennialtech contributors (shader optimizations)"
        grant = "\n\n// Permission is hereby granted"
        if notice not in shader:
            shader = shader.replace(grant, "\n" + notice + grant, 1)
    args.output.write_text(shader, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
