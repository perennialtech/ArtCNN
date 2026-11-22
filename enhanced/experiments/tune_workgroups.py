"""Rewrite mpv shader workgroups while preserving each thread's output footprint."""
from __future__ import annotations

import argparse
from pathlib import Path
import re

COMPUTE = re.compile(r"(?m)^//!COMPUTE (\d+) (\d+) (\d+) (\d+)$")


def retile_shader(shader: str, width: int, height: int) -> str:
    if width <= 0 or height <= 0 or width * height > 256:
        raise ValueError("Choose a positive workgroup of at most 256 threads")

    def replace(match: re.Match[str]) -> str:
        out_width, out_height, src_width, src_height = map(int, match.groups())
        if out_width % src_width or out_height % src_height:
            raise ValueError(f"Nonintegral output footprint: {match[0]}")
        return f"//!COMPUTE {width * (out_width // src_width)} {height * (out_height // src_height)} {width} {height}"

    result, count = COMPUTE.subn(replace, shader)
    if not count:
        raise ValueError("No compute passes found")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    args = parser.parse_args()
    result = retile_shader(args.source.read_text(), args.width, args.height)
    args.output.write_text(result)


if __name__ == "__main__":
    main()
