"""Try explicit vector FMA accumulation in a C-series shader."""
from __future__ import annotations

import argparse
from pathlib import Path
import re

ACCUMULATION = re.compile(r"(?m)^([ \t]*)((?:result|group)\d+) \+= M4\(([^\n]*)\) \* (inp_\w+|group\d+);$")
HELPER = """V4 accumulate_columns(V4 total, M4 weight, V4 values) {
    total = fma(weight[0], V4(values.x), total);
    total = fma(weight[1], V4(values.y), total);
    total = fma(weight[2], V4(values.z), total);
    return fma(weight[3], V4(values.w), total);
}
"""


def explicit_fma(shader: str, group_sums: bool = False) -> str:
    def replace(match):
        indent, target, matrix, value = match.groups()
        if not group_sums or not target.startswith("result") or not value.startswith("inp_"):
            return f"{indent}{target} = accumulate_columns({target}, M4({matrix}), {value});"
        _, group, x, y = value.split("_")
        subtotal = f"{target}_input{group}"
        lines = []
        if x == y == "0":
            lines.append(f"{indent}V4 {subtotal} = V4(0.0);")
        lines.append(f"{indent}{subtotal} = accumulate_columns({subtotal}, M4({matrix}), {value});")
        if x == y == "2":
            lines.append(f"{indent}{target} += {subtotal};")
        return "\n".join(lines)

    chunks = shader.split("//!DESC ")
    for index, chunk in enumerate(chunks):
        chunk, matches = ACCUMULATION.subn(replace, chunk)
        if matches:
            if chunk.count("void hook()") != 1:
                raise ValueError("Expected a single compute hook in this pass")
            chunks[index] = chunk.replace("void hook()", HELPER + "void hook()")
    result = "//!DESC ".join(chunks)
    if result == shader:
        raise ValueError("No matrix accumulations found")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--group-sums", action="store_true")
    args = parser.parse_args()
    args.output.write_text(explicit_fma(args.source.read_text(), args.group_sums))


if __name__ == "__main__":
    main()
