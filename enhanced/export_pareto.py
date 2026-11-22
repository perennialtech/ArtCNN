"""Regenerate a release shader from the portable training checkpoint."""
import argparse
import json
from pathlib import Path

import numpy as np

from artcnn.glsl_export import export_artcnn_glsl
from artcnn.models_np import ArtCNNC, ArtCNNCConfig
from experiments.fused_accumulation import explicit_fma
from experiments.tune_workgroups import retile_shader


def export_checkpoint(path: Path, workgroup: tuple[int, int], fma: bool) -> str:
    with np.load(path) as weights:
        model = ArtCNNC(ArtCNNCConfig(**json.loads(weights["config"].item())))
        model.load_state_dict(weights)
        grouped = {
            name: (weights[f"{name}.spatial"], weights[f"{name}.mix"])
            for name, _ in model.layers() if f"{name}.spatial" in weights
        }
    shader = retile_shader(export_artcnn_glsl(model, grouped_kernels=grouped), *workgroup)
    return explicit_fma(shader) if fma else shader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--workgroup", nargs=2, type=int, default=(8, 8))
    parser.add_argument("--fma", action="store_true")
    args = parser.parse_args()
    args.output.write_text(export_checkpoint(args.checkpoint, tuple(args.workgroup), args.fma))


if __name__ == "__main__":
    main()
