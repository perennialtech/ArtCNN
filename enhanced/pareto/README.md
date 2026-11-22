# GPU benchmark artifacts and training checkpoints

The released GLSL shaders now live in [GLSL/Optimized](../../GLSL/Optimized/README.md). Update playback profiles that point here to use the same filename in that directory.

The [manifest](models.json) records shader paths and SHA-256 hashes. Its `file` and `checkpoint` fields resolve relative to this directory. The `source` fields describe provenance from the original benchmark workspace.

The [GPU benchmark](../PARETO.md) documents the [frontier measurements](frontier.json), [720p timings](timing-720.json), and how to regenerate shaders from the checkpoints here.
