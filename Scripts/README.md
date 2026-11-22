# Dataset and metric scripts

Run these scripts from a scratch directory using the repository's locked uv environment. The dataset scripts work on PNGs in the current directory. Use copies of source images because augmentation, patch extraction, and pruning delete inputs.

```sh
repo=/path/to/ArtCNN
uv run --project "$repo" --locked --group dataset python "$repo/Scripts/dataset_augment.py"
```

Use the same `dataset` group for `dataset_patchify.py`. Run `dataset_pruner.py --help` with the default environment for its size-based pruning options.

For the image metric script, use the `jax-metrics` group. Place `reference.png` and the PNGs to compare in the working directory. The script writes `benchmark_result.txt` there.

```sh
uv run --project "$repo" --locked --group jax-metrics python "$repo/Scripts/benchmarker.py"
```
