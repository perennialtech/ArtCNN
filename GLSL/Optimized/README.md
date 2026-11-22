# Optimized GPU shaders

These shaders double luma resolution in mpv with `vo=gpu-next`. Choose one ArtCNN shader per playback profile. The shader activates when both output dimensions exceed 1.3× the source dimensions.

| Trained model | Intended use |
| --- | --- |
| [C3F8 Fast](ArtCNN_C3F8_Fast.glsl) | Lowest shader cost among the trained models |
| [C3F12 Fast](ArtCNN_C3F12_Fast.glsl) | Compact 12-channel option |
| [C4F16 Clean](ArtCNN_C4F16_Clean.glsl) | Preserve detail in clean sources |
| [C4F16 Balanced](ArtCNN_C4F16_Balanced.glsl) | A mix of clean, blurred, and compressed sources |
| [C4F16 Restore](ArtCNN_C4F16_Restore.glsl) | Restore blurred or compressed sources |
| [C6F16 Restore](ArtCNN_C6F16_Restore.glsl) | Deeper 6-layer 16-channel restoration model (~260 FPS, larger receptive field) |
| [C4F32 Clean](ArtCNN_C4F32_Clean.glsl) | Wider 32-channel clean model (~118 FPS, lowest clean LPIPS/DISTS among GPU shaders) |
| [C4F32 Restore](ArtCNN_C4F32_Restore.glsl) | Wider restoration model |
| [C4F16 Grouped](ArtCNN_C4F16_Grouped.glsl) · [C4F32 Grouped](ArtCNN_C4F32_Grouped.glsl) | Grouped convolutions for lower playback cost at each width |

## Optimized stock presets

FusedFMA uses the upstream weights with pass fusion, workgroup tuning, and fused multiply-adds. FMA can change rounding. The Exact shaders use pass fusion and workgroup changes and matched stock 16-bit luma on the scored frames.

| Upstream preset | FusedFMA | Exact |
| --- | --- | --- |
| C4F16 neutral | [GLSL](ArtCNN_C4F16_FusedFMA.glsl) | [GLSL](ArtCNN_C4F16_Exact.glsl) |
| C4F16 DN | [GLSL](ArtCNN_C4F16_DN_FusedFMA.glsl) | — |
| C4F16 DS | [GLSL](ArtCNN_C4F16_DS_FusedFMA.glsl) | [GLSL](ArtCNN_C4F16_DS_Exact.glsl) |
| C4F32 neutral | [GLSL](ArtCNN_C4F32_FusedFMA.glsl) | [GLSL](ArtCNN_C4F32_Exact.glsl) |
| C4F32 DN | [GLSL](ArtCNN_C4F32_DN_FusedFMA.glsl) | — |
| C4F32 DS | [GLSL](ArtCNN_C4F32_DS_FusedFMA.glsl) | [GLSL](ArtCNN_C4F32_DS_Exact.glsl) |

If you installed these shaders before this directory existed, update your mpv paths from `enhanced/pareto/` to `GLSL/Optimized/` and keep the same filename. See the [GPU benchmark](../../enhanced/PARETO.md) for speed, reconstruction quality, and export instructions.
