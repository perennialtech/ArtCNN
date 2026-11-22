// SPDX-FileCopyrightText: 2026 perennialtech contributors
// SPDX-License-Identifier: MIT

import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

import { bilinear2x, rgbaToYCbCrBT709, ycbcrToRgbaBT709 } from "../src/color.js";
import { CpuUpscaler } from "../src/cpu.js";
import { OnnxLumaUpscaler } from "../src/onnx.js";
import {
  buildConvDenseShader,
  buildConvGroupedShader,
  buildConvIn1Shader,
  buildFinalD2SRgbShader,
  buildPresentShader,
  buildRgbToYShader,
} from "../src/shaders.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const WEB_DIR = path.resolve(__dirname, "..");
const REPO_ROOT = path.resolve(WEB_DIR, "..");
const MODELS_DIR = path.join(WEB_DIR, "models");

function loadBundleFromDisk(modelId) {
  const manifest = JSON.parse(fs.readFileSync(path.join(MODELS_DIR, `${modelId}.json`), "utf8"));
  const weights = new Uint8Array(fs.readFileSync(path.join(MODELS_DIR, manifest.weights_file)));
  return { manifest, weights };
}

test("BT.709 RGB <-> YCbCr roundtrip stays within 1 LSB", () => {
  const rgba = new Uint8ClampedArray([
    0, 0, 0, 255,
    255, 255, 255, 255,
    220, 40, 80, 255,
    40, 180, 230, 255,
  ]);
  const { y, cb, cr } = rgbaToYCbCrBT709(rgba, 2, 2);
  const back = ycbcrToRgbaBT709(y, cb, cr, 2, 2);
  for (let i = 0; i < rgba.length; i++) {
    assert.ok(Math.abs(back[i] - rgba[i]) <= 1, `channel ${i}: got ${back[i]}, expected ${rgba[i]}`);
  }
});

test("bilinear2x preserves constant planes and doubles dimensions", () => {
  const src = new Float32Array(6).fill(0.42);
  const dst = bilinear2x(src, 3, 2);
  assert.equal(dst.length, 24);
  for (const v of dst) {
    assert.ok(Math.abs(v - 0.42) < 1e-6);
  }
});

test("WGSL shader builders generate valid entrypoints for dense and grouped configs", () => {
  assert.match(buildRgbToYShader(), /@compute @workgroup_size\(8, 8\)/);
  assert.match(buildConvIn1Shader({ cOutV4: 4, activation: "none" }), /const C_OUT_V4: u32 = 4u;/);
  assert.match(buildConvDenseShader({ cInV4: 4, cOutV4: 4, activation: "leaky_relu", leakySlope: 0.1 }), /0\.100000/);
  assert.match(buildConvGroupedShader({ gV4: 4, activation: "leaky_relu", leakySlope: 0.1 }), /var grouped: array<vec4<f32>, 4>;/);
  assert.match(buildFinalD2SRgbShader({ cInV4: 4, inputSkip: true }), /\+ yBilin/);
  assert.match(buildPresentShader(), /fn fsMain/);
});

test("CpuUpscaler matches Python ArtCNNC.forward() on dense and grouped models", () => {
  const width = 12;
  const height = 10;
  const inputY = new Float32Array(width * height);
  for (let i = 0; i < inputY.length; i++) {
    inputY[i] = ((i * 37 + 13) % 97) / 96.0;
  }

  for (const modelId of ["ArtCNN_C3F8_Fast", "ArtCNN_C4F16_Restore", "ArtCNN_C4F16_Grouped"]) {
    const bundle = loadBundleFromDisk(modelId);
    const upscaler = new CpuUpscaler(bundle);
    const jsOut = upscaler.upscaleLuma(inputY, width, height);
    assert.equal(jsOut.length, width * 2 * height * 2);

    const pyScript = `
import json, sys, numpy as np
from pathlib import Path
from artcnn.webgpu_export import load_checkpoint

ckpt = Path(sys.argv[1])
w, h = int(sys.argv[2]), int(sys.argv[3])
model, grouped = load_checkpoint(ckpt)
x = np.array([((i * 37 + 13) % 97) / 96.0 for i in range(w * h)], dtype=np.float32).reshape(1, 1, h, w)

if not grouped:
    out = model.forward(x, clip=True)[0, 0]
else:
    # Run with grouped spatial + pointwise mixing kernels to match shader semantics
    from artcnn.models_np import im2col_3x3, depth_to_space_2x, bilinear_upscale_2x
    def grouped_fwd(inp, spatial, mix, b):
        N, C, H, W = inp.shape
        g_v4 = C // 4
        out_g = np.empty_like(inp)
        for z in range(g_v4):
            sub = inp[:, z*4:(z+1)*4, :, :]
            cols = im2col_3x3(sub)
            wm = spatial[z*4:(z+1)*4].reshape(4, 36)
            out_g[:, z*4:(z+1)*4, :, :] = (cols @ wm.T).reshape(N, H, W, 4).transpose(0, 3, 1, 2)
        flat = out_g.transpose(0, 2, 3, 1).reshape(N * H * W, C)
        mixed = flat @ mix.T + b
        return mixed.reshape(N, H, W, C).transpose(0, 3, 1, 2)

    c0 = model.conv0.forward(x)
    h_act = c0
    for name, layer in model.layers()[1:-2]:
        if name in grouped:
            pre = grouped_fwd(h_act, *grouped[name], layer.b)
        else:
            pre = layer.forward(h_act)
        h_act = model._act_forward(pre)
    mix_name, mix_layer = model.layers()[-2]
    if mix_name in grouped:
        mg = grouped_fwd(h_act, *grouped[mix_name], mix_layer.b)
    else:
        mg = mix_layer.forward(h_act)
    feats = model.features.forward(mg + c0)
    out = depth_to_space_2x(feats)
    if model.config.input_skip:
        out = out + bilinear_upscale_2x(x)
    out = np.clip(out, 0.0, 1.0)[0, 0]

print(json.dumps(out.flatten().tolist()))
`;
    const ckptPath = path.join(REPO_ROOT, "enhanced", "pareto", `${modelId}.npz`);
    const stdout = execFileSync(
      "uv",
      ["run", "--locked", "python", "-c", pyScript, ckptPath, String(width), String(height)],
      {
        cwd: REPO_ROOT,
        env: { ...process.env, PYTHONPATH: path.join(REPO_ROOT, "enhanced") },
        encoding: "utf8",
      },
    );
    const pyOut = JSON.parse(stdout);
    let maxDiff = 0;
    for (let i = 0; i < pyOut.length; i++) {
      maxDiff = Math.max(maxDiff, Math.abs(jsOut[i] - pyOut[i]));
    }
    assert.ok(maxDiff < 1e-4, `${modelId} max abs diff ${maxDiff} exceeded 1e-4`);
  }
});

test("OnnxLumaUpscaler wraps BT.709 color conversion around an ONNX session", async () => {
  const mockOrt = {
    Tensor: class {
      constructor(type, data, dims) {
        this.type = type;
        this.data = data;
        this.dims = dims;
      }
    },
    InferenceSession: {
      async create() {
        return {
          inputNames: ["input"],
          outputNames: ["output"],
          async run(feeds) {
            const inp = feeds.input;
            const [, , h, w] = inp.dims;
            return { output: { data: bilinear2x(inp.data, w, h) } };
          },
        };
      },
    },
  };
  const upscaler = await OnnxLumaUpscaler.create(mockOrt, "dummy.onnx");
  const rgba = new Uint8ClampedArray(4 * 4 * 4).fill(180);
  const out = await upscaler.upscaleRgba(rgba, 4, 4);
  assert.equal(out.length, 8 * 8 * 4);
  assert.ok(Math.abs(out[0] - 180) <= 1);
});

import { float16ToFloat32 } from "../src/color.js";

test("float16ToFloat32 decodes IEEE-754 binary16 values accurately", () => {
  // 0.0, -0.0, 1.0, -2.0, 0.5, smallest normal (2^-14)
  const u16 = new Uint16Array([0x0000, 0x8000, 0x3c00, 0xc000, 0x3800, 0x0400]);
  const f32 = float16ToFloat32(u16);
  assert.equal(f32[0], 0);
  assert.equal(Object.is(f32[1], -0), true);
  assert.equal(f32[2], 1.0);
  assert.equal(f32[3], -2.0);
  assert.equal(f32[4], 0.5);
  assert.ok(Math.abs(f32[5] - Math.pow(2, -14)) < 1e-9);
});

test("WGSL shader builders emit valid shader-f16 modules when useF16 is true", () => {
  const s1 = buildConvIn1Shader({ cOutV4: 4, activation: "none", useF16: true });
  const s2 = buildConvDenseShader({ cInV4: 4, cOutV4: 4, activation: "leaky_relu", leakySlope: 0.1, useF16: true });
  const s3 = buildConvGroupedShader({ gV4: 4, activation: "leaky_relu", leakySlope: 0.1, useF16: true });
  const s4 = buildFinalD2SRgbShader({ cInV4: 4, inputSkip: true, useF16: true });
  for (const s of [s1, s2, s3, s4]) {
    assert.match(s, /^enable f16;/);
    assert.match(s, /vec4<f16>/);
  }
  assert.match(s2, /mat4x4<f16>/);
  assert.match(s3, /mat4x4<f16>/);
  assert.match(s4, /vec4<f32>\(acc\)/);
});

test("FP16 bundles match FP32 output closely across dense and grouped models", () => {
  const width = 12;
  const height = 10;
  const inputY = new Float32Array(width * height);
  for (let i = 0; i < inputY.length; i++) {
    inputY[i] = ((i * 37 + 13) % 97) / 96.0;
  }
  for (const modelId of ["ArtCNN_C3F8_Fast", "ArtCNN_C4F16_Restore", "ArtCNN_C4F16_Grouped"]) {
    const manifest = JSON.parse(fs.readFileSync(path.join(MODELS_DIR, `${modelId}.json`), "utf8"));
    const f32Weights = new Uint8Array(fs.readFileSync(path.join(MODELS_DIR, manifest.weights_file)));
    const f16Weights = new Uint8Array(fs.readFileSync(path.join(MODELS_DIR, manifest.weights_f16_file)));
    assert.equal(f16Weights.byteLength * 2, f32Weights.byteLength);

    const out32 = new CpuUpscaler({ manifest, weights: f32Weights, precision: "f32" }).upscaleLuma(inputY, width, height);
    const out16 = new CpuUpscaler({ manifest, weights: f16Weights, precision: "f16" }).upscaleLuma(inputY, width, height);
    let maxDiff = 0;
    for (let i = 0; i < out32.length; i++) {
      maxDiff = Math.max(maxDiff, Math.abs(out32[i] - out16[i]));
    }
    assert.ok(maxDiff < 2e-3, `${modelId} f16 vs f32 max abs diff ${maxDiff} exceeded 2e-3`);
  }
});
