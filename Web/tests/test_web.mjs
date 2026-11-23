// SPDX-FileCopyrightText: 2026 perennialtech contributors
// SPDX-License-Identifier: MIT

import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

import { bilinear2xRgba, float16ToFloat32 } from "../src/color.js";
import { CpuUpscaler } from "../src/cpu.js";
import {
  buildConvDenseShader,
  buildConvIn3Shader,
  buildFinalD2SRgbOutShader,
  buildPresentShader,
} from "../src/shaders.js";
import { planTiles } from "../src/tiling.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const WEB_DIR = path.resolve(__dirname, "..");
const REPO_ROOT = path.resolve(WEB_DIR, "..");
const MODELS_DIR = path.join(WEB_DIR, "models");

function loadBundleFromDisk(modelId, precision = "f32") {
  const manifest = JSON.parse(fs.readFileSync(path.join(MODELS_DIR, `${modelId}.json`), "utf8"));
  const file = precision === "f16" ? manifest.weights_f16_file : manifest.weights_file;
  const weights = new Uint8Array(fs.readFileSync(path.join(MODELS_DIR, file)));
  return { manifest, weights, precision };
}

test("planTiles returns single tile when image fits and full coverage when tiled", () => {
  const single = planTiles(128, 96, 256, 16);
  assert.equal(single.length, 1);
  assert.deepEqual(single[0], {
    srcX: 0, srcY: 0, srcW: 128, srcH: 96,
    validX: 0, validY: 0, validW: 128, validH: 96,
    dstX: 0, dstY: 0,
  });

  const grid = planTiles(300, 280, 128, 16);
  assert.equal(grid.length, 3 * 3);
  let covered = 0;
  for (const t of grid) {
    assert.ok(t.srcX >= 0 && t.srcY >= 0);
    assert.ok(t.srcX + t.srcW <= 300 && t.srcY + t.srcH <= 280);
    assert.equal(t.validX, t.dstX - t.srcX);
    assert.equal(t.validY, t.dstY - t.srcY);
    covered += t.validW * t.validH;
  }
  assert.equal(covered, 300 * 280);
});

test("bilinear2xRgba preserves constant color and doubles dimensions", () => {
  const rgba = new Uint8ClampedArray(3 * 2 * 4);
  for (let i = 0; i < rgba.length; i += 4) {
    rgba[i] = 120;
    rgba[i + 1] = 180;
    rgba[i + 2] = 60;
    rgba[i + 3] = 200;
  }
  const out = bilinear2xRgba(rgba, 3, 2);
  assert.equal(out.length, 6 * 4 * 4);
  assert.equal(out[0], 120);
  assert.equal(out[1], 180);
  assert.equal(out[2], 60);
  assert.equal(out[3], 200);
});

test("float16ToFloat32 decodes IEEE-754 binary16 values accurately", () => {
  const u16 = new Uint16Array([0x0000, 0x8000, 0x3c00, 0xc000, 0x3800, 0x0400]);
  const f32 = float16ToFloat32(u16);
  assert.equal(f32[0], 0);
  assert.equal(Object.is(f32[1], -0), true);
  assert.equal(f32[2], 1.0);
  assert.equal(f32[3], -2.0);
  assert.equal(f32[4], 0.5);
  assert.ok(Math.abs(f32[5] - Math.pow(2, -14)) < 1e-9);
});

test("WGSL shader builders emit valid f32 and f16 RGB shaders with residual add", () => {
  for (const useF16 of [false, true]) {
    const sIn = buildConvIn3Shader({ cOutV4: 4, useF16 });
    const sDense = buildConvDenseShader({ cInV4: 4, cOutV4: 4, activation: "leaky_relu", leakySlope: 0.1, residualAdd: true, useF16 });
    const sFinal = buildFinalD2SRgbOutShader({ cInV4: 4, inputSkip: true, useF16 });
    assert.match(sIn, /var srcTex: texture_2d<f32>/);
    assert.match(sDense, /var<storage, read> resBuf:/);
    assert.match(sDense, /acc \+= resBuf\[p \* planeStride \+ outPix\];/);
    assert.match(sFinal, /var acc2 = params\[2\];/);
    if (useF16) {
      assert.match(sIn, /^enable f16;/);
      assert.match(sDense, /mat4x4<f16>/);
      assert.match(sFinal, /vec3<f32>\(f32\(rVal\)/);
    }
  }
  assert.match(buildPresentShader(), /fn fsMain/);
});

test("CpuUpscaler matches catalog models, overlap tiling is seam-free, and FP16 matches FP32", () => {
  const catalog = JSON.parse(fs.readFileSync(path.join(MODELS_DIR, "catalog.json"), "utf8"));
  assert.equal(catalog.format, "artcnn-webgpu-rgb-catalog");
  assert.ok(catalog.models.length >= 3);

  const w = 28;
  const h = 24;
  const rgba = new Uint8ClampedArray(w * h * 4);
  for (let i = 0; i < w * h; i++) {
    rgba[i * 4] = (i * 37 + 13) % 256;
    rgba[i * 4 + 1] = (i * 53 + 7) % 256;
    rgba[i * 4 + 2] = (i * 97 + 29) % 256;
    rgba[i * 4 + 3] = 200;
  }

  for (const entry of catalog.models) {
    const b32 = loadBundleFromDisk(entry.id, "f32");
    const b16 = loadBundleFromDisk(entry.id, "f16");
    const up32 = new CpuUpscaler(b32);
    const up16 = new CpuUpscaler(b16);

    const full = up32.upscaleRgba(rgba, w, h, { scale: 2, tileSize: 256 });
    const tiled = up32.upscaleRgba(rgba, w, h, { scale: 2, tileSize: 12, overlap: 16 });
    assert.equal(full.length, w * 2 * h * 2 * 4);
    assert.deepEqual(tiled, full, `${entry.id} tiled output diverged from full-frame`);

    // Alpha preserved
    assert.equal(full[3], 200);

    // FP16 parity (within 2 LSBs)
    const out16 = up16.upscaleRgba(rgba, w, h, { scale: 2, tileSize: 256 });
    let maxL1 = 0;
    for (let i = 0; i < full.length; i++) {
      maxL1 = Math.max(maxL1, Math.abs(full[i] - out16[i]));
    }
    assert.ok(maxL1 <= 2, `${entry.id} f16 vs f32 max diff ${maxL1} > 2`);
  }

  // 4x chained upscale check
  const fast = new CpuUpscaler(loadBundleFromDisk(catalog.models[0].id, "f32"));
  const out4x = fast.upscaleRgba(rgba, w, h, { scale: 4, tileSize: 256 });
  assert.equal(out4x.length, w * 4 * h * 4 * 4);
});
