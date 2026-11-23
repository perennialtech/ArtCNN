// SPDX-FileCopyrightText: 2026 perennialtech contributors
// SPDX-License-Identifier: MIT

/**
 * WGSL compute and render shader builders for clean-still RGB ArtCNN models (C-series & R-series).
 *
 * All intermediate feature maps use planar vec4<F> storage buffers indexed as:
 *   ((z * height + y) * width + x)
 * When `useF16` is true (adapters with `shader-f16`), weights, workgroup tiles,
 * and intermediate feature buffers use `f16` / `vec4<f16>` / `mat4x4<f16>`.
 */

export const WG_X = 8;
export const WG_Y = 8;
const TILE_W = WG_X + 2;
const TILE_H = WG_Y + 2;
const TILE_AREA = TILE_W * TILE_H;

function types(useF16) {
  return useF16
    ? {
        enable: "enable f16;\n",
        F: "f16",
        V3: "vec3<f16>",
        V4: "vec4<f16>",
        M4: "mat4x4<f16>",
        zeroV4: "vec4<f16>(0.0h)",
        lit: (v) => `${Number(v).toFixed(6)}h`,
      }
    : {
        enable: "",
        F: "f32",
        V3: "vec3<f32>",
        V4: "vec4<f32>",
        M4: "mat4x4<f32>",
        zeroV4: "vec4<f32>(0.0)",
        lit: (v) => `${Number(v).toFixed(6)}`,
      };
}

function preluConstDecl(preluSlopes, countV4, t) {
  if (!preluSlopes || preluSlopes.length !== countV4 * 4) return "";
  const vecs = [];
  for (let p = 0; p < countV4; p++) {
    const a = t.lit(preluSlopes[p * 4]);
    const b = t.lit(preluSlopes[p * 4 + 1]);
    const c = t.lit(preluSlopes[p * 4 + 2]);
    const d = t.lit(preluSlopes[p * 4 + 3]);
    vecs.push(`${t.V4}(${a}, ${b}, ${c}, ${d})`);
  }
  return `const PRELU_SLOPES = array<${t.V4}, ${countV4}>(${vecs.join(", ")});\n`;
}

function activationExpr(varName, activation, leakySlope, t, preluSlopes = null) {
  if (activation === "relu") {
    return `max(${varName}, ${t.zeroV4})`;
  }
  if (activation === "leaky_relu") {
    return `max(${varName}, ${varName} * ${t.lit(leakySlope)})`;
  }
  if (activation === "prelu" && preluSlopes) {
    return `max(${varName}, ${varName} * PRELU_SLOPES[p])`;
  }
  return varName;
}

/**
 * First 3x3 convolution: 3-channel RGB texture_2d<f32> -> C_OUT_V4 planar vec4<F> buffer.
 */
export function buildConvIn3Shader({ cOutV4, activation = "none", leakySlope = 0.1, preluSlopes = null, useF16 = false }) {
  const t = types(useF16);
  const preluDecl = activation === "prelu" ? preluConstDecl(preluSlopes, cOutV4, t) : "";
  const act = activationExpr("acc", activation, leakySlope, t, preluSlopes);
  const rgbExpr = useF16
    ? "vec4<f16>(vec3<f16>(textureLoad(srcTex, vec2<i32>(sx, sy), 0).rgb), 0.0h)"
    : "vec4<f32>(textureLoad(srcTex, vec2<i32>(sx, sy), 0).rgb, 0.0)";
  return /* wgsl */ `${t.enable}
struct Dims {
  width: u32,
  height: u32,
};

const C_OUT_V4: u32 = ${cOutV4}u;
const TILE_W: u32 = ${TILE_W}u;
const TILE_H: u32 = ${TILE_H}u;
const TILE_AREA: u32 = ${TILE_AREA}u;

@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var srcTex: texture_2d<f32>;
@group(0) @binding(2) var<storage, read> params: array<${t.V4}>;
@group(0) @binding(3) var<storage, read_write> outBuf: array<${t.V4}>;

${preluDecl}var<workgroup> tile: array<${t.V4}, ${TILE_AREA}>;

@compute @workgroup_size(${WG_X}, ${WG_Y})
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(workgroup_id) wid: vec3<u32>,
) {
  let baseX = i32(wid.x * ${WG_X}u) - 1;
  let baseY = i32(wid.y * ${WG_Y}u) - 1;
  let w = i32(dims.width);
  let h = i32(dims.height);

  for (var ty = lid.y; ty < TILE_H; ty += ${WG_Y}u) {
    let sy = baseY + i32(ty);
    for (var tx = lid.x; tx < TILE_W; tx += ${WG_X}u) {
      let sx = baseX + i32(tx);
      var v = ${t.zeroV4};
      if (sx >= 0 && sx < w && sy >= 0 && sy < h) {
        v = ${rgbExpr};
      }
      tile[ty * TILE_W + tx] = v;
    }
  }
  workgroupBarrier();

  if (gid.x >= dims.width || gid.y >= dims.height) {
    return;
  }

  let planeStride = dims.width * dims.height;
  let outPix = gid.y * dims.width + gid.x;

  for (var p = 0u; p < C_OUT_V4; p++) {
    var acc = params[p];
    let wBase = C_OUT_V4 + p * 27u;
    for (var ky = 0u; ky < 3u; ky++) {
      let rowBase = (lid.y + ky) * TILE_W + lid.x;
      for (var kx = 0u; kx < 3u; kx++) {
        let v = tile[rowBase + kx];
        let wOff = wBase + (ky * 3u + kx) * 3u;
        acc += params[wOff] * v.x + params[wOff + 1u] * v.y + params[wOff + 2u] * v.z;
      }
    }
    outBuf[p * planeStride + outPix] = ${act};
  }
}
`;
}

/**
 * Dense 3x3 convolution: C_IN_V4 planar vec4<F> -> C_OUT_V4 planar vec4<F>.
 * When `residualAdd` is true, adds `resBuf[p * planeStride + outPix]` before activation.
 */
export function buildConvDenseShader({
  cInV4,
  cOutV4,
  activation = "none",
  leakySlope = 0.1,
  preluSlopes = null,
  residualAdd = false,
  useF16 = false,
}) {
  const t = types(useF16);
  const preluDecl = activation === "prelu" ? preluConstDecl(preluSlopes, cOutV4, t) : "";
  const act = activationExpr("acc", activation, leakySlope, t, preluSlopes);
  const sharedElems = TILE_AREA * cInV4;
  const resBinding = residualAdd
    ? `@group(0) @binding(4) var<storage, read> resBuf: array<${t.V4}>;\n`
    : "";
  const resAddStmt = residualAdd ? `    acc += resBuf[p * planeStride + outPix];\n` : "";

  return /* wgsl */ `${t.enable}
struct Dims {
  width: u32,
  height: u32,
};

const C_IN_V4: u32 = ${cInV4}u;
const C_OUT_V4: u32 = ${cOutV4}u;
const TILE_W: u32 = ${TILE_W}u;
const TILE_H: u32 = ${TILE_H}u;
const TILE_AREA: u32 = ${TILE_AREA}u;

@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> inp: array<${t.V4}>;
@group(0) @binding(2) var<storage, read> params: array<${t.V4}>;
@group(0) @binding(3) var<storage, read_write> outBuf: array<${t.V4}>;
${resBinding}
${preluDecl}var<workgroup> tile: array<${t.V4}, ${sharedElems}>;

@compute @workgroup_size(${WG_X}, ${WG_Y})
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(workgroup_id) wid: vec3<u32>,
) {
  let baseX = i32(wid.x * ${WG_X}u) - 1;
  let baseY = i32(wid.y * ${WG_Y}u) - 1;
  let w = i32(dims.width);
  let h = i32(dims.height);
  let planeStride = dims.width * dims.height;

  for (var ty = lid.y; ty < TILE_H; ty += ${WG_Y}u) {
    let sy = baseY + i32(ty);
    for (var tx = lid.x; tx < TILE_W; tx += ${WG_X}u) {
      let sx = baseX + i32(tx);
      let inBounds = (sx >= 0 && sx < w && sy >= 0 && sy < h);
      let pix = select(0u, u32(sy) * dims.width + u32(sx), inBounds);
      let tOff = ty * TILE_W + tx;
      for (var z = 0u; z < C_IN_V4; z++) {
        tile[z * TILE_AREA + tOff] = select(${t.zeroV4}, inp[z * planeStride + pix], inBounds);
      }
    }
  }
  workgroupBarrier();

  if (gid.x >= dims.width || gid.y >= dims.height) {
    return;
  }

  let outPix = gid.y * dims.width + gid.x;

  for (var p = 0u; p < C_OUT_V4; p++) {
    var acc = params[p];
    let pBase = C_OUT_V4 + p * C_IN_V4 * 36u;
    for (var z = 0u; z < C_IN_V4; z++) {
      let zTile = z * TILE_AREA;
      let zBase = pBase + z * 36u;
      for (var ky = 0u; ky < 3u; ky++) {
        let rowBase = zTile + (lid.y + ky) * TILE_W + lid.x;
        for (var kx = 0u; kx < 3u; kx++) {
          let v = tile[rowBase + kx];
          let mOff = zBase + (ky * 3u + kx) * 4u;
          let m = ${t.M4}(params[mOff], params[mOff + 1u], params[mOff + 2u], params[mOff + 3u]);
          acc += m * v;
        }
      }
    }
${resAddStmt}    outBuf[p * planeStride + outPix] = ${act};
  }
}
`;
}

/**
 * Fused final pass for RGB 2x PixelShuffle:
 *   1. Loads (mix_global + conv0_skip) into shared memory
 *   2. Runs F -> 12 (3 vec4s: R_2x2, G_2x2, B_2x2) 3x3 conv
 *   3. Adds optional 2x bilinear RGB skip from srcTex, preserves alpha, and writes 2x2 RGBA pixels.
 */
export function buildFinalD2SRgbOutShader({ cInV4, inputSkip = true, useF16 = false }) {
  const t = types(useF16);
  const sharedElems = TILE_AREA * cInV4;
  const skipExpr = inputSkip ? " + baseRgba.rgb" : "";
  const cast3 = useF16
    ? "vec3<f32>(f32(rVal), f32(gVal), f32(bVal))"
    : "vec3<f32>(rVal, gVal, bVal)";

  return /* wgsl */ `${t.enable}
struct Dims {
  width: u32,
  height: u32,
};

const C_IN_V4: u32 = ${cInV4}u;
const TILE_W: u32 = ${TILE_W}u;
const TILE_H: u32 = ${TILE_H}u;
const TILE_AREA: u32 = ${TILE_AREA}u;

@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> mixBuf: array<${t.V4}>;
@group(0) @binding(2) var<storage, read> skipBuf: array<${t.V4}>;
@group(0) @binding(3) var<storage, read> params: array<${t.V4}>;
@group(0) @binding(4) var srcTex: texture_2d<f32>;
@group(0) @binding(5) var linSampler: sampler;
@group(0) @binding(6) var outTex: texture_storage_2d<rgba8unorm, write>;

var<workgroup> tile: array<${t.V4}, ${sharedElems}>;

fn writeSubpixel(ox: u32, oy: u32, rVal: ${t.F}, gVal: ${t.F}, bVal: ${t.F}, invOutSize: vec2<f32>) {
  let uv = (vec2<f32>(f32(ox), f32(oy)) + vec2<f32>(0.5, 0.5)) * invOutSize;
  let baseRgba = textureSampleLevel(srcTex, linSampler, uv, 0.0);
  let resRgb = ${cast3};
  let rgbOut = clamp(resRgb${skipExpr}, vec3<f32>(0.0), vec3<f32>(1.0));
  textureStore(outTex, vec2<i32>(i32(ox), i32(oy)), vec4<f32>(rgbOut, baseRgba.a));
}

@compute @workgroup_size(${WG_X}, ${WG_Y})
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(workgroup_id) wid: vec3<u32>,
) {
  let baseX = i32(wid.x * ${WG_X}u) - 1;
  let baseY = i32(wid.y * ${WG_Y}u) - 1;
  let w = i32(dims.width);
  let h = i32(dims.height);
  let planeStride = dims.width * dims.height;

  for (var ty = lid.y; ty < TILE_H; ty += ${WG_Y}u) {
    let sy = baseY + i32(ty);
    for (var tx = lid.x; tx < TILE_W; tx += ${WG_X}u) {
      let sx = baseX + i32(tx);
      let inBounds = (sx >= 0 && sx < w && sy >= 0 && sy < h);
      let pix = select(0u, u32(sy) * dims.width + u32(sx), inBounds);
      let tOff = ty * TILE_W + tx;
      for (var z = 0u; z < C_IN_V4; z++) {
        let idx = z * planeStride + pix;
        tile[z * TILE_AREA + tOff] = select(${t.zeroV4}, mixBuf[idx] + skipBuf[idx], inBounds);
      }
    }
  }
  workgroupBarrier();

  if (gid.x >= dims.width || gid.y >= dims.height) {
    return;
  }

  var acc0 = params[0];
  var acc1 = params[1];
  var acc2 = params[2];
  let strideP = C_IN_V4 * 36u;
  let b0 = 3u;
  let b1 = 3u + strideP;
  let b2 = 3u + strideP * 2u;

  for (var z = 0u; z < C_IN_V4; z++) {
    let zTile = z * TILE_AREA;
    let zOff = z * 36u;
    for (var ky = 0u; ky < 3u; ky++) {
      let rowBase = zTile + (lid.y + ky) * TILE_W + lid.x;
      for (var kx = 0u; kx < 3u; kx++) {
        let v = tile[rowBase + kx];
        let kOff = zOff + (ky * 3u + kx) * 4u;
        let m0 = ${t.M4}(params[b0 + kOff], params[b0 + kOff + 1u], params[b0 + kOff + 2u], params[b0 + kOff + 3u]);
        let m1 = ${t.M4}(params[b1 + kOff], params[b1 + kOff + 1u], params[b1 + kOff + 2u], params[b1 + kOff + 3u]);
        let m2 = ${t.M4}(params[b2 + kOff], params[b2 + kOff + 1u], params[b2 + kOff + 2u], params[b2 + kOff + 3u]);
        acc0 += m0 * v;
        acc1 += m1 * v;
        acc2 += m2 * v;
      }
    }
  }

  let ox = gid.x * 2u;
  let oy = gid.y * 2u;
  let invOutSize = vec2<f32>(1.0 / f32(dims.width * 2u), 1.0 / f32(dims.height * 2u));
  writeSubpixel(ox, oy, acc0.x, acc1.x, acc2.x, invOutSize);
  writeSubpixel(ox + 1u, oy, acc0.y, acc1.y, acc2.y, invOutSize);
  writeSubpixel(ox, oy + 1u, acc0.z, acc1.z, acc2.z, invOutSize);
  writeSubpixel(ox + 1u, oy + 1u, acc0.w, acc1.w, acc2.w, invOutSize);
}
`;
}

/**
 * Fullscreen render shader that displays either ArtCNN 2x output or a live split
 * comparison against bilinear 2x of the source texture.
 */
export function buildPresentShader() {
  return /* wgsl */ `
struct PresentParams {
  split: f32,
  showLine: f32,
  _pad0: f32,
  _pad1: f32,
};

struct VSOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) uv: vec2<f32>,
};

@vertex
fn vsMain(@builtin(vertex_index) idx: u32) -> VSOut {
  var positions = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -3.0),
    vec2<f32>(3.0, 1.0),
    vec2<f32>(-1.0, 1.0),
  );
  let p = positions[idx];
  var out: VSOut;
  out.pos = vec4<f32>(p, 0.0, 1.0);
  out.uv = vec2<f32>(p.x * 0.5 + 0.5, 0.5 - p.y * 0.5);
  return out;
}

@group(0) @binding(0) var<uniform> params: PresentParams;
@group(0) @binding(1) var srcTex: texture_2d<f32>;
@group(0) @binding(2) var upTex: texture_2d<f32>;
@group(0) @binding(3) var linSampler: sampler;

@fragment
fn fsMain(in: VSOut) -> @location(0) vec4<f32> {
  let srcCol = textureSample(srcTex, linSampler, in.uv);
  let upCol = textureSample(upTex, linSampler, in.uv);
  return select(upCol, srcCol, in.uv.x < params.split);
}
`;
}
