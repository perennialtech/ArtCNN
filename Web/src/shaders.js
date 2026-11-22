// SPDX-FileCopyrightText: 2026 perennialtech contributors
// SPDX-License-Identifier: MIT

/**
 * WGSL compute and render shader builders for ArtCNN C-series models.
 *
 * All intermediate feature maps use planar vec4<f32> storage buffers indexed as:
 *   ((z * height + y) * width + x)
 * for coalesced 16-byte loads/stores across 8x8 workgroups and full core WebGPU
 * compatibility (no optional extensions required).
 */

export const WG_X = 8;
export const WG_Y = 8;
const TILE_W = WG_X + 2;
const TILE_H = WG_Y + 2;
const TILE_AREA = TILE_W * TILE_H;

function activationExpr(varName, activation, leakySlope) {
  if (activation === "relu") {
    return `max(${varName}, vec4<f32>(0.0))`;
  }
  if (activation === "leaky_relu") {
    return `max(${varName}, ${varName} * ${Number(leakySlope).toFixed(6)})`;
  }
  return varName;
}

/**
 * Extract BT.709 luma from an RGBA texture_2d<f32> into a flat f32 buffer.
 */
export function buildRgbToYShader() {
  return /* wgsl */ `
struct Dims {
  width: u32,
  height: u32,
};

@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var srcTex: texture_2d<f32>;
@group(0) @binding(2) var<storage, read_write> outY: array<f32>;

@compute @workgroup_size(${WG_X}, ${WG_Y})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= dims.width || gid.y >= dims.height) {
    return;
  }
  let rgb = textureLoad(srcTex, vec2<i32>(i32(gid.x), i32(gid.y)), 0).rgb;
  let y = clamp(dot(rgb, vec3<f32>(0.2126, 0.7152, 0.0722)), 0.0, 1.0);
  outY[gid.y * dims.width + gid.x] = y;
}
`;
}

/**
 * First 3x3 convolution: 1-channel luma f32 buffer -> C_OUT_V4 planar vec4 buffer.
 */
export function buildConvIn1Shader({ cOutV4, activation = "none", leakySlope = 0.1 }) {
  const act = activationExpr("acc", activation, leakySlope);
  return /* wgsl */ `
struct Dims {
  width: u32,
  height: u32,
};

const C_OUT_V4: u32 = ${cOutV4}u;
const TILE_W: u32 = ${TILE_W}u;
const TILE_H: u32 = ${TILE_H}u;
const TILE_AREA: u32 = ${TILE_AREA}u;

@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> inp: array<f32>;
@group(0) @binding(2) var<storage, read> params: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> outBuf: array<vec4<f32>>;

var<workgroup> tile: array<f32, ${TILE_AREA}>;

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
      var v = 0.0;
      if (sx >= 0 && sx < w && sy >= 0 && sy < h) {
        v = inp[u32(sy) * dims.width + u32(sx)];
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
    let wBase = C_OUT_V4 + p * 9u;
    for (var ky = 0u; ky < 3u; ky++) {
      let rowBase = (lid.y + ky) * TILE_W + lid.x;
      for (var kx = 0u; kx < 3u; kx++) {
        let v = tile[rowBase + kx];
        acc += params[wBase + ky * 3u + kx] * v;
      }
    }
    outBuf[p * planeStride + outPix] = ${act};
  }
}
`;
}

/**
 * Dense 3x3 convolution: C_IN_V4 planar vec4 -> C_OUT_V4 planar vec4.
 */
export function buildConvDenseShader({ cInV4, cOutV4, activation = "none", leakySlope = 0.1 }) {
  const act = activationExpr("acc", activation, leakySlope);
  const sharedElems = TILE_AREA * cInV4;
  return /* wgsl */ `
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
@group(0) @binding(1) var<storage, read> inp: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> params: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> outBuf: array<vec4<f32>>;

var<workgroup> tile: array<vec4<f32>, ${sharedElems}>;

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
        tile[z * TILE_AREA + tOff] = select(vec4<f32>(0.0), inp[z * planeStride + pix], inBounds);
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
          let m = mat4x4<f32>(params[mOff], params[mOff + 1u], params[mOff + 2u], params[mOff + 3u]);
          acc += m * v;
        }
      }
    }
    outBuf[p * planeStride + outPix] = ${act};
  }
}
`;
}

/**
 * Grouped 4-channel 3x3 spatial conv + 1x1 pointwise mixing conv.
 */
export function buildConvGroupedShader({ gV4, activation = "none", leakySlope = 0.1 }) {
  const act = activationExpr("acc", activation, leakySlope);
  const sharedElems = TILE_AREA * gV4;
  return /* wgsl */ `
struct Dims {
  width: u32,
  height: u32,
};

const G_V4: u32 = ${gV4}u;
const TILE_W: u32 = ${TILE_W}u;
const TILE_H: u32 = ${TILE_H}u;
const TILE_AREA: u32 = ${TILE_AREA}u;

@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> inp: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> params: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> outBuf: array<vec4<f32>>;

var<workgroup> tile: array<vec4<f32>, ${sharedElems}>;

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
      for (var z = 0u; z < G_V4; z++) {
        tile[z * TILE_AREA + tOff] = select(vec4<f32>(0.0), inp[z * planeStride + pix], inBounds);
      }
    }
  }
  workgroupBarrier();

  if (gid.x >= dims.width || gid.y >= dims.height) {
    return;
  }

  var grouped: array<vec4<f32>, ${gV4}>;
  let spatialBase = G_V4;
  for (var z = 0u; z < G_V4; z++) {
    var gAcc = vec4<f32>(0.0);
    let zTile = z * TILE_AREA;
    let zSpatial = spatialBase + z * 36u;
    for (var ky = 0u; ky < 3u; ky++) {
      let rowBase = zTile + (lid.y + ky) * TILE_W + lid.x;
      for (var kx = 0u; kx < 3u; kx++) {
        let v = tile[rowBase + kx];
        let mOff = zSpatial + (ky * 3u + kx) * 4u;
        let m = mat4x4<f32>(params[mOff], params[mOff + 1u], params[mOff + 2u], params[mOff + 3u]);
        gAcc += m * v;
      }
    }
    grouped[z] = gAcc;
  }

  let mixBase = spatialBase + G_V4 * 36u;
  let outPix = gid.y * dims.width + gid.x;

  for (var p = 0u; p < G_V4; p++) {
    var acc = params[p];
    let pMix = mixBase + p * G_V4 * 4u;
    for (var z = 0u; z < G_V4; z++) {
      let mOff = pMix + z * 4u;
      let m = mat4x4<f32>(params[mOff], params[mOff + 1u], params[mOff + 2u], params[mOff + 3u]);
      acc += m * grouped[z];
    }
    outBuf[p * planeStride + outPix] = ${act};
  }
}
`;
}

/**
 * Fused final pass:
 *   1. Loads (mix_global + conv0_skip) into shared memory
 *   2. Runs F -> 4 3x3 conv
 *   3. Performs 2x2 depth-to-space + optional bilinear input_skip + clamp(0, 1)
 *   4. Composites upscaled luma with 2x bilinear BT.709 chroma and writes 2x2 RGBA pixels.
 */
export function buildFinalD2SRgbShader({ cInV4, inputSkip }) {
  const sharedElems = TILE_AREA * cInV4;
  const skipExpr = inputSkip ? " + yBilin" : "";
  return /* wgsl */ `
struct Dims {
  width: u32,
  height: u32,
};

const C_IN_V4: u32 = ${cInV4}u;
const TILE_W: u32 = ${TILE_W}u;
const TILE_H: u32 = ${TILE_H}u;
const TILE_AREA: u32 = ${TILE_AREA}u;

@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> mixBuf: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> skipBuf: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> params: array<vec4<f32>>;
@group(0) @binding(4) var srcTex: texture_2d<f32>;
@group(0) @binding(5) var linSampler: sampler;
@group(0) @binding(6) var outTex: texture_storage_2d<rgba8unorm, write>;

var<workgroup> tile: array<vec4<f32>, ${sharedElems}>;

fn writeSubpixel(ox: u32, oy: u32, resY: f32, invOutSize: vec2<f32>) {
  let uv = (vec2<f32>(f32(ox), f32(oy)) + vec2<f32>(0.5, 0.5)) * invOutSize;
  let rgb = textureSampleLevel(srcTex, linSampler, uv, 0.0).rgb;
  let yBilin = clamp(dot(rgb, vec3<f32>(0.2126, 0.7152, 0.0722)), 0.0, 1.0);
  let cb = clamp(dot(rgb, vec3<f32>(-0.114572, -0.385428, 0.5)) + 0.5, 0.0, 1.0) - 0.5;
  let cr = clamp(dot(rgb, vec3<f32>(0.5, -0.454153, -0.045847)) + 0.5, 0.0, 1.0) - 0.5;
  let yOut = clamp(resY${skipExpr}, 0.0, 1.0);
  let r = clamp(yOut + 1.5748 * cr, 0.0, 1.0);
  let g = clamp(yOut - 0.1873 * cb - 0.4681 * cr, 0.0, 1.0);
  let b = clamp(yOut + 1.8556 * cb, 0.0, 1.0);
  textureStore(outTex, vec2<i32>(i32(ox), i32(oy)), vec4<f32>(r, g, b, 1.0));
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
        tile[z * TILE_AREA + tOff] = select(vec4<f32>(0.0), mixBuf[idx] + skipBuf[idx], inBounds);
      }
    }
  }
  workgroupBarrier();

  if (gid.x >= dims.width || gid.y >= dims.height) {
    return;
  }

  var acc = params[0];
  let wStart = 1u;
  for (var z = 0u; z < C_IN_V4; z++) {
    let zTile = z * TILE_AREA;
    let zBase = wStart + z * 36u;
    for (var ky = 0u; ky < 3u; ky++) {
      let rowBase = zTile + (lid.y + ky) * TILE_W + lid.x;
      for (var kx = 0u; kx < 3u; kx++) {
        let v = tile[rowBase + kx];
        let mOff = zBase + (ky * 3u + kx) * 4u;
        let m = mat4x4<f32>(params[mOff], params[mOff + 1u], params[mOff + 2u], params[mOff + 3u]);
        acc += m * v;
      }
    }
  }

  let ox = gid.x * 2u;
  let oy = gid.y * 2u;
  let invOutSize = vec2<f32>(1.0 / f32(dims.width * 2u), 1.0 / f32(dims.height * 2u));
  writeSubpixel(ox, oy, acc.x, invOutSize);
  writeSubpixel(ox + 1u, oy, acc.y, invOutSize);
  writeSubpixel(ox, oy + 1u, acc.z, invOutSize);
  writeSubpixel(ox + 1u, oy + 1u, acc.w, invOutSize);
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
  if (in.uv.x < params.split) {
    return textureSample(srcTex, linSampler, in.uv);
  }
  return textureSample(upTex, linSampler, in.uv);
}
`;
}
