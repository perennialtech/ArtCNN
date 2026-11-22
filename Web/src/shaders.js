// SPDX-FileCopyrightText: 2026 perennialtech contributors
// SPDX-License-Identifier: MIT

/**
 * WGSL compute and render shader builders for ArtCNN C-series models.
 *
 * All intermediate feature maps use planar vec4<T> storage buffers indexed as:
 *   ((z * height + y) * width + x)
 * When `useF16` is true (adapters with `shader-f16`), weights, workgroup tiles,
 * and intermediate feature buffers use `f16` / `vec4<f16>` / `mat4x4<f16>` for
 * half-size downloads, half VRAM bandwidth, and half shared memory footprint.
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
        V4: "vec4<f16>",
        M4: "mat4x4<f16>",
        zeroF: "0.0h",
        zeroV4: "vec4<f16>(0.0h)",
        lit: (v) => `${Number(v).toFixed(6)}h`,
      }
    : {
        enable: "",
        F: "f32",
        V4: "vec4<f32>",
        M4: "mat4x4<f32>",
        zeroF: "0.0",
        zeroV4: "vec4<f32>(0.0)",
        lit: (v) => `${Number(v).toFixed(6)}`,
      };
}

function activationExpr(varName, activation, leakySlope, t) {
  if (activation === "relu") {
    return `max(${varName}, ${t.zeroV4})`;
  }
  if (activation === "leaky_relu") {
    return `max(${varName}, ${varName} * ${t.lit(leakySlope)})`;
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
 * First 3x3 convolution: 1-channel luma f32 buffer -> C_OUT_V4 planar vec4<F> buffer.
 */
export function buildConvIn1Shader({ cOutV4, activation = "none", leakySlope = 0.1, useF16 = false }) {
  const t = types(useF16);
  const act = activationExpr("acc", activation, leakySlope, t);
  const castIn = useF16 ? "f16(inp[u32(sy) * dims.width + u32(sx)])" : "inp[u32(sy) * dims.width + u32(sx)]";
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
@group(0) @binding(1) var<storage, read> inp: array<f32>;
@group(0) @binding(2) var<storage, read> params: array<${t.V4}>;
@group(0) @binding(3) var<storage, read_write> outBuf: array<${t.V4}>;

var<workgroup> tile: array<${t.F}, ${TILE_AREA}>;

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
      var v = ${t.zeroF};
      if (sx >= 0 && sx < w && sy >= 0 && sy < h) {
        v = ${castIn};
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
 * Dense 3x3 convolution: C_IN_V4 planar vec4<F> -> C_OUT_V4 planar vec4<F>.
 */
export function buildConvDenseShader({ cInV4, cOutV4, activation = "none", leakySlope = 0.1, useF16 = false }) {
  const t = types(useF16);
  const act = activationExpr("acc", activation, leakySlope, t);
  const sharedElems = TILE_AREA * cInV4;
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

var<workgroup> tile: array<${t.V4}, ${sharedElems}>;

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
    outBuf[p * planeStride + outPix] = ${act};
  }
}
`;
}

/**
 * Grouped 4-channel 3x3 spatial conv + 1x1 pointwise mixing conv.
 */
export function buildConvGroupedShader({ gV4, activation = "none", leakySlope = 0.1, useF16 = false }) {
  const t = types(useF16);
  const act = activationExpr("acc", activation, leakySlope, t);
  const sharedElems = TILE_AREA * gV4;
  return /* wgsl */ `${t.enable}
struct Dims {
  width: u32,
  height: u32,
};

const G_V4: u32 = ${gV4}u;
const TILE_W: u32 = ${TILE_W}u;
const TILE_H: u32 = ${TILE_H}u;
const TILE_AREA: u32 = ${TILE_AREA}u;

@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> inp: array<${t.V4}>;
@group(0) @binding(2) var<storage, read> params: array<${t.V4}>;
@group(0) @binding(3) var<storage, read_write> outBuf: array<${t.V4}>;

var<workgroup> tile: array<${t.V4}, ${sharedElems}>;

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
        tile[z * TILE_AREA + tOff] = select(${t.zeroV4}, inp[z * planeStride + pix], inBounds);
      }
    }
  }
  workgroupBarrier();

  if (gid.x >= dims.width || gid.y >= dims.height) {
    return;
  }

  var grouped: array<${t.V4}, ${gV4}>;
  let spatialBase = G_V4;
  for (var z = 0u; z < G_V4; z++) {
    var gAcc = ${t.zeroV4};
    let zTile = z * TILE_AREA;
    let zSpatial = spatialBase + z * 36u;
    for (var ky = 0u; ky < 3u; ky++) {
      let rowBase = zTile + (lid.y + ky) * TILE_W + lid.x;
      for (var kx = 0u; kx < 3u; kx++) {
        let v = tile[rowBase + kx];
        let mOff = zSpatial + (ky * 3u + kx) * 4u;
        let m = ${t.M4}(params[mOff], params[mOff + 1u], params[mOff + 2u], params[mOff + 3u]);
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
      let m = ${t.M4}(params[mOff], params[mOff + 1u], params[mOff + 2u], params[mOff + 3u]);
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
export function buildFinalD2SRgbShader({ cInV4, inputSkip, useF16 = false }) {
  const t = types(useF16);
  const sharedElems = TILE_AREA * cInV4;
  const skipExpr = inputSkip ? " + yBilin" : "";
  const accOut = useF16 ? "vec4<f32>(acc)" : "acc";
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

fn ign(x: f32, y: f32) -> f32 {
  return fract(52.9829189 * fract(0.06711056 * x + 0.00583715 * y));
}

fn loadChroma(sx: i32, sy: i32, maxX: i32, maxY: i32) -> vec2<f32> {
  let rgb = textureLoad(srcTex, vec2<i32>(clamp(sx, 0, maxX), clamp(sy, 0, maxY)), 0).rgb;
  let cb = clamp(dot(rgb, vec3<f32>(-0.114572, -0.385428, 0.5)) + 0.5, 0.0, 1.0) - 0.5;
  let cr = clamp(dot(rgb, vec3<f32>(0.5, -0.454153, -0.045847)) + 0.5, 0.0, 1.0) - 0.5;
  return vec2<f32>(cb, cr);
}

fn sampleChromaBicubicAR(ox: u32, oy: u32, maxX: i32, maxY: i32) -> vec2<f32> {
  let sx = (f32(ox) + 0.5) * 0.5 - 0.5;
  let sy = (f32(oy) + 0.5) * 0.5 - 0.5;
  let x0 = i32(floor(sx));
  let y0 = i32(floor(sy));
  // Exact 1D Catmull-Rom weights at t = 0.25 and t = 0.75
  let wx = select(
    vec4<f32>(-0.0703125, 0.8671875, 0.2265625, -0.0234375),
    vec4<f32>(-0.0234375, 0.2265625, 0.8671875, -0.0703125),
    (ox & 1u) == 0u,
  );
  let wy = select(
    vec4<f32>(-0.0703125, 0.8671875, 0.2265625, -0.0234375),
    vec4<f32>(-0.0234375, 0.2265625, 0.8671875, -0.0703125),
    (oy & 1u) == 0u,
  );
  var acc = vec2<f32>(0.0, 0.0);
  var lo = vec2<f32>(1.0, 1.0);
  var hi = vec2<f32>(-1.0, -1.0);
  for (var j = 0; j < 4; j++) {
    let ry = y0 + j - 1;
    let c0 = loadChroma(x0 - 1, ry, maxX, maxY);
    let c1 = loadChroma(x0, ry, maxX, maxY);
    let c2 = loadChroma(x0 + 1, ry, maxX, maxY);
    let c3 = loadChroma(x0 + 2, ry, maxX, maxY);
    if (j == 1 || j == 2) {
      lo = min(lo, min(c1, c2));
      hi = max(hi, max(c1, c2));
    }
    acc += wy[j] * (wx.x * c0 + wx.y * c1 + wx.z * c2 + wx.w * c3);
  }
  return clamp(acc, lo, hi);
}

fn writeSubpixel(ox: u32, oy: u32, resY: f32, invOutSize: vec2<f32>, maxX: i32, maxY: i32) {
  let uv = (vec2<f32>(f32(ox), f32(oy)) + vec2<f32>(0.5, 0.5)) * invOutSize;
  let rgb = textureSampleLevel(srcTex, linSampler, uv, 0.0).rgb;
  let yBilin = clamp(dot(rgb, vec3<f32>(0.2126, 0.7152, 0.0722)), 0.0, 1.0);
  let chroma = sampleChromaBicubicAR(ox, oy, maxX, maxY);
  let cb = chroma.x;
  let cr = chroma.y;
  let yOut = clamp(resY${skipExpr}, 0.0, 1.0);
  let d = (ign(f32(ox), f32(oy)) + ign(f32(ox) + 17.0, f32(oy) + 31.0) - 1.0) * (0.5 / 255.0);
  let r = clamp(yOut + 1.5748 * cr + d, 0.0, 1.0);
  let g = clamp(yOut - 0.1873 * cb - 0.4681 * cr + d, 0.0, 1.0);
  let b = clamp(yOut + 1.8556 * cb + d, 0.0, 1.0);
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
        tile[z * TILE_AREA + tOff] = select(${t.zeroV4}, mixBuf[idx] + skipBuf[idx], inBounds);
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
        let m = ${t.M4}(params[mOff], params[mOff + 1u], params[mOff + 2u], params[mOff + 3u]);
        acc += m * v;
      }
    }
  }

  let outF32 = ${accOut};
  let ox = gid.x * 2u;
  let oy = gid.y * 2u;
  let invOutSize = vec2<f32>(1.0 / f32(dims.width * 2u), 1.0 / f32(dims.height * 2u));
  let maxX = i32(dims.width) - 1;
  let maxY = i32(dims.height) - 1;
  writeSubpixel(ox, oy, outF32.x, invOutSize, maxX, maxY);
  writeSubpixel(ox + 1u, oy, outF32.y, invOutSize, maxX, maxY);
  writeSubpixel(ox, oy + 1u, outF32.z, invOutSize, maxX, maxY);
  writeSubpixel(ox + 1u, oy + 1u, outF32.w, invOutSize, maxX, maxY);
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
