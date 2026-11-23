// SPDX-FileCopyrightText: 2026 perennialtech contributors
// SPDX-License-Identifier: MIT

/**
 * Overlap-tiled WebGPU ArtCNN clean-still RGB upscaler (2x and 4x).
 * Supports both core f32 and `shader-f16` half-precision pipelines.
 */

import {
  WG_X,
  WG_Y,
  buildConvDenseShader,
  buildConvIn3Shader,
  buildFinalD2SRgbOutShader,
  buildPresentShader,
} from "./shaders.js";
import { planTiles } from "./tiling.js";

function sourceDimensions(source) {
  if ("naturalWidth" in source && source.naturalWidth > 0) {
    return { width: source.naturalWidth, height: source.naturalHeight };
  }
  if ("width" in source && "height" in source) {
    return { width: source.width, height: source.height };
  }
  throw new Error("Unsupported image source: missing dimensions.");
}

function sourceToImageData(source) {
  if (source instanceof ImageData) return source;
  const { width, height } = sourceDimensions(source);
  const canvas = typeof OffscreenCanvas !== "undefined" ? new OffscreenCanvas(width, height) : document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(source, 0, 0, width, height);
  return ctx.getImageData(0, 0, width, height);
}

function cropRgba(rgba, width, sx, sy, sw, sh) {
  const out = new Uint8Array(sw * sh * 4);
  for (let y = 0; y < sh; y++) {
    const srcOff = ((sy + y) * width + sx) * 4;
    out.set(rgba.subarray(srcOff, srcOff + sw * 4), y * sw * 4);
  }
  return out;
}

export class WebGPUUpscaler {
  constructor(device, model, pipelines, weightBuffers, useF16) {
    this.backend = "webgpu";
    this.precision = useF16 ? "f16" : "f32";
    this.device = device;
    this.model = model;
    this._useF16 = useF16;
    this._pipelines = pipelines;
    this._weightBuffers = weightBuffers;
    this._sampler = device.createSampler({
      magFilter: "linear",
      minFilter: "linear",
      addressModeU: "clamp-to-edge",
      addressModeV: "clamp-to-edge",
    });
    this._dimsBuffer = device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this._presentUniform = device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this._presentModule = device.createShaderModule({ code: buildPresentShader() });
    this._presentPipelines = new Map();
    this._canvasContexts = new WeakMap();
    this._tileW = 0;
    this._tileH = 0;
    this._tileResources = null;
    this._presentSrcTex = null;
    this._presentUpTex = null;
    this._presentW = 0;
    this._presentH = 0;
    this._destroyed = false;
    this._queue = Promise.resolve();
  }

  static async create(model, options = {}) {
    if (!navigator.gpu) {
      throw new Error("WebGPU is not available in this browser.");
    }
    const wantF16 = model.precision === "f16";
    const device =
      options.device ||
      (await (async () => {
        const adapter = await navigator.gpu.requestAdapter({
          powerPreference: options.powerPreference || "high-performance",
        });
        if (!adapter) {
          throw new Error("No compatible WebGPU adapter found.");
        }
        if (wantF16 && !adapter.features.has("shader-f16")) {
          throw new Error("This WebGPU adapter does not support shader-f16 for FP16 weights.");
        }
        return adapter.requestDevice(wantF16 ? { requiredFeatures: ["shader-f16"] } : undefined);
      })());

    if (wantF16 && !device.features.has("shader-f16")) {
      throw new Error("Provided GPUDevice does not have shader-f16 enabled.");
    }

    const { manifest, weights } = model;
    const rawBytes = new Uint8Array(weights.buffer, weights.byteOffset, weights.byteLength);
    const layerPipelines = [];
    const weightBuffers = [];

    for (const layer of manifest.layers) {
      const offset = wantF16 ? layer.byte_offset_f16 : layer.byte_offset;
      const length = wantF16 ? layer.byte_length_f16 : layer.byte_length;
      const slice = rawBytes.subarray(offset, offset + length);
      const buf = device.createBuffer({
        size: slice.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      device.queue.writeBuffer(buf, 0, slice);
      weightBuffers.push(buf);

      let code;
      if (layer.role === "entry") {
        code = buildConvIn3Shader({
          cOutV4: layer.c_out / 4,
          activation: layer.activation,
          leakySlope: manifest.leaky_slope,
          preluSlopes: layer.prelu_slopes ?? null,
          useF16: wantF16,
        });
      } else if (layer.role === "features_d2s") {
        code = buildFinalD2SRgbOutShader({
          cInV4: layer.c_in / 4,
          inputSkip: manifest.input_skip,
          useF16: wantF16,
        });
      } else {
        code = buildConvDenseShader({
          cInV4: layer.c_in / 4,
          cOutV4: layer.c_out / 4,
          activation: layer.activation,
          leakySlope: manifest.leaky_slope,
          preluSlopes: layer.prelu_slopes ?? null,
          residualAdd: Boolean(layer.residual_add),
          useF16: wantF16,
        });
      }

      layerPipelines.push(
        device.createComputePipeline({
          layout: "auto",
          compute: {
            module: device.createShaderModule({ code }),
            entryPoint: "main",
          },
        }),
      );
    }

    return new WebGPUUpscaler(device, model, { layers: layerPipelines }, weightBuffers, wantF16);
  }

  _ensureTileSize(width, height) {
    if (this._tileResources && this._tileW === width && this._tileH === height) {
      return;
    }
    this._releaseTileResources();
    const device = this.device;
    const filtersV4 = this.model.manifest.filters / 4;
    const vec4Bytes = this._useF16 ? 8 : 16;
    const featBytes = width * height * filtersV4 * vec4Bytes;

    const srcTex = device.createTexture({
      size: [width, height, 1],
      format: "rgba8unorm",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
    });
    const outTex = device.createTexture({
      size: [width * 2, height * 2, 1],
      format: "rgba8unorm",
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
    });

    const bytesPerRow = Math.ceil((width * 2 * 4) / 256) * 256;
    const readbackBuf = device.createBuffer({
      size: bytesPerRow * height * 2,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    const skipBuf = device.createBuffer({ size: featBytes, usage: GPUBufferUsage.STORAGE });
    const bufA = device.createBuffer({ size: featBytes, usage: GPUBufferUsage.STORAGE });
    const bufB = device.createBuffer({ size: featBytes, usage: GPUBufferUsage.STORAGE });
    const bufC = device.createBuffer({ size: featBytes, usage: GPUBufferUsage.STORAGE });

    device.queue.writeBuffer(this._dimsBuffer, 0, new Uint32Array([width, height, 0, 0]));

    const layerBindGroups = [];
    const layers = this.model.manifest.layers;
    let curBuf = skipBuf;
    let resAnchor = skipBuf;

    for (let i = 0; i < layers.length; i++) {
      const layer = layers[i];
      const pipeline = this._pipelines.layers[i];
      if (layer.role === "entry") {
        layerBindGroups.push(
          device.createBindGroup({
            layout: pipeline.getBindGroupLayout(0),
            entries: [
              { binding: 0, resource: { buffer: this._dimsBuffer } },
              { binding: 1, resource: srcTex.createView() },
              { binding: 2, resource: { buffer: this._weightBuffers[i] } },
              { binding: 3, resource: { buffer: skipBuf } },
            ],
          }),
        );
        curBuf = skipBuf;
      } else if (layer.role === "res_conv1") {
        resAnchor = curBuf;
        layerBindGroups.push(
          device.createBindGroup({
            layout: pipeline.getBindGroupLayout(0),
            entries: [
              { binding: 0, resource: { buffer: this._dimsBuffer } },
              { binding: 1, resource: { buffer: curBuf } },
              { binding: 2, resource: { buffer: this._weightBuffers[i] } },
              { binding: 3, resource: { buffer: bufC } },
            ],
          }),
        );
        curBuf = bufC;
      } else if (layer.role === "res_conv2") {
        const nextBuf = resAnchor === bufA ? bufB : bufA;
        layerBindGroups.push(
          device.createBindGroup({
            layout: pipeline.getBindGroupLayout(0),
            entries: [
              { binding: 0, resource: { buffer: this._dimsBuffer } },
              { binding: 1, resource: { buffer: curBuf } },
              { binding: 2, resource: { buffer: this._weightBuffers[i] } },
              { binding: 3, resource: { buffer: nextBuf } },
              { binding: 4, resource: { buffer: resAnchor } },
            ],
          }),
        );
        curBuf = nextBuf;
      } else if (layer.role === "features_d2s") {
        layerBindGroups.push(
          device.createBindGroup({
            layout: pipeline.getBindGroupLayout(0),
            entries: [
              { binding: 0, resource: { buffer: this._dimsBuffer } },
              { binding: 1, resource: { buffer: curBuf } },
              { binding: 2, resource: { buffer: skipBuf } },
              { binding: 3, resource: { buffer: this._weightBuffers[i] } },
              { binding: 4, resource: srcTex.createView() },
              { binding: 5, resource: this._sampler },
              { binding: 6, resource: outTex.createView() },
            ],
          }),
        );
      } else {
        const dstBuf = curBuf === bufA ? bufB : bufA;
        layerBindGroups.push(
          device.createBindGroup({
            layout: pipeline.getBindGroupLayout(0),
            entries: [
              { binding: 0, resource: { buffer: this._dimsBuffer } },
              { binding: 1, resource: { buffer: curBuf } },
              { binding: 2, resource: { buffer: this._weightBuffers[i] } },
              { binding: 3, resource: { buffer: dstBuf } },
            ],
          }),
        );
        curBuf = dstBuf;
      }
    }

    this._tileW = width;
    this._tileH = height;
    this._tileResources = {
      srcTex,
      outTex,
      readbackBuf,
      bytesPerRow,
      skipBuf,
      bufA,
      bufB,
      bufC,
      layerBindGroups,
    };
  }

  async _runTile2x(tileRgba, width, height) {
    this._ensureTileSize(width, height);
    const { srcTex, outTex, readbackBuf, bytesPerRow, layerBindGroups } = this._tileResources;

    this.device.queue.writeTexture(
      { texture: srcTex },
      tileRgba,
      { bytesPerRow: width * 4, rowsPerImage: height },
      [width, height, 1],
    );

    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    const gx = Math.ceil(width / WG_X);
    const gy = Math.ceil(height / WG_Y);

    for (let i = 0; i < this._pipelines.layers.length; i++) {
      pass.setPipeline(this._pipelines.layers[i]);
      pass.setBindGroup(0, layerBindGroups[i]);
      pass.dispatchWorkgroups(gx, gy);
    }
    pass.end();

    encoder.copyTextureToBuffer(
      { texture: outTex },
      { buffer: readbackBuf, bytesPerRow, rowsPerImage: height * 2 },
      [width * 2, height * 2, 1],
    );
    this.device.queue.submit([encoder.finish()]);

    await readbackBuf.mapAsync(GPUMapMode.READ);
    const mapped = new Uint8Array(readbackBuf.getMappedRange());
    const outW = width * 2;
    const outH = height * 2;
    const out = new Uint8ClampedArray(outW * outH * 4);
    for (let y = 0; y < outH; y++) {
      const srcOff = y * bytesPerRow;
      out.set(mapped.subarray(srcOff, srcOff + outW * 4), y * outW * 4);
    }
    readbackBuf.unmap();
    return out;
  }

  async _upscale2xTiled(rgba, width, height, options = {}, progressOffset = 0, progressSpan = 1) {
    const tileSize = options.tileSize ?? 256;
    const defaultOverlap = Math.max(16, (this.model.manifest.receptive_radius || 8) + 2);
    const overlap = options.overlap ?? defaultOverlap;
    const tiles = planTiles(width, height, tileSize, overlap);

    if (tiles.length === 1) {
      const out = await this._runTile2x(rgba, width, height);
      options.onProgress?.(progressOffset + progressSpan);
      return out;
    }

    const outW = width * 2;
    const out = new Uint8ClampedArray(outW * height * 2 * 4);
    for (let i = 0; i < tiles.length; i++) {
      if (options.signal?.aborted) throw new DOMException("Aborted", "AbortError");
      const t = tiles[i];
      const crop = cropRgba(rgba, width, t.srcX, t.srcY, t.srcW, t.srcH);
      const tileOut = await this._runTile2x(crop, t.srcW, t.srcH);
      const tOutW = t.srcW * 2;
      const vx = t.validX * 2;
      const vy = t.validY * 2;
      const vw = t.validW * 2;
      const vh = t.validH * 2;
      const dx = t.dstX * 2;
      const dy = t.dstY * 2;
      for (let y = 0; y < vh; y++) {
        const srcOff = ((vy + y) * tOutW + vx) * 4;
        const dstOff = ((dy + y) * outW + dx) * 4;
        out.set(tileOut.subarray(srcOff, srcOff + vw * 4), dstOff);
      }
      options.onProgress?.(progressOffset + ((i + 1) / tiles.length) * progressSpan);
    }
    return out;
  }

  upscaleRgba(rgba, width, height, options = {}) {
    const run = async () => {
      if (this._destroyed) throw new Error("Upscaler has been destroyed.");
      const scale = options.scale ?? 2;
      const srcBytes = rgba instanceof Uint8Array ? rgba : new Uint8Array(rgba.buffer, rgba.byteOffset, rgba.byteLength);
      if (scale === 2) {
        return this._upscale2xTiled(srcBytes, width, height, options, 0, 1);
      }
      if (scale === 4) {
        const mid = await this._upscale2xTiled(srcBytes, width, height, options, 0, 0.25);
        return this._upscale2xTiled(mid, width * 2, height * 2, options, 0.25, 0.75);
      }
      throw new Error(`Unsupported scale: ${scale}. Use 2 or 4.`);
    };
    const next = this._queue.then(run, run);
    this._queue = next.catch(() => {});
    return next;
  }

  async upscaleToImageData(source, options = {}) {
    const scale = options.scale ?? 2;
    const imgData = sourceToImageData(source);
    const outRgba = await this.upscaleRgba(imgData.data, imgData.width, imgData.height, options);
    return new ImageData(outRgba, imgData.width * scale, imgData.height * scale);
  }

  async upscaleToBlob(source, options = {}) {
    const imgData = await this.upscaleToImageData(source, options);
    const canvas = document.createElement("canvas");
    canvas.width = imgData.width;
    canvas.height = imgData.height;
    canvas.getContext("2d").putImageData(imgData, 0, 0);
    return new Promise((resolve, reject) => {
      canvas.toBlob(
        (blob) => (blob ? resolve(blob) : reject(new Error("Failed to encode image blob."))),
        options.type || "image/png",
        options.quality,
      );
    });
  }

  _getPresentPipeline(format) {
    let pipeline = this._presentPipelines.get(format);
    if (!pipeline) {
      pipeline = this.device.createRenderPipeline({
        layout: "auto",
        vertex: { module: this._presentModule, entryPoint: "vsMain" },
        fragment: {
          module: this._presentModule,
          entryPoint: "fsMain",
          targets: [{ format }],
        },
        primitive: { topology: "triangle-list" },
      });
      this._presentPipelines.set(format, pipeline);
    }
    return pipeline;
  }

  _uploadPresentPair(srcData, upData) {
    const outW = upData.width;
    const outH = upData.height;
    if (
      !this._presentSrcTex ||
      !this._presentUpTex ||
      this._presentW !== outW ||
      this._presentH !== outH ||
      this._presentSrcTex.width !== srcData.width ||
      this._presentSrcTex.height !== srcData.height
    ) {
      this._presentSrcTex?.destroy();
      this._presentUpTex?.destroy();
      this._presentSrcTex = this.device.createTexture({
        size: [srcData.width, srcData.height, 1],
        format: "rgba8unorm",
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
      });
      this._presentUpTex = this.device.createTexture({
        size: [outW, outH, 1],
        format: "rgba8unorm",
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
      });
      this._presentW = outW;
      this._presentH = outH;
    }
    this.device.queue.writeTexture(
      { texture: this._presentSrcTex },
      srcData.data,
      { bytesPerRow: srcData.width * 4, rowsPerImage: srcData.height },
      [srcData.width, srcData.height, 1],
    );
    this.device.queue.writeTexture(
      { texture: this._presentUpTex },
      upData.data,
      { bytesPerRow: outW * 4, rowsPerImage: outH },
      [outW, outH, 1],
    );
  }

  async renderToCanvas(source, canvas, options = {}) {
    const scale = options.scale ?? 2;
    const split = options.split ?? 0;
    const srcData = sourceToImageData(source);
    const outRgba = await this.upscaleRgba(srcData.data, srcData.width, srcData.height, options);
    const upData = new ImageData(outRgba, srcData.width * scale, srcData.height * scale);
    this._uploadPresentPair(srcData, upData);

    if (canvas.width !== upData.width || canvas.height !== upData.height) {
      canvas.width = upData.width;
      canvas.height = upData.height;
    }
    this.presentOnly(canvas, split);
    return { width: upData.width, height: upData.height };
  }

  presentOnly(canvas, split = 0) {
    if (!this._presentSrcTex || !this._presentUpTex) return;
    let ctx = this._canvasContexts.get(canvas);
    const format = navigator.gpu.getPreferredCanvasFormat();
    if (!ctx) {
      ctx = canvas.getContext("webgpu");
      ctx.configure({ device: this.device, format, alphaMode: "premultiplied" });
      this._canvasContexts.set(canvas, ctx);
    }
    this.device.queue.writeBuffer(this._presentUniform, 0, new Float32Array([split, 0, 0, 0]));
    const pipeline = this._getPresentPipeline(format);
    const bg = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this._presentUniform } },
        { binding: 1, resource: this._presentSrcTex.createView() },
        { binding: 2, resource: this._presentUpTex.createView() },
        { binding: 3, resource: this._sampler },
      ],
    });
    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: ctx.getCurrentTexture().createView(),
          loadOp: "clear",
          storeOp: "store",
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
        },
      ],
    });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bg);
    pass.draw(3);
    pass.end();
    this.device.queue.submit([encoder.finish()]);
  }

  _releaseTileResources() {
    if (!this._tileResources) return;
    this._tileResources.srcTex.destroy();
    this._tileResources.outTex.destroy();
    this._tileResources.readbackBuf.destroy();
    this._tileResources.skipBuf.destroy();
    this._tileResources.bufA.destroy();
    this._tileResources.bufB.destroy();
    this._tileResources.bufC.destroy();
    this._tileResources = null;
  }

  destroy() {
    if (this._destroyed) return;
    this._destroyed = true;
    this._releaseTileResources();
    this._presentSrcTex?.destroy();
    this._presentUpTex?.destroy();
    for (const buf of this._weightBuffers) buf.destroy();
    this._dimsBuffer.destroy();
    this._presentUniform.destroy();
  }
}
