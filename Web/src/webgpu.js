// SPDX-FileCopyrightText: 2026 perennialtech contributors
// SPDX-License-Identifier: MIT

/**
 * WebGPU real-time ArtCNN 2x upscaler for images and <video> elements.
 */

import {
  WG_X,
  WG_Y,
  buildConvDenseShader,
  buildConvGroupedShader,
  buildConvIn1Shader,
  buildFinalD2SRgbShader,
  buildPresentShader,
  buildRgbToYShader,
} from "./shaders.js";

function sourceDimensions(source) {
  if ("videoWidth" in source && source.videoWidth > 0) {
    return { width: source.videoWidth, height: source.videoHeight };
  }
  if ("naturalWidth" in source && source.naturalWidth > 0) {
    return { width: source.naturalWidth, height: source.naturalHeight };
  }
  if ("width" in source && "height" in source) {
    return { width: source.width, height: source.height };
  }
  throw new Error("Unsupported image or video source: missing dimensions.");
}

export class WebGPUUpscaler {
  constructor(device, model, pipelines, weightBuffers) {
    this.backend = "webgpu";
    this.device = device;
    this.model = model;
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
    this._width = 0;
    this._height = 0;
    this._resources = null;
    this._destroyed = false;
  }

  static async create(model, options = {}) {
    if (!navigator.gpu) {
      throw new Error("WebGPU is not available in this browser.");
    }
    const device = options.device || (await (async () => {
      const adapter = await navigator.gpu.requestAdapter({
        powerPreference: options.powerPreference || "high-performance",
      });
      if (!adapter) {
        throw new Error("No compatible WebGPU adapter found.");
      }
      return adapter.requestDevice();
    })());

    const { manifest, weights } = model;
    const rawBytes = new Uint8Array(weights.buffer, weights.byteOffset, weights.byteLength);

    const rgbToY = device.createComputePipeline({
      layout: "auto",
      compute: {
        module: device.createShaderModule({ code: buildRgbToYShader() }),
        entryPoint: "main",
      },
    });

    const layerPipelines = [];
    const weightBuffers = [];

    for (const layer of manifest.layers) {
      const slice = rawBytes.subarray(layer.byte_offset, layer.byte_offset + layer.byte_length);
      const buf = device.createBuffer({
        size: slice.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      device.queue.writeBuffer(buf, 0, slice);
      weightBuffers.push(buf);

      let code;
      if (layer.role === "entry") {
        code = buildConvIn1Shader({
          cOutV4: layer.c_out / 4,
          activation: layer.activation,
          leakySlope: manifest.leaky_slope,
        });
      } else if (layer.role === "features_d2s") {
        code = buildFinalD2SRgbShader({
          cInV4: layer.c_in / 4,
          inputSkip: manifest.input_skip,
        });
      } else if (layer.kind === "conv_grouped") {
        code = buildConvGroupedShader({
          gV4: layer.c_in / 4,
          activation: layer.activation,
          leakySlope: manifest.leaky_slope,
        });
      } else {
        code = buildConvDenseShader({
          cInV4: layer.c_in / 4,
          cOutV4: layer.c_out / 4,
          activation: layer.activation,
          leakySlope: manifest.leaky_slope,
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

    return new WebGPUUpscaler(device, model, { rgbToY, layers: layerPipelines }, weightBuffers);
  }

  _ensureSize(width, height) {
    if (this._resources && this._width === width && this._height === height) {
      return;
    }
    this._releaseFrameResources();
    const device = this.device;
    const filtersV4 = this.model.manifest.filters / 4;
    const featBytes = width * height * filtersV4 * 16;

    const srcTex = device.createTexture({
      size: [width, height, 1],
      format: "rgba8unorm",
      usage:
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.COPY_DST |
        GPUTextureUsage.RENDER_ATTACHMENT,
    });
    const outTex = device.createTexture({
      size: [width * 2, height * 2, 1],
      format: "rgba8unorm",
      usage:
        GPUTextureUsage.STORAGE_BINDING |
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.COPY_SRC,
    });
    const yBuf = device.createBuffer({
      size: width * height * 4,
      usage: GPUBufferUsage.STORAGE,
    });
    const skipBuf = device.createBuffer({ size: featBytes, usage: GPUBufferUsage.STORAGE });
    const pingBuf = device.createBuffer({ size: featBytes, usage: GPUBufferUsage.STORAGE });
    const pongBuf = device.createBuffer({ size: featBytes, usage: GPUBufferUsage.STORAGE });

    device.queue.writeBuffer(this._dimsBuffer, 0, new Uint32Array([width, height, 0, 0]));

    const rgbToYBG = device.createBindGroup({
      layout: this._pipelines.rgbToY.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this._dimsBuffer } },
        { binding: 1, resource: srcTex.createView() },
        { binding: 2, resource: { buffer: yBuf } },
      ],
    });

    const layerBindGroups = [];
    const layers = this.model.manifest.layers;
    let prevBuf = skipBuf;
    let curA = pingBuf;
    let curB = pongBuf;

    for (let i = 0; i < layers.length; i++) {
      const layer = layers[i];
      const pipeline = this._pipelines.layers[i];
      if (layer.role === "entry") {
        layerBindGroups.push(
          device.createBindGroup({
            layout: pipeline.getBindGroupLayout(0),
            entries: [
              { binding: 0, resource: { buffer: this._dimsBuffer } },
              { binding: 1, resource: { buffer: yBuf } },
              { binding: 2, resource: { buffer: this._weightBuffers[i] } },
              { binding: 3, resource: { buffer: skipBuf } },
            ],
          }),
        );
        prevBuf = skipBuf;
      } else if (layer.role === "features_d2s") {
        layerBindGroups.push(
          device.createBindGroup({
            layout: pipeline.getBindGroupLayout(0),
            entries: [
              { binding: 0, resource: { buffer: this._dimsBuffer } },
              { binding: 1, resource: { buffer: prevBuf } },
              { binding: 2, resource: { buffer: skipBuf } },
              { binding: 3, resource: { buffer: this._weightBuffers[i] } },
              { binding: 4, resource: srcTex.createView() },
              { binding: 5, resource: this._sampler },
              { binding: 6, resource: outTex.createView() },
            ],
          }),
        );
      } else {
        const dstBuf = prevBuf === curA ? curB : curA;
        layerBindGroups.push(
          device.createBindGroup({
            layout: pipeline.getBindGroupLayout(0),
            entries: [
              { binding: 0, resource: { buffer: this._dimsBuffer } },
              { binding: 1, resource: { buffer: prevBuf } },
              { binding: 2, resource: { buffer: this._weightBuffers[i] } },
              { binding: 3, resource: { buffer: dstBuf } },
            ],
          }),
        );
        prevBuf = dstBuf;
      }
    }

    this._width = width;
    this._height = height;
    this._resources = {
      srcTex,
      outTex,
      yBuf,
      skipBuf,
      pingBuf,
      pongBuf,
      rgbToYBG,
      layerBindGroups,
    };
  }

  _encodeInference(source) {
    const { width, height } = sourceDimensions(source);
    this._ensureSize(width, height);
    const { srcTex, rgbToYBG, layerBindGroups } = this._resources;

    if (source instanceof ImageData) {
      this.device.queue.writeTexture(
        { texture: srcTex },
        source.data,
        { bytesPerRow: width * 4, rowsPerImage: height },
        [width, height, 1],
      );
    } else {
      this.device.queue.copyExternalImageToTexture(
        { source },
        { texture: srcTex },
        [width, height],
      );
    }

    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    const gx = Math.ceil(width / WG_X);
    const gy = Math.ceil(height / WG_Y);

    pass.setPipeline(this._pipelines.rgbToY);
    pass.setBindGroup(0, rgbToYBG);
    pass.dispatchWorkgroups(gx, gy);

    for (let i = 0; i < this._pipelines.layers.length; i++) {
      pass.setPipeline(this._pipelines.layers[i]);
      pass.setBindGroup(0, layerBindGroups[i]);
      pass.dispatchWorkgroups(gx, gy);
    }
    pass.end();
    return { encoder, width, height };
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

  /**
   * Upscale an image or video frame and present it to a WebGPU canvas.
   * Pass options.split in [0, 1] to show bilinear on the left and ArtCNN on the right.
   */
  renderToCanvas(source, canvas, options = {}) {
    const split = options.split ?? 0;
    const { encoder, width, height } = this._encodeInference(source);
    const outW = width * 2;
    const outH = height * 2;
    if (canvas.width !== outW || canvas.height !== outH) {
      canvas.width = outW;
      canvas.height = outH;
    }

    let ctxInfo = this._canvasContexts.get(canvas);
    if (!ctxInfo) {
      const ctx = canvas.getContext("webgpu");
      if (!ctx) {
        throw new Error("Failed to acquire WebGPU context from canvas.");
      }
      const format = navigator.gpu.getPreferredCanvasFormat();
      ctx.configure({ device: this.device, format, alphaMode: "opaque" });
      ctxInfo = { ctx, format };
      this._canvasContexts.set(canvas, ctxInfo);
    }

    this.device.queue.writeBuffer(this._presentUniform, 0, new Float32Array([split, 0, 0, 0]));
    const pipeline = this._getPresentPipeline(ctxInfo.format);
    const bg = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this._presentUniform } },
        { binding: 1, resource: this._resources.srcTex.createView() },
        { binding: 2, resource: this._resources.outTex.createView() },
        { binding: 3, resource: this._sampler },
      ],
    });

    const renderPass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: ctxInfo.ctx.getCurrentTexture().createView(),
          loadOp: "clear",
          storeOp: "store",
        },
      ],
    });
    renderPass.setPipeline(pipeline);
    renderPass.setBindGroup(0, bg);
    renderPass.draw(3);
    renderPass.end();

    this.device.queue.submit([encoder.finish()]);
    return { width: outW, height: outH };
  }

  /**
   * Re-present the most recently upscaled frame with a new split position without re-running inference.
   */
  presentOnly(canvas, split = 0) {
    if (!this._resources) return;
    const ctxInfo = this._canvasContexts.get(canvas);
    if (!ctxInfo) return;
    this.device.queue.writeBuffer(this._presentUniform, 0, new Float32Array([split, 0, 0, 0]));
    const pipeline = this._getPresentPipeline(ctxInfo.format);
    const bg = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this._presentUniform } },
        { binding: 1, resource: this._resources.srcTex.createView() },
        { binding: 2, resource: this._resources.outTex.createView() },
        { binding: 3, resource: this._sampler },
      ],
    });
    const encoder = this.device.createCommandEncoder();
    const renderPass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: ctxInfo.ctx.getCurrentTexture().createView(),
          loadOp: "clear",
          storeOp: "store",
        },
      ],
    });
    renderPass.setPipeline(pipeline);
    renderPass.setBindGroup(0, bg);
    renderPass.draw(3);
    renderPass.end();
    this.device.queue.submit([encoder.finish()]);
  }

  /**
   * Upscale an image or video frame and read back RGBA pixels as ImageData.
   */
  async upscaleToImageData(source) {
    const { encoder, width, height } = this._encodeInference(source);
    const outW = width * 2;
    const outH = height * 2;
    const unpaddedRow = outW * 4;
    const bytesPerRow = Math.ceil(unpaddedRow / 256) * 256;
    const readback = this.device.createBuffer({
      size: bytesPerRow * outH,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    encoder.copyTextureToBuffer(
      { texture: this._resources.outTex },
      { buffer: readback, bytesPerRow, rowsPerImage: outH },
      [outW, outH, 1],
    );
    this.device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const mapped = new Uint8Array(readback.getMappedRange());
    const rgba = new Uint8ClampedArray(outW * outH * 4);
    for (let y = 0; y < outH; y++) {
      rgba.set(mapped.subarray(y * bytesPerRow, y * bytesPerRow + unpaddedRow), y * unpaddedRow);
    }
    readback.unmap();
    readback.destroy();
    return new ImageData(rgba, outW, outH);
  }

  _releaseFrameResources() {
    if (!this._resources) return;
    this._resources.srcTex.destroy();
    this._resources.outTex.destroy();
    this._resources.yBuf.destroy();
    this._resources.skipBuf.destroy();
    this._resources.pingBuf.destroy();
    this._resources.pongBuf.destroy();
    this._resources = null;
  }

  destroy() {
    if (this._destroyed) return;
    this._destroyed = true;
    this._releaseFrameResources();
    for (const buf of this._weightBuffers) buf.destroy();
    this._dimsBuffer.destroy();
    this._presentUniform.destroy();
  }
}
