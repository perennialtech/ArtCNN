// SPDX-FileCopyrightText: 2026 perennialtech contributors
// SPDX-License-Identifier: MIT

/**
 * Thin helper for running ArtCNN's luma ONNX models (ONNX/ and ONNX/CPU/)
 * directly in onnxruntime-web with BT.709 RGB <-> YCbCr and 2x bilinear chroma.
 */

import { bilinear2x, rgbaToYCbCrBT709, ycbcrToRgbaBT709 } from "./color.js";

export class OnnxLumaUpscaler {
  constructor(ort, session) {
    this.backend = "onnxruntime-web";
    this.ort = ort;
    this.session = session;
    this.inputName = session.inputNames[0];
    this.outputName = session.outputNames[0];
  }

  /**
   * Create an OnnxLumaUpscaler given an imported `onnxruntime-web` module and model URL/bytes.
   */
  static async create(ort, modelSource, sessionOptions = { executionProviders: ["webgpu", "wasm"] }) {
    const session = await ort.InferenceSession.create(modelSource, sessionOptions);
    return new OnnxLumaUpscaler(ort, session);
  }

  async upscaleRgba(rgba, width, height) {
    const { y, cb, cr } = rgbaToYCbCrBT709(rgba, width, height);
    const inputTensor = new this.ort.Tensor("float32", y, [1, 1, height, width]);
    const outputs = await this.session.run({ [this.inputName]: inputTensor });
    const outY = outputs[this.outputName].data;
    const outCb = bilinear2x(cb, width, height);
    const outCr = bilinear2x(cr, width, height);
    return ycbcrToRgbaBT709(outY, outCb, outCr, width * 2, height * 2);
  }

  async upscaleToImageData(imageData) {
    const outRgba = await this.upscaleRgba(imageData.data, imageData.width, imageData.height);
    return new ImageData(outRgba, imageData.width * 2, imageData.height * 2);
  }
}
