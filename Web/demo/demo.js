// SPDX-FileCopyrightText: 2026 perennialtech contributors
// SPDX-License-Identifier: MIT

import { createWorkerUpscaler, detectWebGPUSupport } from "../src/index.js";

const MODELS_BASE = new URL("../models/", import.meta.url);

const modelSelect = document.getElementById("modelSelect");
const scaleSelect = document.getElementById("scaleSelect");
const backendSelect = document.getElementById("backendSelect");
const zoomSelect = document.getElementById("zoomSelect");
const curtain = document.getElementById("curtain");
const curtainHandle = document.getElementById("curtainHandle");
const rightLabel = document.getElementById("rightLabel");
const engineDot = document.getElementById("engineDot");
const engineLabel = document.getElementById("engineLabel");
const sizeReadout = document.getElementById("sizeReadout");
const timingReadout = document.getElementById("timingReadout");
const statusText = document.getElementById("statusText");
const openFileBtn = document.getElementById("openFileBtn");
const sampleBtn = document.getElementById("sampleBtn");
const downloadBtn = document.getElementById("downloadBtn");
const fileInput = document.getElementById("fileInput");
const dropZone = document.getElementById("dropZone");
const viewerWrap = document.getElementById("viewerWrap");
const viewer = document.getElementById("viewer");
const outputCanvas = document.getElementById("outputCanvas");
const progressBar = document.getElementById("progressBar");
const progressFill = document.getElementById("progressFill");

let catalog = [];
let upscaler = null;
let currentSrcData = null;
let lastOutData = null;
let lastSrcCanvas = null;
let lastUpCanvas = null;
let split = 0.5;
let renderSeq = 0;
let webgpuSupport = { available: false, f16: false };

function createSampleImageData(width = 400, height = 240) {
  const c = document.createElement("canvas");
  c.width = width;
  c.height = height;
  const ctx = c.getContext("2d");

  const sky = ctx.createLinearGradient(0, 0, 0, height * 0.72);
  sky.addColorStop(0, "#1d3568");
  sky.addColorStop(0.48, "#4b79bd");
  sky.addColorStop(0.84, "#e69a73");
  sky.addColorStop(1, "#fbd399");
  ctx.fillStyle = sky;
  ctx.fillRect(0, 0, width, height);

  ctx.fillStyle = "rgba(255, 241, 224, 0.78)";
  ctx.beginPath();
  ctx.moveTo(0, 126);
  ctx.quadraticCurveTo(64, 102, 132, 118);
  ctx.quadraticCurveTo(192, 96, 260, 116);
  ctx.quadraticCurveTo(334, 100, 400, 122);
  ctx.lineTo(400, 146);
  ctx.lineTo(0, 146);
  ctx.closePath();
  ctx.fill();

  ctx.fillStyle = "#fff6dc";
  ctx.beginPath();
  ctx.arc(286, 84, 24, 0, Math.PI * 2);
  ctx.fill();

  ctx.fillStyle = "#3a4a78";
  ctx.beginPath();
  ctx.moveTo(0, 168);
  ctx.lineTo(54, 138);
  ctx.lineTo(128, 156);
  ctx.lineTo(206, 126);
  ctx.lineTo(286, 154);
  ctx.lineTo(356, 132);
  ctx.lineTo(400, 150);
  ctx.lineTo(400, 240);
  ctx.lineTo(0, 240);
  ctx.closePath();
  ctx.fill();

  const buildings = [
    { x: 16, y: 122, w: 68, h: 118, fill: "#2a3350", shade: "#20273e" },
    { x: 90, y: 144, w: 54, h: 96, fill: "#313b5c", shade: "#252d47" },
    { x: 294, y: 130, w: 86, h: 110, fill: "#262f4a", shade: "#1d243b" },
  ];
  for (const b of buildings) {
    ctx.fillStyle = b.fill;
    ctx.fillRect(b.x, b.y, b.w, b.h);
    ctx.fillStyle = b.shade;
    ctx.fillRect(b.x + b.w * 0.62, b.y, b.w * 0.38, b.h);
    ctx.strokeStyle = "#111521";
    ctx.lineWidth = 1.4;
    ctx.strokeRect(b.x, b.y, b.w, b.h);
  }

  ctx.fillStyle = "#ffe09a";
  for (let row = 0; row < 4; row++) {
    for (let col = 0; col < 3; col++) {
      ctx.fillRect(26 + col * 18, 136 + row * 19, 10, 11);
    }
  }
  for (let row = 0; row < 3; row++) {
    for (let col = 0; col < 4; col++) {
      if ((row + col) % 3 === 0) continue;
      ctx.fillRect(304 + col * 18, 144 + row * 20, 10, 11);
    }
  }

  ctx.fillStyle = "#171c2b";
  ctx.fillRect(198, 42, 7, 198);
  ctx.fillRect(172, 62, 58, 4);
  ctx.fillRect(178, 82, 46, 3);

  ctx.strokeStyle = "#d94f3d";
  ctx.lineWidth = 1.3;
  const wires = [
    [0, 52, 400, 74],
    [0, 64, 400, 92],
    [0, 82, 400, 106],
    [174, 64, 400, 40],
  ];
  for (const [x1, y1, x2, y2] of wires) {
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();
  }

  ctx.strokeStyle = "#f2f4f8";
  ctx.lineWidth = 4.5;
  ctx.beginPath();
  ctx.moveTo(186, 152);
  ctx.lineTo(217, 183);
  ctx.moveTo(217, 152);
  ctx.lineTo(186, 183);
  ctx.stroke();
  ctx.strokeStyle = "#111521";
  ctx.lineWidth = 1.1;
  ctx.strokeRect(184, 150, 35, 35);

  ctx.save();
  ctx.translate(154, 198);
  ctx.rotate(-0.08);
  ctx.fillStyle = "#f5c43d";
  ctx.fillRect(0, 0, 146, 8);
  ctx.fillStyle = "#161922";
  for (let x = 8; x < 142; x += 20) {
    ctx.fillRect(x, 0, 10, 8);
  }
  ctx.strokeStyle = "#111521";
  ctx.lineWidth = 1;
  ctx.strokeRect(0, 0, 146, 8);
  ctx.restore();

  return ctx.getImageData(0, 0, width, height);
}

function presentSplit() {
  if (!lastSrcCanvas || !lastUpCanvas) return;
  const ctx = outputCanvas.getContext("2d");
  const outW = outputCanvas.width;
  const outH = outputCanvas.height;
  const cut = Math.round(outW * Math.max(0, Math.min(1, split)));
  ctx.drawImage(lastUpCanvas, 0, 0);
  if (cut > 0) {
    ctx.save();
    ctx.beginPath();
    ctx.rect(0, 0, cut, outH);
    ctx.clip();
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(lastSrcCanvas, 0, 0, outW, outH);
    ctx.restore();
  }
}

function updateSplitUI() {
  const pct = Math.round(split * 100);
  curtain.style.left = `${split * 100}%`;
  curtainHandle.setAttribute("aria-valuenow", String(pct));
  presentSplit();
}

function applyZoom() {
  const mode = zoomSelect.value;
  if (mode === "fit" || !lastOutData) {
    viewerWrap.classList.remove("zoomed");
    outputCanvas.style.width = "";
    outputCanvas.style.height = "";
  } else {
    const factor = Number(mode);
    viewerWrap.classList.add("zoomed");
    outputCanvas.style.width = `${lastOutData.width * factor}px`;
    outputCanvas.style.height = `${lastOutData.height * factor}px`;
  }
}

async function renderStill() {
  if (!upscaler || !currentSrcData) return;
  const seq = ++renderSeq;
  const scale = Number(scaleSelect.value) || 2;
  const entry = catalog[modelSelect.selectedIndex];

  progressBar.classList.add("active");
  progressFill.style.width = "4%";
  downloadBtn.disabled = true;

  const t0 = performance.now();
  const outRgba = await upscaler.upscaleRgba(currentSrcData.data, currentSrcData.width, currentSrcData.height, {
    scale,
    tileSize: 256,
    onProgress: (p) => {
      if (seq === renderSeq) {
        progressFill.style.width = `${Math.round(p * 100)}%`;
      }
    },
  });
  if (seq !== renderSeq) return;

  const elapsed = performance.now() - t0;
  const outW = currentSrcData.width * scale;
  const outH = currentSrcData.height * scale;
  lastOutData = new ImageData(outRgba, outW, outH);

  lastSrcCanvas = document.createElement("canvas");
  lastSrcCanvas.width = currentSrcData.width;
  lastSrcCanvas.height = currentSrcData.height;
  lastSrcCanvas.getContext("2d").putImageData(currentSrcData, 0, 0);

  lastUpCanvas = document.createElement("canvas");
  lastUpCanvas.width = outW;
  lastUpCanvas.height = outH;
  lastUpCanvas.getContext("2d").putImageData(lastOutData, 0, 0);

  outputCanvas.width = outW;
  outputCanvas.height = outH;
  applyZoom();
  presentSplit();

  sizeReadout.textContent = `${currentSrcData.width}x${currentSrcData.height} -> ${outW}x${outH}`;
  timingReadout.textContent = `${elapsed.toFixed(1)} ms`;
  rightLabel.textContent = `${entry?.label || "ArtCNN"} (${scale}x)`;
  progressBar.classList.remove("active");
  progressFill.style.width = "0%";
  downloadBtn.disabled = false;
}

async function rebuildUpscaler() {
  const entry = catalog[modelSelect.selectedIndex];
  if (!entry) return;
  const requested = backendSelect.value;
  const useF16 = requested !== "cpu" && webgpuSupport.f16 && Boolean(entry.weights_f16);
  const targetPrecision = useF16 ? "f16" : "f32";
  const dlBytes = useF16 ? entry.weights_f16_bytes : entry.weights_bytes;
  statusText.textContent = `Loading ${entry.label} (${(dlBytes / 1024).toFixed(1)} KB, ${targetPrecision})...`;

  if (upscaler) {
    upscaler.destroy();
    upscaler = null;
  }

  const manifestUrl = new URL(entry.manifest, MODELS_BASE).href;
  upscaler = await createWorkerUpscaler({
    modelUrl: manifestUrl,
    backend: requested,
    precision: targetPrecision,
  });

  const isGpu = upscaler.backend === "webgpu";
  engineDot.className = `engine-dot ${isGpu ? "webgpu" : "cpu"}`;
  engineLabel.textContent = isGpu ? `WebGPU (${upscaler.precision})` : "CPU";
  statusText.textContent = `${entry.description}. Processing stays local in your browser.`;

  if (currentSrcData) {
    await renderStill();
  }
}

async function loadImageBlob(blob) {
  try {
    const bitmap = await createImageBitmap(blob);
    const c = document.createElement("canvas");
    c.width = bitmap.width;
    c.height = bitmap.height;
    const ctx = c.getContext("2d");
    ctx.drawImage(bitmap, 0, 0);
    bitmap.close();
    currentSrcData = ctx.getImageData(0, 0, c.width, c.height);
    await renderStill();
  } catch (err) {
    statusText.textContent = `${err.message} Try a clean PNG, WebP, JPEG, or AVIF image.`;
  }
}

async function init() {
  const [support, res] = await Promise.all([
    detectWebGPUSupport(),
    fetch(new URL("catalog.json", MODELS_BASE).href),
  ]);
  webgpuSupport = support;
  const data = await res.json();
  catalog = data.models;

  for (const entry of catalog) {
    const opt = document.createElement("option");
    opt.value = entry.id;
    const bytes = support.f16 && entry.weights_f16_bytes ? entry.weights_f16_bytes : entry.weights_bytes;
    const kb = bytes < 10240 ? (bytes / 1024).toFixed(1) : Math.round(bytes / 1024);
    opt.textContent = `${entry.label} (${kb} KB)`;
    if (entry.id === "ArtCNN_C4F32_Clean") opt.selected = true;
    modelSelect.appendChild(opt);
  }

  currentSrcData = createSampleImageData();
  await rebuildUpscaler();
  updateSplitUI();

  modelSelect.addEventListener("change", rebuildUpscaler);
  backendSelect.addEventListener("change", rebuildUpscaler);
  scaleSelect.addEventListener("change", renderStill);
  zoomSelect.addEventListener("change", applyZoom);

  let dragging = false;
  const setSplitFromEvent = (e) => {
    const rect = viewer.getBoundingClientRect();
    if (rect.width <= 0) return;
    split = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
    updateSplitUI();
  };
  viewer.addEventListener("pointerdown", (e) => {
    dragging = true;
    viewer.setPointerCapture(e.pointerId);
    setSplitFromEvent(e);
  });
  viewer.addEventListener("pointermove", (e) => {
    if (dragging) setSplitFromEvent(e);
  });
  viewer.addEventListener("pointerup", () => {
    dragging = false;
  });
  curtainHandle.addEventListener("keydown", (e) => {
    if (e.key === "ArrowLeft") {
      split = Math.max(0, split - 0.05);
      updateSplitUI();
    } else if (e.key === "ArrowRight") {
      split = Math.min(1, split + 0.05);
      updateSplitUI();
    }
  });

  openFileBtn.addEventListener("click", () => fileInput.click());
  fileInput.addEventListener("change", () => {
    const file = fileInput.files?.[0];
    if (file) loadImageBlob(file);
  });
  sampleBtn.addEventListener("click", () => {
    currentSrcData = createSampleImageData();
    renderStill();
  });

  downloadBtn.addEventListener("click", () => {
    if (!lastUpCanvas) return;
    const scale = Number(scaleSelect.value) || 2;
    lastUpCanvas.toBlob((blob) => {
      if (!blob) return;
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.download = `${catalog[modelSelect.selectedIndex]?.id || "artcnn"}_${scale}x.png`;
      link.href = url;
      link.click();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    }, "image/png");
  });

  dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("dragover");
  });
  dropZone.addEventListener("dragleave", () => dropZone.classList.remove("dragover"));
  dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("dragover");
    const file = e.dataTransfer?.files?.[0];
    if (file) loadImageBlob(file);
  });

  window.addEventListener("paste", (e) => {
    const items = e.clipboardData?.items || [];
    for (const item of items) {
      if (item.type.startsWith("image/")) {
        const file = item.getAsFile();
        if (file) {
          loadImageBlob(file);
          break;
        }
      }
    }
  });

  window.__artcnnDemo = {
    get state() {
      return {
        backend: upscaler?.backend,
        precision: upscaler?.precision,
        model: catalog[modelSelect.selectedIndex]?.id,
        scale: Number(scaleSelect.value),
        zoom: zoomSelect.value,
        width: outputCanvas.width,
        height: outputCanvas.height,
        timing: timingReadout.textContent,
      };
    },
  };
}

init().catch((err) => {
  engineDot.className = "engine-dot cpu";
  engineLabel.textContent = "Unavailable";
  statusText.textContent = `Could not start upscaler: ${err.message}`;
  console.error(err);
});
