// SPDX-FileCopyrightText: 2026 perennialtech contributors
// SPDX-License-Identifier: MIT

import { createUpscaler, loadModel } from "../src/index.js";

const modelSelect = document.getElementById("modelSelect");
const backendSelect = document.getElementById("backendSelect");
const splitRange = document.getElementById("splitRange");
const splitValue = document.getElementById("splitValue");
const splitLine = document.getElementById("splitLine");
const rightLabel = document.getElementById("rightLabel");
const backendBadge = document.getElementById("backendBadge");
const sizeBadge = document.getElementById("sizeBadge");
const timingBadge = document.getElementById("timingBadge");
const statusText = document.getElementById("statusText");
const openFileBtn = document.getElementById("openFileBtn");
const sampleBtn = document.getElementById("sampleBtn");
const downloadBtn = document.getElementById("downloadBtn");
const fileInput = document.getElementById("fileInput");
const dropZone = document.getElementById("dropZone");
const canvasContainer = document.getElementById("canvasContainer");
const videoBar = document.getElementById("videoBar");
const playPauseBtn = document.getElementById("playPauseBtn");
const videoStatus = document.getElementById("videoStatus");

let outputCanvas = document.getElementById("outputCanvas");
let catalog = [];
let modelCache = new Map();
let upscaler = null;
let currentSource = null;
let currentVideo = null;
let videoLoopActive = false;
let split = 0.5;

function replaceCanvas() {
  // WebGPU and Canvas2D contexts are mutually exclusive per <canvas> element.
  const fresh = document.createElement("canvas");
  fresh.id = "outputCanvas";
  outputCanvas.replaceWith(fresh);
  outputCanvas = fresh;
}

function createSampleSource(width = 360, height = 220) {
  const c = document.createElement("canvas");
  c.width = width;
  c.height = height;
  const ctx = c.getContext("2d");

  // Sky gradient + cel-shaded mountains + crisp anime-style ink lines
  const grad = ctx.createLinearGradient(0, 0, 0, height);
  grad.addColorStop(0, "#1c2d5a");
  grad.addColorStop(0.55, "#4c79c9");
  grad.addColorStop(1, "#f7c78a");
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, width, height);

  // Sun disk
  ctx.fillStyle = "#fff5d6";
  ctx.beginPath();
  ctx.arc(width * 0.72, height * 0.32, 28, 0, Math.PI * 2);
  ctx.fill();

  // Cel-shaded silhouettes
  ctx.fillStyle = "#2b3864";
  ctx.beginPath();
  ctx.moveTo(0, height);
  ctx.lineTo(0, height * 0.65);
  ctx.lineTo(width * 0.24, height * 0.44);
  ctx.lineTo(width * 0.46, height * 0.62);
  ctx.lineTo(width * 0.7, height * 0.38);
  ctx.lineTo(width, height * 0.58);
  ctx.lineTo(width, height);
  ctx.closePath();
  ctx.fill();
  ctx.strokeStyle = "#111626";
  ctx.lineWidth = 1.6;
  ctx.stroke();

  // Foreground geometric anime-like rooftop & diagonal powerlines
  ctx.fillStyle = "#192033";
  ctx.fillRect(24, height * 0.66, 94, height * 0.34);
  ctx.strokeStyle = "#0b0e17";
  ctx.lineWidth = 1.5;
  ctx.strokeRect(24, height * 0.66, 94, height * 0.34);

  // Lit windows
  ctx.fillStyle = "#ffd984";
  for (let row = 0; row < 3; row++) {
    for (let col = 0; col < 4; col++) {
      ctx.fillRect(34 + col * 20, height * 0.71 + row * 18, 12, 10);
    }
  }

  // Fine diagonal lines and text (classic upscaler test features)
  ctx.strokeStyle = "#0b0e17";
  ctx.lineWidth = 1.1;
  for (let i = 0; i < 4; i++) {
    ctx.beginPath();
    ctx.moveTo(0, 26 + i * 14);
    ctx.lineTo(width, 64 + i * 20);
    ctx.stroke();
  }

  ctx.fillStyle = "#ffffff";
  ctx.font = "bold 16px sans-serif";
  ctx.fillText("ArtCNN 2x", 22, 34);
  ctx.font = "11px monospace";
  ctx.fillStyle = "#d9e6ff";
  ctx.fillText("360x220 -> 720x440", 22, 50);

  return c;
}

function updateSplitUI() {
  const pct = Math.round(split * 100);
  splitRange.value = String(pct);
  splitValue.textContent = `${pct}%`;
  splitLine.style.left = `${split * 100}%`;
  if (upscaler && !videoLoopActive) {
    upscaler.presentOnly(outputCanvas, split);
  }
}

async function getModel(entry) {
  if (!modelCache.has(entry.id)) {
    const loaded = await loadModel(`../models/${entry.manifest}`);
    modelCache.set(entry.id, loaded);
  }
  return modelCache.get(entry.id);
}

async function rebuildUpscaler() {
  const entry = catalog[modelSelect.selectedIndex];
  if (!entry) return;
  statusText.textContent = `Loading ${entry.label} (${(entry.weights_bytes / 1024).toFixed(1)} KB)...`;
  const previousBackend = upscaler?.backend;
  if (upscaler) {
    upscaler.destroy();
    upscaler = null;
  }

  const model = await getModel(entry);
  const requested = backendSelect.value;
  try {
    upscaler = await createUpscaler({ model, backend: requested });
  } catch (err) {
    statusText.textContent = `${err.message} Falling back to CPU.`;
    upscaler = await createUpscaler({ model, backend: "cpu" });
  }

  if (previousBackend && previousBackend !== upscaler.backend) {
    replaceCanvas();
  }

  backendBadge.textContent = upscaler.backend === "webgpu" ? "WebGPU" : "CPU Fallback";
  backendBadge.className = `badge ${upscaler.backend === "webgpu" ? "ok" : "warn"}`;
  rightLabel.textContent = `${entry.label} 2×`;
  statusText.textContent = `${entry.description} — drag across the image to compare.`;

  if (currentSource && !currentVideo) {
    renderStill();
  }
}

function renderStill() {
  if (!upscaler || !currentSource) return;
  const t0 = performance.now();
  const { width, height } = upscaler.renderToCanvas(currentSource, outputCanvas, { split });
  const elapsed = performance.now() - t0;
  sizeBadge.textContent = `${width / 2}×${height / 2} → ${width}×${height}`;
  timingBadge.textContent = `${elapsed.toFixed(1)} ms`;
}

function stopVideo() {
  videoLoopActive = false;
  if (currentVideo) {
    currentVideo.pause();
    URL.revokeObjectURL(currentVideo.src);
    currentVideo = null;
  }
  videoBar.classList.remove("visible");
}

function startVideoLoop(video) {
  videoLoopActive = true;
  let emaMs = 0;

  const step = () => {
    if (!videoLoopActive || currentVideo !== video) return;
    if (video.readyState >= 2 && upscaler) {
      const t0 = performance.now();
      const { width, height } = upscaler.renderToCanvas(video, outputCanvas, { split });
      const dt = performance.now() - t0;
      emaMs = emaMs === 0 ? dt : emaMs * 0.85 + dt * 0.15;
      sizeBadge.textContent = `${width / 2}×${height / 2} → ${width}×${height}`;
      timingBadge.textContent = `${emaMs.toFixed(1)} ms (${(1000 / Math.max(emaMs, 0.1)).toFixed(0)} FPS)`;
    }
    if ("requestVideoFrameCallback" in video) {
      video.requestVideoFrameCallback(step);
    } else {
      requestAnimationFrame(step);
    }
  };

  if ("requestVideoFrameCallback" in video) {
    video.requestVideoFrameCallback(step);
  } else {
    requestAnimationFrame(step);
  }
}

async function handleFile(file) {
  if (!file) return;
  stopVideo();

  if (file.type.startsWith("video/")) {
    const video = document.createElement("video");
    video.src = URL.createObjectURL(file);
    video.loop = true;
    video.muted = true;
    video.playsInline = true;
    await new Promise((resolve, reject) => {
      video.onloadeddata = resolve;
      video.onerror = () => reject(new Error("Could not decode video file."));
    });
    currentVideo = video;
    currentSource = video;
    videoBar.classList.add("visible");
    playPauseBtn.textContent = "Pause";
    videoStatus.textContent = `${file.name} (${video.videoWidth}×${video.videoHeight})`;
    await video.play();
    startVideoLoop(video);
    return;
  }

  const bitmap = await createImageBitmap(file);
  currentSource = bitmap;
  renderStill();
}

async function init() {
  const res = await fetch("../models/catalog.json");
  const data = await res.json();
  catalog = data.models;
  for (const entry of catalog) {
    const opt = document.createElement("option");
    opt.value = entry.id;
    opt.textContent = `${entry.label} (${entry.architecture}, ${(entry.weights_bytes / 1024).toFixed(0)} KB)`;
    if (entry.id === "ArtCNN_C4F16_Restore") opt.selected = true;
    modelSelect.appendChild(opt);
  }

  currentSource = createSampleSource();
  await rebuildUpscaler();
  updateSplitUI();

  modelSelect.addEventListener("change", rebuildUpscaler);
  backendSelect.addEventListener("change", rebuildUpscaler);
  splitRange.addEventListener("input", () => {
    split = Number(splitRange.value) / 100;
    updateSplitUI();
  });

  let dragging = false;
  const setSplitFromEvent = (e) => {
    const rect = canvasContainer.getBoundingClientRect();
    split = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
    updateSplitUI();
  };
  canvasContainer.addEventListener("pointerdown", (e) => {
    dragging = true;
    canvasContainer.setPointerCapture(e.pointerId);
    setSplitFromEvent(e);
  });
  canvasContainer.addEventListener("pointermove", (e) => {
    if (dragging) setSplitFromEvent(e);
  });
  canvasContainer.addEventListener("pointerup", () => {
    dragging = false;
  });

  openFileBtn.addEventListener("click", () => fileInput.click());
  fileInput.addEventListener("change", () => handleFile(fileInput.files?.[0]));
  sampleBtn.addEventListener("click", () => {
    stopVideo();
    currentSource = createSampleSource();
    renderStill();
  });

  downloadBtn.addEventListener("click", async () => {
    if (!upscaler || !currentSource) return;
    const imgData = await upscaler.upscaleToImageData(currentSource);
    const tmp = document.createElement("canvas");
    tmp.width = imgData.width;
    tmp.height = imgData.height;
    tmp.getContext("2d").putImageData(imgData, 0, 0);
    const link = document.createElement("a");
    link.download = `${catalog[modelSelect.selectedIndex]?.id || "artcnn"}_2x.png`;
    link.href = tmp.toDataURL("image/png");
    link.click();
  });

  playPauseBtn.addEventListener("click", () => {
    if (!currentVideo) return;
    if (currentVideo.paused) {
      currentVideo.play();
      playPauseBtn.textContent = "Pause";
    } else {
      currentVideo.pause();
      playPauseBtn.textContent = "Play";
    }
  });

  dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("dragover");
  });
  dropZone.addEventListener("dragleave", () => dropZone.classList.remove("dragover"));
  dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("dragover");
    handleFile(e.dataTransfer?.files?.[0]);
  });

  // Expose handle for automated browser verification.
  window.__artcnnDemo = {
    get state() {
      return {
        backend: upscaler?.backend,
        model: catalog[modelSelect.selectedIndex]?.id,
        width: outputCanvas.width,
        height: outputCanvas.height,
        timing: timingBadge.textContent,
      };
    },
    async upscaleCurrent() {
      return upscaler.upscaleToImageData(currentSource);
    },
  };
}

init().catch((err) => {
  backendBadge.textContent = "Error";
  backendBadge.className = "badge warn";
  statusText.textContent = String(err);
  console.error(err);
});
