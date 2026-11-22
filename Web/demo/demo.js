// SPDX-FileCopyrightText: 2026 perennialtech contributors
// SPDX-License-Identifier: MIT

import { createUpscaler, loadModel } from "../src/index.js";

const MODELS_BASE = new URL("../models/", import.meta.url);

const modelSelect = document.getElementById("modelSelect");
const backendSelect = document.getElementById("backendSelect");
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
const viewer = document.getElementById("viewer");
const videoControls = document.getElementById("videoControls");
const playPauseBtn = document.getElementById("playPauseBtn");
const videoName = document.getElementById("videoName");

let outputCanvas = document.getElementById("outputCanvas");
let catalog = [];
const modelCache = new Map();
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

/**
 * Generate a crisp cel-shaded dusk crossing scene at 400x240 so lineart,
 * diagonal cables, and flat shading show the 2x upscaling gain clearly.
 */
function createSampleSource(width = 400, height = 240) {
  const c = document.createElement("canvas");
  c.width = width;
  c.height = height;
  const ctx = c.getContext("2d");

  // Dusk sky gradient
  const sky = ctx.createLinearGradient(0, 0, 0, height * 0.72);
  sky.addColorStop(0, "#1d3568");
  sky.addColorStop(0.48, "#4b79bd");
  sky.addColorStop(0.84, "#e69a73");
  sky.addColorStop(1, "#fbd399");
  ctx.fillStyle = sky;
  ctx.fillRect(0, 0, width, height);

  // Soft cel cloud bands
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

  // Sun disk
  ctx.fillStyle = "#fff6dc";
  ctx.beginPath();
  ctx.arc(286, 84, 24, 0, Math.PI * 2);
  ctx.fill();

  // Distant mountain silhouette
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

  // Midground cel-shaded buildings with dark ink outlines
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

  // Warm lit apartment windows
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

  // Vermilion railway crossing bar & utility pole (classic high-contrast anime edges)
  ctx.fillStyle = "#171c2b";
  ctx.fillRect(198, 42, 7, 198);
  ctx.fillRect(172, 62, 58, 4);
  ctx.fillRect(178, 82, 46, 3);

  // Diagonal overhead catenary wires
  ctx.strokeStyle = "#111521";
  ctx.lineWidth = 1.1;
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

  // Crossing X sign
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

  // Striped yellow/black crossing barrier
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

  return c;
}

function updateSplitUI() {
  const pct = Math.round(split * 100);
  curtain.style.left = `${split * 100}%`;
  curtainHandle.setAttribute("aria-valuenow", String(pct));
  if (upscaler && !videoLoopActive) {
    upscaler.presentOnly(outputCanvas, split);
  }
}

async function getModel(entry) {
  if (!modelCache.has(entry.id)) {
    const manifestUrl = new URL(entry.manifest, MODELS_BASE).href;
    const loaded = await loadModel(manifestUrl);
    modelCache.set(entry.id, loaded);
  }
  return modelCache.get(entry.id);
}

async function rebuildUpscaler() {
  const entry = catalog[modelSelect.selectedIndex];
  if (!entry) return;
  statusText.textContent = `Loading ${entry.label} (${(entry.weights_bytes / 1024).toFixed(0)} KB)...`;

  const previousBackend = upscaler?.backend;
  if (upscaler) {
    upscaler.destroy();
    upscaler = null;
  }

  const model = await getModel(entry);
  const requested = backendSelect.value;
  let fallbackNotice = "";
  try {
    upscaler = await createUpscaler({ model, backend: requested });
  } catch (err) {
    fallbackNotice = `${err.message} Using CPU instead. `;
    upscaler = await createUpscaler({ model, backend: "cpu" });
  }

  if (previousBackend && previousBackend !== upscaler.backend) {
    replaceCanvas();
  }

  const isGpu = upscaler.backend === "webgpu";
  engineDot.className = `engine-dot ${isGpu ? "webgpu" : "cpu"}`;
  engineLabel.textContent = isGpu ? "WebGPU" : "CPU";
  rightLabel.textContent = entry.label;
  statusText.textContent = `${fallbackNotice}${entry.description}. Files stay on your device.`;

  if (currentSource && !currentVideo) {
    renderStill();
  }
}

function renderStill() {
  if (!upscaler || !currentSource) return;
  const t0 = performance.now();
  const { width, height } = upscaler.renderToCanvas(currentSource, outputCanvas, { split });
  const elapsed = performance.now() - t0;
  sizeReadout.textContent = `${width / 2}x${height / 2} to ${width}x${height}`;
  timingReadout.textContent = `${elapsed.toFixed(1)} ms`;
}

function stopVideo() {
  videoLoopActive = false;
  if (currentVideo) {
    currentVideo.pause();
    URL.revokeObjectURL(currentVideo.src);
    currentVideo = null;
  }
  videoControls.classList.remove("visible");
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
      sizeReadout.textContent = `${width / 2}x${height / 2} to ${width}x${height}`;
      timingReadout.textContent = `${emaMs.toFixed(1)} ms`;
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

  try {
    if (file.type.startsWith("video/")) {
      const video = document.createElement("video");
      video.src = URL.createObjectURL(file);
      video.loop = true;
      video.muted = true;
      video.playsInline = true;
      await new Promise((resolve, reject) => {
        video.onloadeddata = resolve;
        video.onerror = () => reject(new Error("The browser could not decode this video format."));
      });
      currentVideo = video;
      currentSource = video;
      videoControls.classList.add("visible");
      playPauseBtn.textContent = "Pause";
      videoName.textContent = file.name;
      await video.play();
      startVideoLoop(video);
      return;
    }

    const bitmap = await createImageBitmap(file);
    currentSource = bitmap;
    renderStill();
  } catch (err) {
    statusText.textContent = `${err.message} Try a PNG, JPEG, WebP, or MP4/WebM file.`;
  }
}

async function init() {
  const catalogUrl = new URL("catalog.json", MODELS_BASE).href;
  const res = await fetch(catalogUrl);
  const data = await res.json();
  catalog = data.models;

  for (const entry of catalog) {
    const opt = document.createElement("option");
    opt.value = entry.id;
    opt.textContent = `${entry.label} (${(entry.weights_bytes / 1024).toFixed(0)} KB)`;
    if (entry.id === "ArtCNN_C4F16_Restore") opt.selected = true;
    modelSelect.appendChild(opt);
  }

  currentSource = createSampleSource();
  await rebuildUpscaler();
  updateSplitUI();

  modelSelect.addEventListener("change", rebuildUpscaler);
  backendSelect.addEventListener("change", rebuildUpscaler);

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

  window.__artcnnDemo = {
    get state() {
      return {
        backend: upscaler?.backend,
        model: catalog[modelSelect.selectedIndex]?.id,
        width: outputCanvas.width,
        height: outputCanvas.height,
        timing: timingReadout.textContent,
      };
    },
    async upscaleCurrent() {
      return upscaler.upscaleToImageData(currentSource);
    },
  };
}

init().catch((err) => {
  engineDot.className = "engine-dot cpu";
  engineLabel.textContent = "Unavailable";
  statusText.textContent = `Could not start upscaler: ${err.message}`;
  console.error(err);
});
