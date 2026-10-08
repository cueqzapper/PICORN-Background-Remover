import {
  PicornBrowserRuntime,
  alphaCanvas,
  cropAlpha,
} from "./browser-runtime.js";

import { whiteGraphicCanvases } from './white-graphic.js';
import { decontaminateForeground } from './foreground.js';

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
const PROFILES = { 320: "Turbo", 384: "Balanced", 512: "Quality" };
const state = {
  blob: null,
  name: "",
  source: null,
  probability: null,
  baseAlpha: null,
  result: null,
  background: "checker",
  strokes: [],
  activeStroke: null,
  brushMode: "keep",
  busy: false,
};

function toast(message) {
  const element = $("#toast");
  element.textContent = message;
  element.classList.add("show");
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => element.classList.remove("show"), 2800);
}

function format(value, digits = 1) {
  return Number.isFinite(Number(value)) ? Number(value).toFixed(digits) : "–";
}

async function imageFrom(source) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = reject;
    image.src = source;
  });
}

function runtimeProgress(step, detail) {
  // The edge refiner fell back from WebGPU to the CPU: the model is ready.
  if (step === "refine-fallback") return;
  const pill = $("#runtime-pill");
  pill.className = "runtime-pill loading";
  const messages = {
    runtime: `Preparing ${detail} …`,
    download: "Downloading the 1.2 MB model once …",
    cached: "Opening the model from browser cache …",
    fallback: "WebGPU unavailable · starting WASM …",
  };
  if (step === "ready") {
    pill.className = "runtime-pill ready";
    $("#runtime-text").textContent = `Client-side · ${detail} · 1.2 MB · 524k parameters`;
  } else {
    $("#runtime-text").textContent = messages[step] || "Loading the browser model …";
  }
}

const runtime = new PicornBrowserRuntime(runtimeProgress);

function setBusy(busy, message = "") {
  state.busy = busy;
  $("#rerun-button").disabled = busy || !state.baseAlpha;
  $("#inference-size").disabled = busy;
  $$('[data-example]').forEach((button) => { button.disabled = busy; });
  if (busy && message) {
    $("#empty-result").hidden = false;
    $("#result-content").hidden = true;
    $("#empty-result h2").textContent = message;
    $("#empty-result p").textContent = "The image stays entirely on this device.";
  }
  $(".mask-refiner").classList.toggle("processing", busy && Boolean(state.baseAlpha));
  $("#auto-refine-status").lastChild.textContent = busy && state.baseAlpha ? "Rebuilding trimap and alpha …" : "Edge matting ready";
  updateStrokeUi();
}

function currentSize() {
  return Number($("#inference-size").value);
}

function resetStrokes() {
  state.strokes = [];
  state.activeStroke = null;
  updateStrokeUi();
  drawStrokes();
}

async function loadBlob(blob, name) {
  if (!blob?.type?.startsWith("image/")) return toast("Choose an image first.");
  if (state.source?.src?.startsWith("blob:")) URL.revokeObjectURL(state.source.src);
  state.blob = blob;
  state.name = name || "Image";
  state.source = await imageFrom(URL.createObjectURL(blob));
  state.baseAlpha = null;
  state.probability = null;
  resetStrokes();
  await runBase();
}

function setProbability(cropped, geometry, guided = false) {
  const started = performance.now();
  state.probability = alphaCanvas(cropped, geometry.resizedWidth, geometry.resizedHeight);
  const graphic = guided ? null : whiteGraphicCanvases(state.source, state.probability);
  state.graphic = graphic;
  if (graphic) state.probability = graphic.matte;
  state.graphicMilliseconds = performance.now() - started;
}

function showResult(result, guided = false) {
  state.result = result;
  $("#empty-result").hidden = true;
  $("#result-content").hidden = false;
  $("#result-name").textContent = state.name;
  $("#result-meta").textContent = `${state.source.naturalWidth} × ${state.source.naturalHeight} px · processed locally${guided ? " · guided correction" : ""}`;
  const totalMilliseconds = result.modelMilliseconds + result.postprocessMilliseconds + state.graphicMilliseconds;
  $("#base-time").textContent = `${format(totalMilliseconds)} ms`;
  $("#detail-time").textContent = `${PROFILES[result.geometry.size]} · ${result.geometry.size} · ${result.backend} · ${result.refinementBackend} edge`;
  $("#tile-count").textContent = "524k";
  render();
  setBusy(false);
}

async function runBase() {
  if (!state.source || state.busy) return;
  resetStrokes();
  setBusy(true, "PICORN is reading the edge …");
  try {
    const result = await runtime.inferAutomatic(state.source, currentSize());
    state.baseAlpha = result.rawAlpha;
    // The zoomed second pass is a sharper matte of the whole image; brush
    // corrections keep working on the first pass's inference grid.
    if (result.zoomed) {
      setProbability(result.zoomed.alpha, { resizedWidth: result.zoomed.width, resizedHeight: result.zoomed.height });
    } else {
      setProbability(result.cropped, result.geometry);
    }
    if (state.graphic) {
      // Subsequent brush edits start from the corrected automatic mask.
      const work = document.createElement('canvas');
      work.width = result.geometry.resizedWidth; work.height = result.geometry.resizedHeight;
      const context = work.getContext('2d');
      context.drawImage(state.probability, 0, 0, work.width, work.height);
      const rgba = context.getImageData(0, 0, work.width, work.height).data;
      state.baseAlpha = new Float32Array(result.rawAlpha);
      for (let y = 0; y < work.height; y++) for (let x = 0; x < work.width; x++) {
        state.baseAlpha[(y + result.geometry.top) * result.geometry.size + x + result.geometry.left]
          = rgba[(y * work.width + x) * 4] / 255;
      }
    }
    showResult(result, false);
  } catch (error) {
    setBusy(false);
    $("#empty-result h2").textContent = "The browser model couldn't start.";
    $("#empty-result p").textContent = error.message;
    $("#runtime-pill").className = "runtime-pill error";
    $("#runtime-text").textContent = "WebGPU/WASM failed to start";
    toast(error.message);
  }
}

async function applyRefinement() {
  if (!state.source || !state.baseAlpha || !state.strokes.length || state.busy) return;
  setBusy(true);
  drawStrokes();
  try {
    const result = await runtime.infer(state.source, currentSize(), state.baseAlpha, state.strokes);
    setProbability(result.cropped, result.geometry, true);
    showResult(result, true);
    const selected = result.smartSelection?.at(-1)?.selectedCells || 0;
    toast(selected ? "Edge and alpha rebuilt." : "Mask rebuilt locally.");
  } catch (error) {
    setBusy(false);
    $("#result-content").hidden = false;
    $("#empty-result").hidden = true;
    toast(error.message);
  }
}

function restoreOriginal() {
  if (!state.baseAlpha || !state.result) return;
  const geometry = state.result.geometry;
  setProbability(cropAlpha(state.baseAlpha, geometry), geometry);
  resetStrokes();
  $("#result-meta").textContent = `${state.source.naturalWidth} × ${state.source.naturalHeight} px · processed locally`;
  render();
  toast("Restored the original PICORN mask.");
}

function sizeCanvas(canvas) {
  const scale = Math.min(1, 1600 / Math.max(state.source.naturalWidth, state.source.naturalHeight));
  canvas.width = Math.max(1, Math.round(state.source.naturalWidth * scale));
  canvas.height = Math.max(1, Math.round(state.source.naturalHeight * scale));
}

function adjustedAlpha(value) {
  const contrast = Number($("#alpha-contrast").value) / 100;
  const shift = Number($("#edge-shift").value) / 100;
  const clipped = Math.min(0.9999, Math.max(0.0001, value));
  const logit = Math.log(clipped / (1 - clipped));
  return 1 / (1 + Math.exp(-(logit * contrast + shift * 4)));
}

function alphaValues(width, height) {
  const work = document.createElement("canvas");
  work.width = width;
  work.height = height;
  const context = work.getContext("2d", { willReadFrequently: true });
  context.drawImage(state.probability, 0, 0, width, height);
  const data = context.getImageData(0, 0, width, height).data;
  const values = new Float32Array(width * height);
  for (let pixel = 0, index = 0; index < data.length; index += 4, pixel++) values[pixel] = adjustedAlpha(data[index] / 255);
  return values;
}

function paintBackground(context, width, height) {
  if (state.background === "checker") {
    context.fillStyle = "#f1f1f1";
    context.fillRect(0, 0, width, height);
    const tile = Math.max(10, Math.round(width / 50));
    context.fillStyle = "#c9cdcf";
    for (let y = 0; y < height; y += tile) {
      for (let x = 0; x < width; x += tile) {
        if ((x / tile + y / tile) % 2 === 0) context.fillRect(x, y, tile, tile);
      }
    }
  } else if (state.background === "gradient") {
    const gradient = context.createLinearGradient(0, 0, width, height);
    gradient.addColorStop(0, "#f9df54");
    gradient.addColorStop(0.48, "#ed4c8c");
    gradient.addColorStop(1, "#5145e6");
    context.fillStyle = gradient;
    context.fillRect(0, 0, width, height);
  } else {
    context.fillStyle = state.background;
    context.fillRect(0, 0, width, height);
  }
}

function renderRefiner(original, alpha, width, height) {
  const preview = $("#refine-preview-canvas");
  const strokes = $("#refine-strokes-canvas");
  preview.width = strokes.width = width;
  preview.height = strokes.height = height;
  const context = preview.getContext("2d", { willReadFrequently: true });
  const cutout = new ImageData(new Uint8ClampedArray(original.data), width, height);
  for (let pixel = 0, index = 3; pixel < alpha.length; pixel++, index += 4) cutout.data[index] = Math.round(alpha[pixel] * 255);
  context.putImageData(cutout, 0, 0);
  fitRefinerCanvases(width, height);
  drawStrokes();
}

function fitRefinerCanvases(width = $("#refine-preview-canvas").width, height = $("#refine-preview-canvas").height) {
  if (!width || !height) return;
  const stage = $("#refiner-stage");
  const availableWidth = Math.max(1, stage.clientWidth);
  const availableHeight = Math.min(720, Math.max(320, window.innerHeight * 0.76));
  const scale = Math.min(availableWidth / width, availableHeight / height);
  const displayWidth = Math.max(1, Math.round(width * scale));
  const displayHeight = Math.max(1, Math.round(height * scale));
  [$("#refine-preview-canvas"), $("#refine-strokes-canvas")].forEach((canvas) => {
    canvas.style.width = `${displayWidth}px`;
    canvas.style.height = `${displayHeight}px`;
  });
}

function render() {
  if (!state.result || !state.source || !state.probability) return;
  const canvases = [$("#original-canvas"), $("#cutout-canvas"), $("#composite-canvas"), $("#alpha-canvas")];
  canvases.forEach(sizeCanvas);
  const { width, height } = canvases[0];
  const originalContext = canvases[0].getContext("2d", { willReadFrequently: true });
  originalContext.drawImage(state.source, 0, 0, width, height);
  const original = originalContext.getImageData(0, 0, width, height);
  if (state.graphic) {
    const work = document.createElement('canvas'); work.width = width; work.height = height;
    const ctx = work.getContext('2d');
    ctx.drawImage(state.graphic.foreground, 0, 0, width, height);
    original.data.set(ctx.getImageData(0, 0, width, height).data);
  }
  const alpha = alphaValues(width, height);
  const cutoutContext = canvases[1].getContext("2d");
  cutoutContext.clearRect(0, 0, width, height);
  const cutout = new ImageData(new Uint8ClampedArray(original.data), width, height);
  const matteContext = canvases[3].getContext("2d");
  const matte = matteContext.createImageData(width, height);
  for (let pixel = 0, index = 0; pixel < alpha.length; pixel++, index += 4) {
    const value = Math.round(alpha[pixel] * 255);
    cutout.data[index + 3] = value;
    matte.data[index] = matte.data[index + 1] = matte.data[index + 2] = value;
    matte.data[index + 3] = 255;
  }
  cutoutContext.putImageData(cutout, 0, 0);
  matteContext.putImageData(matte, 0, 0);
  const compositeContext = canvases[2].getContext("2d");
  paintBackground(compositeContext, width, height);
  const background = compositeContext.getImageData(0, 0, width, height);
  const composite = compositeContext.createImageData(width, height);
  for (let pixel = 0, index = 0; pixel < alpha.length; pixel++, index += 4) {
    const value = alpha[pixel];
    composite.data[index] = original.data[index] * value + background.data[index] * (1 - value);
    composite.data[index + 1] = original.data[index + 1] * value + background.data[index + 1] * (1 - value);
    composite.data[index + 2] = original.data[index + 2] * value + background.data[index + 2] * (1 - value);
    composite.data[index + 3] = 255;
  }
  compositeContext.putImageData(composite, 0, 0);
  renderRefiner(original, alpha, width, height);
  $("#contrast-output").value = (Number($("#alpha-contrast").value) / 100).toFixed(2);
  $("#edge-output").value = $("#edge-shift").value;
}

function drawStrokes() {
  const canvas = $("#refine-strokes-canvas");
  if (!canvas.width || !canvas.height) return;
  const context = canvas.getContext("2d");
  context.clearRect(0, 0, canvas.width, canvas.height);
  context.lineCap = context.lineJoin = "round";
  for (const stroke of state.activeStroke ? [state.activeStroke] : []) {
    const points = stroke.points;
    if (!points.length) continue;
    context.strokeStyle = stroke.mode === "keep" ? "rgba(49, 220, 121, .82)" : "rgba(255, 72, 82, .82)";
    context.fillStyle = context.strokeStyle;
    context.lineWidth = Math.max(2, stroke.radius * Math.min(canvas.width, canvas.height) * 2);
    context.beginPath();
    context.arc(points[0].x * canvas.width, points[0].y * canvas.height, context.lineWidth / 2, 0, Math.PI * 2);
    context.fill();
    if (points.length > 1) {
      context.beginPath();
      context.moveTo(points[0].x * canvas.width, points[0].y * canvas.height);
      points.slice(1).forEach((point) => context.lineTo(point.x * canvas.width, point.y * canvas.height));
      context.stroke();
    }
  }
}

function updateStrokeUi(changed = false) {
  const count = state.strokes.length;
  $("#undo-stroke").disabled = state.busy || count === 0;
  $("#clear-strokes").disabled = state.busy || count === 0;
  $("#stroke-count").textContent = count
    ? `${count} active ${count === 1 ? "correction" : "corrections"}${state.busy ? " · processing" : ""}`
    : "No corrections yet";
  $("#refiner-hint").hidden = count > 0;
}

function pointerPoint(event) {
  const canvas = $("#refine-strokes-canvas");
  const rect = canvas.getBoundingClientRect();
  return {
    x: Math.min(1, Math.max(0, (event.clientX - rect.left) / rect.width)),
    y: Math.min(1, Math.max(0, (event.clientY - rect.top) / rect.height)),
  };
}

function startStroke(event) {
  if (!state.probability || state.busy) return;
  event.preventDefault();
  const canvas = $("#refine-strokes-canvas");
  const rect = canvas.getBoundingClientRect();
  const radius = (Number($("#brush-size").value) / 2) / Math.min(rect.width, rect.height);
  state.activeStroke = { mode: state.brushMode, radius, points: [pointerPoint(event)] };
  state.strokes.push(state.activeStroke);
  canvas.setPointerCapture(event.pointerId);
  drawStrokes();
  updateStrokeUi(true);
}

function continueStroke(event) {
  if (!state.activeStroke) return;
  event.preventDefault();
  const point = pointerPoint(event);
  const previous = state.activeStroke.points.at(-1);
  if (Math.hypot(point.x - previous.x, point.y - previous.y) < state.activeStroke.radius * 0.12) return;
  state.activeStroke.points.push(point);
  drawStrokes();
}

function endStroke(event) {
  if (!state.activeStroke) return;
  event.preventDefault();
  state.activeStroke = null;
  drawStrokes();
  updateStrokeUi();
  void applyRefinement();
}

async function example(button) {
  button.disabled = true;
  try {
    const response = await fetch(button.querySelector("img").src);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    await loadBlob(await response.blob(), button.querySelector("b").textContent);
  } catch (error) {
    toast(error.message);
  } finally {
    button.disabled = state.busy;
  }
}

function exportDimensions() {
  const width = state.source.naturalWidth;
  const height = state.source.naturalHeight;
  const maximumPixels = 24_000_000;
  const maximumSide = 8192;
  const scale = Math.min(1, maximumSide / Math.max(width, height), Math.sqrt(maximumPixels / (width * height)));
  return { width: Math.max(1, Math.round(width * scale)), height: Math.max(1, Math.round(height * scale)), scaled: scale < 0.9999 };
}

function exportMask(matte = false) {
  const canvas = document.createElement("canvas");
  canvas.width = state.probability.width;
  canvas.height = state.probability.height;
  const context = canvas.getContext("2d", { willReadFrequently: true });
  context.drawImage(state.probability, 0, 0);
  const pixels = context.getImageData(0, 0, canvas.width, canvas.height);
  for (let index = 0; index < pixels.data.length; index += 4) {
    const value = Math.round(adjustedAlpha(pixels.data[index] / 255) * 255);
    pixels.data[index] = pixels.data[index + 1] = pixels.data[index + 2] = matte ? value : 255;
    pixels.data[index + 3] = matte ? 255 : value;
  }
  context.putImageData(pixels, 0, 0);
  return canvas;
}

async function download(kind) {
  if (!state.source || !state.probability) return;
  const output = document.createElement("canvas");
  const dimensions = exportDimensions();
  output.width = dimensions.width;
  output.height = dimensions.height;
  const context = output.getContext("2d");
  context.imageSmoothingEnabled = true;
  context.imageSmoothingQuality = "high";
  if (kind === "alpha") {
    context.drawImage(exportMask(true), 0, 0, output.width, output.height);
  } else if (!state.graphic) {
    // Photos: write alpha directly and estimate the true edge colour.
    context.drawImage(state.source, 0, 0, output.width, output.height);
    const image = context.getImageData(0, 0, output.width, output.height);
    const maskCanvas = document.createElement("canvas");
    maskCanvas.width = output.width;
    maskCanvas.height = output.height;
    const maskContext = maskCanvas.getContext("2d", { willReadFrequently: true });
    maskContext.imageSmoothingEnabled = true;
    maskContext.imageSmoothingQuality = "high";
    maskContext.drawImage(exportMask(false), 0, 0, output.width, output.height);
    const maskPixels = maskContext.getImageData(0, 0, output.width, output.height).data;
    const alpha = Float32Array.from({ length: output.width * output.height }, (_, p) => maskPixels[p * 4 + 3] / 255);
    decontaminateForeground(image.data, alpha, output.width, output.height);
    context.putImageData(image, 0, 0);
    if (kind === "composite") {
      context.globalCompositeOperation = "destination-over";
      paintBackground(context, output.width, output.height);
      context.globalCompositeOperation = "source-over";
    }
  } else {
    context.drawImage(state.graphic.foreground, 0, 0, output.width, output.height);
    context.globalCompositeOperation = "destination-in";
    context.drawImage(exportMask(false), 0, 0, output.width, output.height);
    if (kind === "composite") {
      context.globalCompositeOperation = "destination-over";
      paintBackground(context, output.width, output.height);
    }
    context.globalCompositeOperation = "source-over";
  }
  const blob = await new Promise((resolve) => output.toBlob(resolve, "image/png"));
  if (!blob) return toast("Couldn't create the PNG.");
  const link = document.createElement("a");
  const url = URL.createObjectURL(blob);
  link.download = `${state.name.replace(/\.[^.]+$/, "")}-${kind}.png`;
  link.href = url;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
  if (dimensions.scaled) toast(`Export limited to ${output.width} × ${output.height}.`);
}

$("#file-input").addEventListener("change", (event) => loadBlob(event.target.files[0], event.target.files[0]?.name));
$("#inference-size").addEventListener("change", runBase);
$("#dropzone").addEventListener("dragover", (event) => event.preventDefault());
$("#dropzone").addEventListener("drop", (event) => {
  event.preventDefault();
  const file = event.dataTransfer.files[0];
  loadBlob(file, file?.name);
});
document.addEventListener("paste", (event) => {
  const file = [...(event.clipboardData?.files || [])][0];
  if (file) loadBlob(file, file.name);
});
$$('[data-example]').forEach((button) => button.addEventListener("click", () => example(button)));
$("#rerun-button").addEventListener("click", restoreOriginal);
[$("#alpha-contrast"), $("#edge-shift")].forEach((input) => input.addEventListener("input", render));
$$('[data-background]').forEach((button) => button.addEventListener("click", () => {
  state.background = button.dataset.background;
  $$('[data-background]').forEach((item) => item.classList.toggle("active", item === button));
  render();
}));
$("#custom-colour").addEventListener("input", (event) => {
  state.background = event.target.value;
  $$('[data-background]').forEach((item) => item.classList.remove("active"));
  render();
});
$$('[data-download]').forEach((button) => button.addEventListener("click", () => download(button.dataset.download)));
$$('[data-brush-mode]').forEach((button) => button.addEventListener("click", () => {
  state.brushMode = button.dataset.brushMode;
  $$('[data-brush-mode]').forEach((item) => item.classList.toggle("active", item === button));
}));
$("#brush-size").addEventListener("input", (event) => { $("#brush-size-output").value = `${event.target.value} px`; });
$("#undo-stroke").addEventListener("click", () => {
  state.strokes.pop();
  drawStrokes();
  updateStrokeUi();
  if (state.strokes.length) void applyRefinement();
  else restoreOriginal();
});
$("#clear-strokes").addEventListener("click", restoreOriginal);
const strokeCanvas = $("#refine-strokes-canvas");
strokeCanvas.addEventListener("pointerdown", startStroke);
strokeCanvas.addEventListener("pointermove", continueStroke);
strokeCanvas.addEventListener("pointerup", endStroke);
strokeCanvas.addEventListener("pointercancel", endStroke);
let resizeTimer;
window.addEventListener("resize", () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => fitRefinerCanvases(), 80);
});

$("#runtime-text").textContent = `Browser ready · ${navigator.gpu ? "WebGPU" : "WASM"} · nothing uploaded`;

// Warm the runtime after the page has painted. The demo stays interactive while
// the small model is fetched, and the first user action can start immediately.
const warmRuntime = () => runtime.initialize().catch((error) => {
  $("#runtime-pill").className = "runtime-pill error";
  $("#runtime-text").textContent = "Runtime unavailable · click an image to retry";
  console.warn("PICORN runtime warm-up failed", error);
});
if ("requestIdleCallback" in window) window.requestIdleCallback(warmRuntime, { timeout: 1800 });
else setTimeout(warmRuntime, 600);

// Shareable demo URLs are useful for automated smoke tests and documentation.
const requestedDemo = new URLSearchParams(window.location.search).get("demo");
if (requestedDemo) {
  const demoButton = $$('[data-example]').find((button) => button.dataset.example.includes(requestedDemo));
  if (demoButton) setTimeout(() => demoButton.click(), 80);
}
