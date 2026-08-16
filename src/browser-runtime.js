import {
  geodesicTrimapCorrection,
  guidedRefineHints,
} from "./interactive-matting.js";
import { HybridAlphaRefiner } from "./hybrid-refine-webgpu.js";

const MODEL_PATH = "models/picorn-remove-background-v9.onnx";
const MODEL_CACHE = "picorn-remove-background-v9-20260816";

function assetUrl(path) {
  const base = import.meta.env?.BASE_URL || "./";
  return new URL(`${base}${path}`, window.location.href).href;
}

async function cachedModel(url, onProgress) {
  const cache = "caches" in window ? await caches.open(MODEL_CACHE) : null;
  const cached = cache ? await cache.match(url) : null;
  if (cached) {
    onProgress?.("cached");
    return new Uint8Array(await cached.arrayBuffer());
  }

  onProgress?.("download");
  const response = await fetch(url);
  if (!response.ok) throw new Error(`The model could not be loaded (HTTP ${response.status}).`);
  if (cache) await cache.put(url, response.clone());
  return new Uint8Array(await response.arrayBuffer());
}

export function letterboxGeometry(width, height, size) {
  const scale = size / Math.max(width, height);
  const resizedWidth = Math.max(1, Math.round(width * scale));
  const resizedHeight = Math.max(1, Math.round(height * scale));
  return {
    size,
    left: Math.floor((size - resizedWidth) / 2),
    top: Math.floor((size - resizedHeight) / 2),
    resizedWidth,
    resizedHeight,
  };
}

export function imageTensorData(source, size) {
  const geometry = letterboxGeometry(source.naturalWidth, source.naturalHeight, size);
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = size;
  const context = canvas.getContext("2d", { willReadFrequently: true });
  context.fillStyle = "rgb(114,114,114)";
  context.fillRect(0, 0, size, size);
  context.imageSmoothingEnabled = true;
  context.imageSmoothingQuality = "high";
  context.drawImage(
    source,
    geometry.left,
    geometry.top,
    geometry.resizedWidth,
    geometry.resizedHeight,
  );
  const rgba = context.getImageData(0, 0, size, size).data;
  const plane = size * size;
  const data = new Float32Array(plane * 3);
  for (let pixel = 0, index = 0; pixel < plane; pixel++, index += 4) {
    data[pixel] = rgba[index] / 255;
    data[plane + pixel] = rgba[index + 1] / 255;
    data[plane * 2 + pixel] = rgba[index + 2] / 255;
  }
  return { data, geometry };
}

function paintCircle(values, width, height, cx, cy, radius, target) {
  const minimumX = Math.max(0, Math.floor(cx - radius));
  const maximumX = Math.min(width - 1, Math.ceil(cx + radius));
  const minimumY = Math.max(0, Math.floor(cy - radius));
  const maximumY = Math.min(height - 1, Math.ceil(cy + radius));
  const radiusSquared = radius * radius;
  for (let y = minimumY; y <= maximumY; y++) {
    for (let x = minimumX; x <= maximumX; x++) {
      if ((x - cx) ** 2 + (y - cy) ** 2 <= radiusSquared) values[y * width + x] = target;
    }
  }
}

function strokeTarget(mode) {
  return mode === "keep" ? 1 : 0;
}

function bilinear(values, width, height, x, y) {
  const x0 = Math.max(0, Math.min(width - 1, Math.floor(x)));
  const y0 = Math.max(0, Math.min(height - 1, Math.floor(y)));
  const x1 = Math.min(width - 1, x0 + 1);
  const y1 = Math.min(height - 1, y0 + 1);
  const tx = Math.max(0, Math.min(1, x - x0));
  const ty = Math.max(0, Math.min(1, y - y0));
  const top = values[y0 * width + x0] * (1 - tx) + values[y0 * width + x1] * tx;
  const bottom = values[y1 * width + x0] * (1 - tx) + values[y1 * width + x1] * tx;
  return top * (1 - ty) + bottom * ty;
}

function smartGrid(geometry, maximumSide = 160) {
  const scale = Math.min(1, maximumSide / Math.max(geometry.resizedWidth, geometry.resizedHeight));
  return {
    width: Math.max(1, Math.round(geometry.resizedWidth * scale)),
    height: Math.max(1, Math.round(geometry.resizedHeight * scale)),
  };
}

function sampleSmartFeatures(image, baseAlpha, geometry, grid) {
  const length = grid.width * grid.height;
  const red = new Float32Array(length);
  const green = new Float32Array(length);
  const blue = new Float32Array(length);
  const luminance = new Float32Array(length);
  const entropy = new Float32Array(length);
  const gradient = new Float32Array(length);
  const alpha = new Float32Array(length);
  const plane = geometry.size * geometry.size;
  const redPlane = image.subarray(0, plane);
  const greenPlane = image.subarray(plane, plane * 2);
  const bluePlane = image.subarray(plane * 2, plane * 3);

  for (let y = 0; y < grid.height; y++) {
    const sourceY = geometry.top + ((y + 0.5) / grid.height) * geometry.resizedHeight - 0.5;
    for (let x = 0; x < grid.width; x++) {
      const sourceX = geometry.left + ((x + 0.5) / grid.width) * geometry.resizedWidth - 0.5;
      const index = y * grid.width + x;
      red[index] = bilinear(redPlane, geometry.size, geometry.size, sourceX, sourceY);
      green[index] = bilinear(greenPlane, geometry.size, geometry.size, sourceX, sourceY);
      blue[index] = bilinear(bluePlane, geometry.size, geometry.size, sourceX, sourceY);
      luminance[index] = red[index] * 0.2126 + green[index] * 0.7152 + blue[index] * 0.0722;
      alpha[index] = bilinear(baseAlpha, geometry.size, geometry.size, sourceX, sourceY);
    }
  }

  // A tiny local Shannon fingerprint separates equally coloured flat areas
  // from hair, fur and textured material. It is deliberately computed on the
  // 160 px interaction grid, not on the full export resolution.
  const bins = new Uint8Array(8);
  for (let y = 0; y < grid.height; y++) {
    for (let x = 0; x < grid.width; x++) {
      bins.fill(0);
      let samples = 0;
      for (let dy = -2; dy <= 2; dy++) {
        const sampleY = Math.max(0, Math.min(grid.height - 1, y + dy));
        for (let dx = -2; dx <= 2; dx++) {
          const sampleX = Math.max(0, Math.min(grid.width - 1, x + dx));
          const bin = Math.min(7, Math.floor(luminance[sampleY * grid.width + sampleX] * 8));
          bins[bin] += 1;
          samples += 1;
        }
      }
      let value = 0;
      for (const count of bins) {
        if (!count) continue;
        const probability = count / samples;
        value -= probability * Math.log2(probability);
      }
      const index = y * grid.width + x;
      entropy[index] = value / 3;
      const left = luminance[y * grid.width + Math.max(0, x - 1)];
      const right = luminance[y * grid.width + Math.min(grid.width - 1, x + 1)];
      const top = luminance[Math.max(0, y - 1) * grid.width + x];
      const bottom = luminance[Math.min(grid.height - 1, y + 1) * grid.width + x];
      gradient[index] = Math.min(1, Math.hypot(right - left, bottom - top) * 1.5);
    }
  }
  return { red, green, blue, entropy, gradient, alpha };
}

const SMART_SEED_CORE_SCALE = 0.34;

function rasterizeSmartStroke(stroke, grid, radiusScale = SMART_SEED_CORE_SCALE) {
  const seeds = new Uint8Array(grid.width * grid.height);
  // The visible cursor is the refinement band, not a hard cookie cutter. Only
  // its narrow centre is a sure FG/BG sample; the surrounding pixels belong
  // to the trimap's unknown region and may snap to the actual image edge.
  const radius = Math.max(1, stroke.radius * Math.min(grid.width, grid.height) * radiusScale);
  const points = stroke.points;
  for (let index = 0; index < points.length; index++) {
    const from = points[Math.max(0, index - 1)];
    const to = points[index];
    const x0 = from.x * grid.width - 0.5;
    const y0 = from.y * grid.height - 0.5;
    const x1 = to.x * grid.width - 0.5;
    const y1 = to.y * grid.height - 0.5;
    const distance = Math.hypot(x1 - x0, y1 - y0);
    const steps = Math.max(1, Math.ceil(distance / Math.max(1, radius * 0.45)));
    for (let step = 0; step <= steps; step++) {
      const mix = step / steps;
      paintCircle(
        seeds,
        grid.width,
        grid.height,
        x0 + (x1 - x0) * mix,
        y0 + (y1 - y0) * mix,
        radius,
        1,
      );
    }
  }
  return seeds;
}

function upscaleSmartInfluence(influence, grid, geometry) {
  const result = new Float32Array(geometry.size * geometry.size);
  for (let y = 0; y < geometry.resizedHeight; y++) {
    const gridY = ((y + 0.5) / geometry.resizedHeight) * grid.height - 0.5;
    const targetY = geometry.top + y;
    for (let x = 0; x < geometry.resizedWidth; x++) {
      const gridX = ((x + 0.5) / geometry.resizedWidth) * grid.width - 0.5;
      result[targetY * geometry.size + geometry.left + x] = bilinear(influence, grid.width, grid.height, gridX, gridY);
    }
  }
  return result;
}

/**
 * Turns sparse strokes plus the existing alpha into a local trimap. Weighted
 * foreground/background distances solve its unknown band; a guided filter
 * then computes a soft matte on the original image grid.
 */
export function smartStrokeHintMasks(image, baseAlpha, geometry, strokes, maximumSide = 160) {
  if (!strokes.length) return { ...strokeHintMasks(geometry, strokes), diagnostics: [] };
  const grid = smartGrid(geometry, maximumSide);
  const features = sampleSmartFeatures(image, baseAlpha, geometry, grid);
  const keep = new Float32Array(geometry.size * geometry.size);
  const remove = new Float32Array(geometry.size * geometry.size);
  const diagnostics = [];
  for (const stroke of strokes) {
    const seeds = rasterizeSmartStroke(stroke, grid);
    const brushRadius = Math.max(1, stroke.radius * Math.min(grid.width, grid.height));
    const grown = geodesicTrimapCorrection(features, grid, seeds, stroke.mode, brushRadius);
    for (let index = 0; index < grown.influence.length; index++) {
      const influence = grown.influence[index];
      if (stroke.mode === "keep") features.alpha[index] += (1 - features.alpha[index]) * influence;
      else features.alpha[index] *= 1 - influence;
    }
    const influence = upscaleSmartInfluence(grown.influence, grid, geometry);
    const target = stroke.mode === "keep" ? keep : remove;
    const opposite = stroke.mode === "keep" ? remove : keep;
    for (let index = 0; index < influence.length; index++) {
      if (influence[index] <= 0) continue;
      target[index] = Math.max(target[index], influence[index]);
      opposite[index] *= 1 - influence[index];
    }
    diagnostics.push({
      mode: stroke.mode,
      selectedCells: grown.selected,
      softCells: grown.soft,
      unknownRadius: grown.unknownRadius,
      maximumDistance: grown.maximumDistance,
      targetAnchors: grown.targetAnchors,
      oppositeAnchors: grown.oppositeAnchors,
      componentCells: grown.componentCells,
    });
  }

  // Keep just the stroke centre as a hard contract. Treating the whole visible
  // brush disk as hard FG/BG is what caused the old round holes and blobs.
  const hard = strokeHintMasks(geometry, strokes, SMART_SEED_CORE_SCALE);
  const refined = guidedRefineHints(
    image,
    baseAlpha,
    geometry.size,
    keep,
    remove,
    hard.keep,
    hard.remove,
  );
  return {
    ...refined,
    hardKeep: hard.keep,
    hardRemove: hard.remove,
    diagnostics,
    grid,
  };
}

export function applyHintMasksToGuidance(baseAlpha, hints) {
  const result = new Float32Array(baseAlpha.length);
  for (let index = 0; index < result.length; index++) {
    result[index] = Math.min(1 - hints.remove[index], Math.max(hints.keep[index], baseAlpha[index]));
  }
  return result;
}

export function mergeGuidedAlpha(baseAlpha, refinedAlpha, hints) {
  const result = new Float32Array(baseAlpha.length);
  for (let index = 0; index < result.length; index++) {
    // Interactive correction must be local: the prototype update may improve
    // the selected material but must never rewrite already-correct, distant
    // parts of the matte. The smart hint's soft rim provides the merge seam.
    const keep = hints.keep[index];
    const remove = hints.remove[index];
    const update = Math.max(keep, remove);
    let proposed = Math.max(refinedAlpha[index], keep);
    proposed *= 1 - remove;
    result[index] = baseAlpha[index] * (1 - update) + proposed * update;
  }
  return result;
}

export function applyStrokesToGuidance(baseAlpha, geometry, strokes) {
  const result = new Float32Array(baseAlpha);
  for (const stroke of strokes) {
    // A wider influence field reaches the 1/8 prototype grid even for a thin
    // brush. The separate hard-hint maps below retain the exact brush radius.
    const radius = Math.max(1, stroke.radius * Math.min(geometry.resizedWidth, geometry.resizedHeight) * 1.5);
    const points = stroke.points;
    for (let index = 0; index < points.length; index++) {
      const from = points[Math.max(0, index - 1)];
      const to = points[index];
      const x0 = geometry.left + from.x * geometry.resizedWidth;
      const y0 = geometry.top + from.y * geometry.resizedHeight;
      const x1 = geometry.left + to.x * geometry.resizedWidth;
      const y1 = geometry.top + to.y * geometry.resizedHeight;
      const distance = Math.hypot(x1 - x0, y1 - y0);
      const steps = Math.max(1, Math.ceil(distance / Math.max(1, radius * 0.45)));
      for (let step = 0; step <= steps; step++) {
        const mix = step / steps;
        paintCircle(
          result,
          geometry.size,
          geometry.size,
          x0 + (x1 - x0) * mix,
          y0 + (y1 - y0) * mix,
          radius,
          strokeTarget(stroke.mode),
        );
      }
    }
  }
  return result;
}

export function strokeHintMasks(geometry, strokes, radiusScale = 1) {
  const keep = new Float32Array(geometry.size * geometry.size);
  const remove = new Float32Array(geometry.size * geometry.size);
  for (const stroke of strokes) {
    const radius = Math.max(
      1,
      stroke.radius * Math.min(geometry.resizedWidth, geometry.resizedHeight) * radiusScale,
    );
    const points = stroke.points;
    for (let index = 0; index < points.length; index++) {
      const from = points[Math.max(0, index - 1)];
      const to = points[index];
      const x0 = geometry.left + from.x * geometry.resizedWidth;
      const y0 = geometry.top + from.y * geometry.resizedHeight;
      const x1 = geometry.left + to.x * geometry.resizedWidth;
      const y1 = geometry.top + to.y * geometry.resizedHeight;
      const distance = Math.hypot(x1 - x0, y1 - y0);
      const steps = Math.max(1, Math.ceil(distance / Math.max(1, radius * 0.45)));
      for (let step = 0; step <= steps; step++) {
        const mix = step / steps;
        const x = x0 + (x1 - x0) * mix;
        const y = y0 + (y1 - y0) * mix;
        if (stroke.mode === "keep") {
          paintCircle(keep, geometry.size, geometry.size, x, y, radius, 1);
          paintCircle(remove, geometry.size, geometry.size, x, y, radius, 0);
        } else {
          paintCircle(remove, geometry.size, geometry.size, x, y, radius, 1);
          paintCircle(keep, geometry.size, geometry.size, x, y, radius, 0);
        }
      }
    }
  }
  return { keep, remove };
}

export function cropAlpha(alpha, geometry) {
  const cropped = new Float32Array(geometry.resizedWidth * geometry.resizedHeight);
  for (let y = 0; y < geometry.resizedHeight; y++) {
    const sourceOffset = (geometry.top + y) * geometry.size + geometry.left;
    cropped.set(alpha.subarray(sourceOffset, sourceOffset + geometry.resizedWidth), y * geometry.resizedWidth);
  }
  return cropped;
}

export function alphaCanvas(alpha, width, height) {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext("2d", { willReadFrequently: true });
  const image = context.createImageData(width, height);
  for (let pixel = 0, index = 0; pixel < alpha.length; pixel++, index += 4) {
    const value = Math.round(Math.min(1, Math.max(0, alpha[pixel])) * 255);
    image.data[index] = image.data[index + 1] = image.data[index + 2] = value;
    image.data[index + 3] = 255;
  }
  context.putImageData(image, 0, 0);
  return canvas;
}

export class PicornBrowserRuntime {
  constructor(onProgress) {
    this.onProgress = onProgress;
    this.session = null;
    this.ort = null;
    this.backend = null;
    this.initializing = null;
    this.hybridRefiner = new HybridAlphaRefiner({
      onFallback: (error) => this.onProgress?.("refine-fallback", String(error)),
    });
  }

  async initialize() {
    if (this.session) return this;
    if (this.initializing) return this.initializing;
    this.initializing = this.#initialize();
    try {
      return await this.initializing;
    } finally {
      this.initializing = null;
    }
  }

  async #initialize() {
    const wantsWebGpu = Boolean(navigator.gpu);
    this.onProgress?.("runtime", wantsWebGpu ? "WebGPU" : "WASM");
    const supportsJspi = typeof WebAssembly.Suspending === "function";
    const runtimeModule = wantsWebGpu
      ? supportsJspi
        ? "vendor/ort.jspi.bundle.min.mjs"
        : "vendor/ort.webgpu.bundle.min.mjs"
      : "vendor/ort.wasm.bundle.min.mjs";
    // The minified upstream bundle is copied as a static vendor asset. Vite
    // must not transform the 15-24 MB runtime during the first page request.
    this.ort = await import(/* @vite-ignore */ assetUrl(runtimeModule));
    this.ort.env.wasm.numThreads = globalThis.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
    const bytes = await cachedModel(assetUrl(MODEL_PATH), this.onProgress);
    const options = {
      executionProviders: wantsWebGpu ? ["webgpu", "wasm"] : ["wasm"],
      graphOptimizationLevel: "all",
    };
    try {
      this.session = await this.ort.InferenceSession.create(bytes, options);
      this.backend = wantsWebGpu ? "WebGPU + WASM" : "WASM CPU";
    } catch (error) {
      if (!wantsWebGpu) throw error;
      this.onProgress?.("fallback", String(error));
      this.session = await this.ort.InferenceSession.create(bytes, {
        executionProviders: ["wasm"],
        graphOptimizationLevel: "all",
      });
      this.backend = "WASM CPU";
    }
    this.onProgress?.("ready", this.backend);
    return this;
  }

  async infer(source, size, baseAlpha = null, strokes = []) {
    await this.initialize();
    const preprocessStarted = performance.now();
    const { data, geometry } = imageTensorData(source, size);
    const hints = baseAlpha && strokes.length
      ? smartStrokeHintMasks(data, baseAlpha, geometry, strokes)
      : strokeHintMasks(geometry, strokes);
    const guidance = baseAlpha && strokes.length
      ? applyHintMasksToGuidance(baseAlpha, hints)
      : new Float32Array(size * size);
    const weight = baseAlpha && strokes.length ? 1 : 0;
    const feeds = {
      image: new this.ort.Tensor("float32", data, [1, 3, size, size]),
      guidance: new this.ort.Tensor("float32", guidance, [1, 1, size, size]),
      guidance_weight: new this.ort.Tensor("float32", new Float32Array([weight]), [1]),
      // Only the pixels actually painted by the user are hard constraints.
      // The automatically grown area remains a soft proposal for guidance and
      // local merging, so a graph mistake cannot erase or restore an object.
      keep_hint: new this.ort.Tensor("float32", hints.hardKeep || hints.keep, [1, 1, size, size]),
      remove_hint: new this.ort.Tensor("float32", hints.hardRemove || hints.remove, [1, 1, size, size]),
    };
    const preprocessMilliseconds = performance.now() - preprocessStarted;
    const modelStarted = performance.now();
    const outputs = await this.session.run(feeds);
    const modelMilliseconds = performance.now() - modelStarted;
    const modelAlpha = new Float32Array(outputs.alpha.data);
    Object.values(feeds).forEach((tensor) => tensor.dispose?.());
    outputs.alpha.dispose?.();
    let hybrid = null;
    const rawAlpha = baseAlpha && strokes.length
      ? mergeGuidedAlpha(baseAlpha, modelAlpha, hints)
      : (hybrid = await this.hybridRefiner.refine(data, modelAlpha, size)).alpha;
    const cropped = cropAlpha(rawAlpha, geometry);
    return {
      rawAlpha,
      cropped,
      geometry,
      modelMilliseconds,
      preprocessMilliseconds,
      postprocessMilliseconds: hybrid?.milliseconds || 0,
      backend: this.backend,
      refinementBackend: hybrid?.backend || "interactive",
      refinementVersion: hybrid?.version || null,
      guided: strokes.length > 0,
      smartSelection: hints.diagnostics || [],
    };
  }
}
