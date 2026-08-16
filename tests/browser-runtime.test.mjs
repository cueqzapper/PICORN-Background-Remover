import test from "node:test";
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import {
  applyStrokesToGuidance,
  applyHintMasksToGuidance,
  cropAlpha,
  letterboxGeometry,
  mergeGuidedAlpha,
  smartStrokeHintMasks,
  strokeHintMasks,
} from "../src/browser-runtime.js";

test("the demo ships the versioned PICORN V9 model", async () => {
  const model = await readFile(new URL(
    "../public/models/picorn-remove-background-v9.onnx",
    import.meta.url,
  ));
  assert.equal(model.byteLength, 1_417_388);
  assert.equal(
    createHash("sha256").update(model).digest("hex"),
    "fb2d32ee2c07c9b7bc8f2adc2b5947d11c901c4bb2a8f1d23859c5107bc01295",
  );
});

test("letterbox geometry matches the Python preprocessing", () => {
  assert.deepEqual(letterboxGeometry(1000, 500, 320), {
    size: 320,
    left: 0,
    top: 80,
    resizedWidth: 320,
    resizedHeight: 160,
  });
});

test("green and red strokes become neural guidance", () => {
  const geometry = letterboxGeometry(100, 100, 32);
  const base = new Float32Array(32 * 32).fill(0.5);
  const strokes = [
    { mode: "keep", radius: 0.1, points: [{ x: 0.25, y: 0.5 }] },
    { mode: "remove", radius: 0.1, points: [{ x: 0.75, y: 0.5 }] },
  ];
  const guidance = applyStrokesToGuidance(base, geometry, strokes);
  assert.equal(guidance[16 * 32 + 8], 1);
  assert.equal(guidance[16 * 32 + 24], 0);
  assert.equal(guidance[0], 0.5);
});

test("the latest overlapping brush wins in ONNX hint channels", () => {
  const geometry = letterboxGeometry(100, 100, 32);
  const point = { x: 0.5, y: 0.5 };
  const hints = strokeHintMasks(geometry, [
    { mode: "keep", radius: 0.1, points: [point] },
    { mode: "remove", radius: 0.1, points: [point] },
  ]);
  const index = 16 * 32 + 16;
  assert.equal(hints.keep[index], 0);
  assert.equal(hints.remove[index], 1);
});

test("alpha crop removes letterbox padding", () => {
  const geometry = letterboxGeometry(100, 50, 8);
  const alpha = new Float32Array(64);
  alpha.fill(0.25);
  alpha.fill(0.75, geometry.top * 8, (geometry.top + geometry.resizedHeight) * 8);
  const cropped = cropAlpha(alpha, geometry);
  assert.equal(cropped.length, 32);
  assert.ok(cropped.every((value) => value === 0.75));
});

function twoToneImage(size, split) {
  const plane = size * size;
  const image = new Float32Array(plane * 3);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const value = x < split ? 0.15 : 0.85;
      const index = y * size + x;
      image[index] = value;
      image[plane + index] = value * 0.9;
      image[plane * 2 + index] = value * 0.8;
    }
  }
  return image;
}

test("a trimap correction remains bounded when the image has no useful edge", () => {
  const size = 64;
  const geometry = letterboxGeometry(size, size, size);
  const base = new Float32Array(size * size).fill(0);
  const hints = smartStrokeHintMasks(twoToneImage(size, 32), base, geometry, [
    { mode: "keep", radius: 0.025, points: [{ x: 0.2, y: 0.5 }] },
  ], 64);
  assert.ok(hints.keep[32 * size + 17] > 0.05, "selection should grow softly beyond the brush");
  assert.ok(hints.keep[32 * size + 27] < 0.05, "single click must stay local even without an edge");
  assert.ok(hints.keep[32 * size + 48] < 0.05, "strong colour edge should stop selection");
  assert.ok(hints.diagnostics[0].selectedCells > 20, "selection should be larger than brush seed");
  assert.ok(hints.diagnostics[0].selectedCells < 500, "selection must be bounded");
  assert.equal(hints.hardKeep[32 * size + 27], 0, "grown proposal must not become a hard ONNX lock");
});

test("a remove trimap also remains bounded on a uniform false-positive component", () => {
  const size = 64;
  const geometry = letterboxGeometry(size, size, size);
  const base = new Float32Array(size * size).fill(1);
  const hints = smartStrokeHintMasks(twoToneImage(size, 32), base, geometry, [
    { mode: "remove", radius: 0.025, points: [{ x: 0.8, y: 0.5 }] },
  ], 64);
  const guidance = applyHintMasksToGuidance(base, hints);
  assert.ok(hints.remove[32 * size + 46] > 0.05);
  assert.ok(hints.remove[32 * size + 34] < 0.05);
  assert.ok(hints.remove[32 * size + 16] < 0.05);
  assert.ok(guidance[32 * size + 46] < 0.95);
  assert.ok(guidance[32 * size + 16] > 0.95);
});

test("red scribble removes a shifted mask edge and stops at the real image edge", () => {
  const size = 64;
  const plane = size * size;
  const image = new Float32Array(plane * 3);
  const base = new Float32Array(plane);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const index = y * size + x;
      const value = x < 32 ? 0.1 : 0.9;
      image[index] = image[plane + index] = image[plane * 2 + index] = value;
      base[index] = x >= 28 ? 1 : 0;
    }
  }
  const hints = smartStrokeHintMasks(image, base, letterboxGeometry(size, size, size), [
    { mode: "remove", radius: 0.012, points: [{ x: 30 / size, y: 0.5 }] },
  ], 64);
  assert.ok(hints.remove[32 * size + 28] > 0.45, "false foreground should be removed beyond the brush");
  assert.ok(hints.remove[32 * size + 32] < 0.05, "the real colour edge should protect foreground");
  assert.equal(hints.hardRemove[32 * size + 28], 0, "automatic edge correction remains soft");
  assert.ok(hints.diagnostics[0].softCells > 20, "trimap must contain a real soft unknown band");
});

test("green scribble restores a missing mask edge with a soft alpha transition", () => {
  const size = 64;
  const plane = size * size;
  const image = new Float32Array(plane * 3);
  const base = new Float32Array(plane);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const index = y * size + x;
      const value = x < 32 ? 0.1 : 0.9;
      image[index] = image[plane + index] = image[plane * 2 + index] = value;
      base[index] = x >= 36 ? 1 : 0;
    }
  }
  const hints = smartStrokeHintMasks(image, base, letterboxGeometry(size, size, size), [
    { mode: "keep", radius: 0.012, points: [{ x: 34 / size, y: 0.5 }] },
  ], 64);
  const edge = hints.keep[32 * size + 32];
  assert.ok(edge > 0.25 && edge < 0.8, "guided matting should create partial alpha at the image edge");
  assert.ok(hints.keep[32 * size + 31] < 0.05, "background side should stay protected");
  assert.equal(hints.keep[32 * size + 36], 0, "already-correct foreground needs no correction");
});

test("a red hint removes a whole detached mask island without touching the main subject", () => {
  const size = 64;
  const geometry = letterboxGeometry(size, size, size);
  const image = twoToneImage(size, 32);
  const base = new Float32Array(size * size);
  for (let y = 22; y <= 40; y++) {
    for (let x = 4; x <= 20; x++) base[y * size + x] = 1;
    for (let x = 35; x < size; x++) base[y * size + x] = 1;
  }
  const hints = smartStrokeHintMasks(image, base, geometry, [
    { mode: "remove", radius: 0.01, points: [{ x: 10 / size, y: 31 / size }] },
  ], 64);
  assert.ok(hints.remove[31 * size + 19] > 0.95, "the complete detached island should be selected");
  assert.equal(hints.remove[31 * size + 36], 0, "the separate main subject must stay protected");
  assert.equal(hints.diagnostics[0].componentCells, 17 * 19);
});

test("guided inference cannot modify pixels outside the smart correction", () => {
  const base = new Float32Array([0.1, 0.2, 0.8, 0.9]);
  const refined = new Float32Array([1, 1, 0, 0]);
  const hints = {
    keep: new Float32Array([0, 1, 0, 0]),
    remove: new Float32Array([0, 0, 0.5, 0]),
  };
  const merged = mergeGuidedAlpha(base, refined, hints);
  assert.ok(Math.abs(merged[0] - 0.1) < 1e-6);
  assert.equal(merged[1], 1);
  assert.ok(Math.abs(merged[2] - 0.4) < 1e-6);
  assert.ok(Math.abs(merged[3] - 0.9) < 1e-6);
});
