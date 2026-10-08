import { test } from "node:test";
import assert from "node:assert/strict";
import { boxBlur, decontaminateForeground } from "../src/foreground.js";
import { composeZoomedAlpha, foregroundZoomRegion, snapConfidentAlpha } from "../src/browser-runtime.js";

test("box blur keeps a constant image constant and preserves the mean", () => {
  const values = new Float32Array(7 * 5).fill(0.25);
  const blurred = boxBlur(values, 7, 5, 1, 3);
  assert.ok(blurred.every((value) => Math.abs(value - 0.25) < 1e-6));
});

test("decontamination recovers the foreground colour of soft edge pixels", () => {
  // Red subject on a green ground with a 6 px soft ramp in between.
  const width = 512, height = 32;
  const alpha = new Float32Array(width * height);
  const rgba = new Uint8ClampedArray(width * height * 4);
  const foreground = [220, 30, 40], background = [20, 200, 60];
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const p = y * width + x;
      const a = Math.min(1, Math.max(0, (x - 250) / 6));
      alpha[p] = a;
      for (let c = 0; c < 3; c++) rgba[p * 4 + c] = Math.round(a * foreground[c] + (1 - a) * background[c]);
      rgba[p * 4 + 3] = 255;
    }
  }
  const naive = [];
  for (let x = 251; x < 256; x++) naive.push(rgba[(16 * width + x) * 4 + 1]);
  const changed = decontaminateForeground(rgba, alpha, width, height);
  assert.ok(changed > 0);
  for (let x = 251; x < 256; x++) {
    const p = 16 * width + x;
    // The green channel of the old ground must be largely gone from the subject colour.
    assert.ok(rgba[p * 4 + 1] < naive[x - 251] - 20 || rgba[p * 4 + 1] < 80, `x=${x} green=${rgba[p * 4 + 1]}`);
    assert.equal(rgba[p * 4 + 3], Math.round(alpha[p] * 255));
  }
  // Opaque and empty pixels keep their colour.
  assert.deepEqual([...rgba.subarray((16 * width + 400) * 4, (16 * width + 400) * 4 + 3)], foreground);
});

test("zoom region frames the subject and skips frame-filling subjects", () => {
  const width = 100, height = 50;
  const alpha = new Float32Array(width * height);
  for (let y = 20; y < 30; y++) for (let x = 40; x < 60; x++) alpha[y * width + x] = 1;
  const region = foregroundZoomRegion(alpha, width, height, 1000, 500);
  assert.ok(region.x < 400 && region.x + region.width > 600);
  assert.ok(region.y < 200 && region.y + region.height > 300);
  assert.ok(region.width * region.height < 0.8 * 1000 * 500);
  assert.equal(foregroundZoomRegion(new Float32Array(width * height).fill(1), width, height, 1000, 500), null);
  assert.equal(foregroundZoomRegion(new Float32Array(width * height), width, height, 1000, 500), null);
});

test("zoomed matte is placed at the region and zero outside", () => {
  const region = { x: 400, y: 100, width: 200, height: 100 };
  const cropGeometry = { size: 512, left: 0, top: 128, resizedWidth: 512, resizedHeight: 256 };
  const crop = new Float32Array(512 * 256).fill(1);
  const zoomed = composeZoomedAlpha(crop, cropGeometry, region, 1000, 500);
  assert.equal(zoomed.width, 2560);
  assert.equal(zoomed.height, 1280);
  assert.equal(zoomed.alpha[0], 0);
  assert.equal(zoomed.alpha[Math.round(150 * 2.56) * zoomed.width + Math.round(500 * 2.56)], 1);
});

test("confident interior haze and background specks snap, soft edges stay", () => {
  const width = 400, height = 400;
  const alpha = new Float32Array(width * height);
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
    const p = y * width + x;
    const inside = x >= 100 && x < 300 && y >= 100 && y < 300;
    alpha[p] = inside ? 0.93 : 0.08;
    if (x === 100 || x === 299) alpha[p] = 0.5; // a genuine soft edge column
  }
  const snapped = snapConfidentAlpha(alpha, width, height);
  assert.equal(snapped[200 * width + 200], 1);
  assert.equal(snapped[20 * width + 20], 0);
  assert.equal(snapped[200 * width + 100], 0.5);
  assert.ok(Math.abs(snapped[200 * width + 102] - 0.93) < 1e-6, "pixels next to the contour keep their alpha");
});
