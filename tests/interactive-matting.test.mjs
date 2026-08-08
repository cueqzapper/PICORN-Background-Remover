import test from "node:test";
import assert from "node:assert/strict";
import { guidedFilter } from "../src/interactive-matting.js";

test("guided matting keeps a soft mask transition attached to a sharp image edge", () => {
  const width = 40;
  const height = 12;
  const guide = new Float32Array(width * height);
  const coarse = new Float32Array(width * height);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const index = y * width + x;
      guide[index] = x < 20 ? 0.08 : 0.92;
      coarse[index] = Math.min(1, Math.max(0, (x - 15) / 10));
    }
  }
  const matte = guidedFilter(guide, coarse, width, height, 4, 0.0025);
  const row = Math.floor(height / 2) * width;
  assert.ok(matte[row + 18] < 0.35, "background side should be pushed down");
  assert.ok(matte[row + 21] > 0.65, "foreground side should be pulled up");
  assert.ok(matte[row + 19] < matte[row + 20], "transition should follow the guide edge");
});
