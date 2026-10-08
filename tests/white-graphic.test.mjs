import test from 'node:test';
import assert from 'node:assert/strict';
import { refineWhiteGraphic } from '../src/white-graphic.js';

function fixture() {
  const width = 128, height = 96;
  const data = new Uint8ClampedArray(width * height * 4).fill(255);
  const neural = new Float32Array(width * height).fill(1);
  const rect = (x0, y0, x1, y1, color, a = 1) => {
    for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) {
      const p = y * width + x;
      data.set([...color, 255], p * 4); neural[p] = a;
    }
  };
  rect(24, 20, 72, 76, [20, 70, 120]);
  return { data, neural, rect, width, height, run() { return refineWhiteGraphic(data, width, height, neural); } };
}

test('white paper overrides a confidently wrong foreground mask; disconnected ink is recovered', () => {
  const f = fixture(); f.rect(85, 35, 94, 60, [30, 30, 30], 0);
  const r = f.run(); assert.ok(r);
  assert.equal(r.alpha[0], 0); assert.equal(r.alpha[40 * f.width + 88], 1);
});
test('enclosed paper with background evidence is removed, white artwork is preserved', () => {
  const f = fixture(); f.rect(30, 28, 43, 43, [255, 255, 255], 0.05);
  f.rect(50, 50, 64, 65, [255, 255, 255], 0.98);
  const r = f.run(); assert.equal(r.alpha[32 * f.width + 35], 0);
  assert.equal(r.alpha[55 * f.width + 55], 1);
  assert.deepEqual(r.diagnostics, { enclosedRemoved: 1, preserved: 1 });
});
test('uncertain enclosed white is retained', () => {
  const f = fixture(); f.rect(30, 28, 43, 43, [255, 255, 255], 0.5);
  assert.equal(f.run().alpha[32 * f.width + 35], 1);
});
test('pale solid ink remains opaque', () => {
  const f = fixture(); f.rect(85, 35, 98, 65, [232, 232, 232], 0);
  assert.equal(f.run().alpha[40 * f.width + 90], 1);
});
test('antialiased ink is unmatted from white and recomposites to the source', () => {
  const f = fixture(); f.rect(23, 20, 24, 76, [138, 163, 188]);
  const r = f.run(), p = 40 * f.width + 23;
  assert.ok(Math.abs(r.alpha[p] - 0.5) < 0.01);
  for (let c = 0; c < 3; c++) {
    assert.ok(Math.abs(r.foreground[p * 4 + c] * r.alpha[p] + 255 * (1 - r.alpha[p]) - f.data[p * 4 + c]) <= 1);
  }
});
test('textured photo-like subject on white is rejected', () => {
  const f = fixture();
  for (let y = 20; y < 76; y++) for (let x = 24; x < 72; x++) {
    f.data.set([(x * 17 + y * 7) % 220, (x * 31 + y * 13) % 220, (x * 11 + y * 29) % 220], (y * f.width + x) * 4);
  }
  assert.equal(f.run(), null);
});
test('transparent input and nonwhite border are rejected', () => {
  const f = fixture(); f.data[3] = 0;
  f.rect(0, 0, 128, 2, [10, 10, 10]); assert.equal(f.run(), null);
  const t = fixture(); t.data[(30 * t.width + 30) * 4 + 3] = 0; assert.equal(t.run(), null);
});
test('blank paper and invalid dimensions do not produce destructive empty results', () => {
  const f = fixture(); f.data.fill(255); assert.equal(f.run(), null);
  assert.throws(() => refineWhiteGraphic(f.data, 0, 96, f.neural), /dimensions/);
});
function colouredFixture(ground = [18, 40, 92]) {
  const width = 128, height = 96;
  const data = new Uint8ClampedArray(width * height * 4);
  for (let p = 0; p < width * height; p++) data.set([...ground, 255], p * 4);
  const neural = new Float32Array(width * height).fill(1);
  const rect = (x0, y0, x1, y1, color, a = 1) => {
    for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) {
      const p = y * width + x;
      data.set([...color, 255], p * 4); neural[p] = a;
    }
  };
  rect(24, 20, 72, 76, [250, 196, 30]);
  return { data, neural, rect, width, height, ground, run() { return refineWhiteGraphic(data, width, height, neural); } };
}

test('a flat coloured ground is removed like white paper; disconnected ink is recovered', () => {
  const f = colouredFixture(); f.rect(85, 35, 94, 60, [240, 240, 240], 0);
  const r = f.run(); assert.ok(r);
  assert.equal(r.alpha[0], 0); assert.equal(r.alpha[40 * f.width + 50], 1); assert.equal(r.alpha[40 * f.width + 88], 1);
});
test('antialiased ink is unmatted from a coloured ground and recomposites to the source', () => {
  const f = colouredFixture(); const mid = [134, 118, 61];
  f.rect(23, 20, 24, 76, mid);
  const r = f.run(), p = 40 * f.width + 23;
  assert.ok(Math.abs(r.alpha[p] - 0.5) < 0.02, `alpha ${r.alpha[p]}`);
  for (let c = 0; c < 3; c++) {
    assert.ok(Math.abs(r.foreground[p * 4 + c] * r.alpha[p] + f.ground[c] * (1 - r.alpha[p]) - f.data[p * 4 + c]) <= 1);
  }
});
test('textured photo-like subject on a coloured ground is rejected', () => {
  const f = colouredFixture();
  for (let y = 20; y < 76; y++) for (let x = 24; x < 72; x++) {
    f.data.set([(x * 17 + y * 7) % 220, (x * 31 + y * 13) % 220, (x * 11 + y * 29) % 220], (y * f.width + x) * 4);
  }
  assert.equal(f.run(), null);
});
test('a noisy (non-flat) coloured border is rejected', () => {
  const f = colouredFixture();
  for (let p = 0; p < f.width * f.height; p += 3) f.data[p * 4] = (p * 37) % 255;
  assert.equal(f.run(), null);
});
