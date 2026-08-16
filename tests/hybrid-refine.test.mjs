import assert from "node:assert/strict";
import { test } from "node:test";

import {
  HYBRID_REFINER_VERSION,
  edgeAwareJacobiCpu,
  fastGuidedTopologyCpu,
  refineAlphaHybridCpu,
} from "../src/hybrid-refine.js";

function syntheticComposite(size = 64) {
  const pixels = size * size;
  const image = new Float32Array(pixels * 3);
  const alpha = new Float32Array(pixels);
  const target = new Float32Array(pixels);
  for (let y = 0; y < size; y += 1) {
    for (let x = 0; x < size; x += 1) {
      const index = y * size + x;
      target[index] = Math.max(0, Math.min(1, (x - (size / 2 - 1)) / 3));
      alpha[index] = Math.max(0, Math.min(1, (x - (size / 2 - 3)) / 10));
      for (let channel = 0; channel < 3; channel += 1) {
        image[channel * pixels + index] = target[index];
      }
    }
  }
  return { image, alpha, target };
}

function meanAbsoluteError(actual, expected) {
  let total = 0;
  for (let index = 0; index < actual.length; index += 1) {
    total += Math.abs(actual[index] - expected[index]);
  }
  return total / actual.length;
}

test("the hybrid refiner has an explicit versioned recipe", () => {
  assert.equal(HYBRID_REFINER_VERSION, "v10-j8-topology-s75");
});
test("Jacobi keeps confident neural anchors bit-identical", () => {
  const { image, alpha } = syntheticComposite();
  const refined = edgeAwareJacobiCpu(image, alpha, 64, 64);
  for (let index = 0; index < alpha.length; index += 1) {
    if (alpha[index] <= 0.06 || alpha[index] >= 0.94) {
      assert.equal(refined[index], alpha[index]);
    }
  }
});

test("the complete hybrid improves a broad shifted edge", () => {
  const { image, alpha, target } = syntheticComposite();
  const refined = refineAlphaHybridCpu(image, alpha, 64, 64);
  assert.ok(meanAbsoluteError(refined, target) < meanAbsoluteError(alpha, target));
  assert.ok(refined.every((value) => Number.isFinite(value) && value >= 0 && value <= 1));
});

test("the guided stage cannot undo Jacobi's binary boundary", () => {
  const { image, alpha } = syntheticComposite();
  const jacobi = edgeAwareJacobiCpu(image, alpha, 64, 64);
  const refined = fastGuidedTopologyCpu(image, jacobi, 64, 64);
  for (let index = 0; index < alpha.length; index += 1) {
    assert.equal(refined[index] >= 0.5, jacobi[index] >= 0.5);
  }
});

test("topology protection never erodes a thin foreground without a safe core", () => {
  const size = 64;
  const pixels = size * size;
  const image = new Float32Array(pixels * 3).fill(0.1);
  const alpha = new Float32Array(pixels);
  for (let y = 0; y < size; y += 1) {
    for (let x = 30; x <= 32; x += 1) {
      const index = y * size + x;
      alpha[index] = 0.72;
      image[index] = image[pixels + index] = image[pixels * 2 + index] = 0.9;
    }
  }
  const refined = fastGuidedTopologyCpu(image, alpha, size, size);
  for (let y = 0; y < size; y += 1) {
    for (let x = 30; x <= 32; x += 1) {
      const index = y * size + x;
      assert.ok(refined[index] >= alpha[index]);
    }
  }
});
