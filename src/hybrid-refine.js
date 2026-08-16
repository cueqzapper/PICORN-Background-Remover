/**
 * PICORN V10 deterministic alpha refinement.
 *
 * The tiny neural network owns object semantics. This stage only adjusts its
 * uncertain alpha band with an edge-aware graph stencil and a topology-safe
 * fast guided filter. The CPU implementation is the reference and fallback
 * for browsers without WebGPU.
 */

export const HYBRID_REFINER_VERSION = "v10-j8-topology-s75";

export const HYBRID_CONFIG = Object.freeze({
  backgroundAnchor: 0.06,
  foregroundAnchor: 0.94,
  jacobiIterations: 8,
  jacobiColorSigma: 0.08,
  jacobiSmoothness: 1.25,
  jacobiDataFloor: 0.12,
  jacobiDataScale: 3,
  jacobiStrength: 0.65,
  guidedRadius: 8,
  guidedSubsample: 4,
  guidedEpsilon: 0.0025,
  guidedStrength: 0.75,
});

function validate(image, alpha, width, height) {
  const pixels = width * height;
  if (!(image instanceof Float32Array) || image.length !== pixels * 3) {
    throw new Error("PICORN hybrid refinement needs planar RGB float data.");
  }
  if (!(alpha instanceof Float32Array) || alpha.length !== pixels) {
    throw new Error("PICORN hybrid refinement needs one alpha value per pixel.");
  }
}

function uncertainty(alpha, config) {
  if (alpha <= config.backgroundAnchor || alpha >= config.foregroundAnchor) return 0;
  return Math.max(0, 4 * alpha * (1 - alpha));
}

function neighbourIndex(x, y, dx, dy, width, height) {
  const nx = Math.max(0, Math.min(width - 1, x + dx));
  const ny = Math.max(0, Math.min(height - 1, y + dy));
  return ny * width + nx;
}

function neighbourWeights(image, width, height, colorSigma) {
  const pixels = width * height;
  const result = new Float32Array(pixels * 4);
  const denominator = 2 * colorSigma * colorSigma;
  const directions = [[-1, 0], [1, 0], [0, -1], [0, 1]];
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const index = y * width + x;
      for (let direction = 0; direction < directions.length; direction += 1) {
        const [dx, dy] = directions[direction];
        const adjacent = neighbourIndex(x, y, dx, dy, width, height);
        let distance = 0;
        for (let channel = 0; channel < 3; channel += 1) {
          const difference = image[channel * pixels + index] - image[channel * pixels + adjacent];
          distance += difference * difference;
        }
        result[index * 4 + direction] = Math.exp(-(distance / 3) / denominator);
      }
    }
  }
  return result;
}

export function edgeAwareJacobiCpu(image, alpha, width, height, overrides = {}) {
  const config = { ...HYBRID_CONFIG, ...overrides };
  validate(image, alpha, width, height);
  const weights = neighbourWeights(image, width, height, config.jacobiColorSigma);
  const directions = [[-1, 0], [1, 0], [0, -1], [0, 1]];
  let current = new Float32Array(alpha);
  let next = new Float32Array(alpha.length);
  for (let iteration = 0; iteration < config.jacobiIterations; iteration += 1) {
    for (let y = 0; y < height; y += 1) {
      for (let x = 0; x < width; x += 1) {
        const index = y * width + x;
        const original = alpha[index];
        const transition = uncertainty(original, config);
        if (transition === 0) {
          next[index] = original;
          continue;
        }
        let weightedAlpha = 0;
        let weightSum = 0;
        for (let direction = 0; direction < directions.length; direction += 1) {
          const [dx, dy] = directions[direction];
          const weight = weights[index * 4 + direction];
          weightedAlpha += weight * current[neighbourIndex(x, y, dx, dy, width, height)];
          weightSum += weight;
        }
        const dataConfidence = config.jacobiDataFloor
          + config.jacobiDataScale * (1 - transition) ** 2;
        const candidate = (
          dataConfidence * original
          + config.jacobiSmoothness * weightedAlpha
        ) / Math.max(1e-6, dataConfidence + config.jacobiSmoothness * weightSum);
        next[index] = Math.max(0, Math.min(
          1,
          current[index] + transition * config.jacobiStrength * (candidate - current[index]),
        ));
      }
    }
    [current, next] = [next, current];
  }
  return current;
}

function integralBoxMean(values, width, height, radius) {
  const stride = width + 1;
  const integral = new Float64Array((width + 1) * (height + 1));
  for (let y = 0; y < height; y += 1) {
    let row = 0;
    for (let x = 0; x < width; x += 1) {
      row += values[y * width + x];
      integral[(y + 1) * stride + x + 1] = integral[y * stride + x + 1] + row;
    }
  }
  const result = new Float32Array(values.length);
  for (let y = 0; y < height; y += 1) {
    const top = Math.max(0, y - radius);
    const bottom = Math.min(height - 1, y + radius);
    for (let x = 0; x < width; x += 1) {
      const left = Math.max(0, x - radius);
      const right = Math.min(width - 1, x + radius);
      const sum = integral[(bottom + 1) * stride + right + 1]
        - integral[top * stride + right + 1]
        - integral[(bottom + 1) * stride + left]
        + integral[top * stride + left];
      result[y * width + x] = sum / ((right - left + 1) * (bottom - top + 1));
    }
  }
  return result;
}

function downsample(image, alpha, width, height, factor) {
  const lowWidth = Math.floor(width / factor);
  const lowHeight = Math.floor(height / factor);
  if (lowWidth < 1 || lowHeight < 1 || width % factor || height % factor) {
    throw new Error("PICORN hybrid refinement needs dimensions divisible by four.");
  }
  const pixels = width * height;
  const lowLuma = new Float32Array(lowWidth * lowHeight);
  const lowAlpha = new Float32Array(lowWidth * lowHeight);
  const samples = factor * factor;
  for (let lowY = 0; lowY < lowHeight; lowY += 1) {
    for (let lowX = 0; lowX < lowWidth; lowX += 1) {
      let luma = 0;
      let matte = 0;
      for (let dy = 0; dy < factor; dy += 1) {
        const y = lowY * factor + dy;
        for (let dx = 0; dx < factor; dx += 1) {
          const x = lowX * factor + dx;
          const index = y * width + x;
          luma += image[index] * 0.2126
            + image[pixels + index] * 0.7152
            + image[pixels * 2 + index] * 0.0722;
          matte += alpha[index];
        }
      }
      const lowIndex = lowY * lowWidth + lowX;
      lowLuma[lowIndex] = luma / samples;
      lowAlpha[lowIndex] = matte / samples;
    }
  }
  return { lowLuma, lowAlpha, lowWidth, lowHeight };
}

function localAnchorSupport(alpha, width, height, radius, config) {
  const foreground = new Uint8Array(alpha.length);
  const background = new Uint8Array(alpha.length);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      let maximum = 0;
      let minimum = 1;
      for (let dy = -radius; dy <= radius; dy += 1) {
        const sampleY = y + dy;
        if (sampleY < 0 || sampleY >= height) continue;
        for (let dx = -radius; dx <= radius; dx += 1) {
          const sampleX = x + dx;
          if (sampleX < 0 || sampleX >= width) continue;
          const value = alpha[sampleY * width + sampleX];
          maximum = Math.max(maximum, value);
          minimum = Math.min(minimum, value);
        }
      }
      const index = y * width + x;
      foreground[index] = maximum >= config.foregroundAnchor ? 1 : 0;
      background[index] = minimum <= config.backgroundAnchor ? 1 : 0;
    }
  }
  return { foreground, background };
}

function bilinearSample(values, width, height, x, y) {
  const x0Raw = Math.floor(x);
  const y0Raw = Math.floor(y);
  const x0 = Math.max(0, Math.min(width - 1, x0Raw));
  const y0 = Math.max(0, Math.min(height - 1, y0Raw));
  const x1 = Math.max(0, Math.min(width - 1, x0Raw + 1));
  const y1 = Math.max(0, Math.min(height - 1, y0Raw + 1));
  const tx = Math.max(0, Math.min(1, x - x0Raw));
  const ty = Math.max(0, Math.min(1, y - y0Raw));
  const top = values[y0 * width + x0] * (1 - tx) + values[y0 * width + x1] * tx;
  const bottom = values[y1 * width + x0] * (1 - tx) + values[y1 * width + x1] * tx;
  return top * (1 - ty) + bottom * ty;
}

export function fastGuidedTopologyCpu(image, alpha, width, height, overrides = {}) {
  const config = { ...HYBRID_CONFIG, ...overrides };
  validate(image, alpha, width, height);
  const factor = config.guidedSubsample;
  const { lowLuma, lowAlpha, lowWidth, lowHeight } = downsample(
    image,
    alpha,
    width,
    height,
    factor,
  );
  const radius = Math.max(1, Math.round(config.guidedRadius / factor));
  const meanGuide = integralBoxMean(lowLuma, lowWidth, lowHeight, radius);
  const meanAlpha = integralBoxMean(lowAlpha, lowWidth, lowHeight, radius);
  const guideSquared = Float32Array.from(lowLuma, (value) => value * value);
  const guideAlpha = Float32Array.from(lowLuma, (value, index) => value * lowAlpha[index]);
  const meanGuideSquared = integralBoxMean(guideSquared, lowWidth, lowHeight, radius);
  const meanGuideAlpha = integralBoxMean(guideAlpha, lowWidth, lowHeight, radius);
  const coefficient = new Float32Array(lowAlpha.length);
  const intercept = new Float32Array(lowAlpha.length);
  for (let index = 0; index < lowAlpha.length; index += 1) {
    const variance = meanGuideSquared[index] - meanGuide[index] ** 2;
    const covariance = meanGuideAlpha[index] - meanGuide[index] * meanAlpha[index];
    coefficient[index] = covariance / (variance + config.guidedEpsilon);
    intercept[index] = meanAlpha[index] - coefficient[index] * meanGuide[index];
  }
  const meanCoefficient = integralBoxMean(coefficient, lowWidth, lowHeight, radius);
  const meanIntercept = integralBoxMean(intercept, lowWidth, lowHeight, radius);
  const support = localAnchorSupport(lowAlpha, lowWidth, lowHeight, radius, config);
  const pixels = width * height;
  const result = new Float32Array(alpha.length);
  for (let y = 0; y < height; y += 1) {
    const lowY = ((y + 0.5) * lowHeight) / height - 0.5;
    const nearestY = Math.min(lowHeight - 1, Math.floor((y * lowHeight) / height));
    for (let x = 0; x < width; x += 1) {
      const index = y * width + x;
      const reference = alpha[index];
      const transition = uncertainty(reference, config);
      if (transition === 0) {
        result[index] = reference;
        continue;
      }
      const lowX = ((x + 0.5) * lowWidth) / width - 0.5;
      const nearestX = Math.min(lowWidth - 1, Math.floor((x * lowWidth) / width));
      const lowIndex = nearestY * lowWidth + nearestX;
      const luma = image[index] * 0.2126
        + image[pixels + index] * 0.7152
        + image[pixels * 2 + index] * 0.0722;
      let candidate = bilinearSample(meanCoefficient, lowWidth, lowHeight, lowX, lowY) * luma
        + bilinearSample(meanIntercept, lowWidth, lowHeight, lowX, lowY);
      // An uncertain class may only be eroded when a same-class safe core is
      // nearby. This protects spokes, hair, wires and small holes.
      if (reference >= 0.5 && !support.foreground[lowIndex]) {
        candidate = Math.max(candidate, reference);
      } else if (reference < 0.5 && !support.background[lowIndex]) {
        candidate = Math.min(candidate, reference);
      }
      candidate = Math.max(0, Math.min(1, candidate));
      let refined = reference + transition * config.guidedStrength * (candidate - reference);
      // The guided pass improves soft alpha but cannot undo Jacobi's boundary.
      refined = reference >= 0.5
        ? Math.max(0.5, refined)
        : Math.min(0.5 - 1e-6, refined);
      result[index] = Math.max(0, Math.min(1, refined));
    }
  }
  return result;
}

export function refineAlphaHybridCpu(image, alpha, width, height, overrides = {}) {
  const jacobi = edgeAwareJacobiCpu(image, alpha, width, height, overrides);
  return fastGuidedTopologyCpu(image, jacobi, width, height, overrides);
}
