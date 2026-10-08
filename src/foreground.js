/**
 * Foreground colour estimation for soft matte edges ("decontamination").
 *
 * Without it, a semi-transparent hair or edge pixel keeps the colour of the
 * old background and shows a grey/green seam on the new one. This is the
 * blur-fusion estimator of Forte & Pitié, "Approximate Fast Foreground Colour
 * Estimation" (ICIP 2021), run twice (wide, then narrow radius).
 *
 * The blurred foreground/background estimates are smooth, so they are
 * computed on a working copy of at most WORK_SIDE px and only the final
 * per-pixel step  F = F̂ + α (I − α F̂ − (1 − α) B̂)  runs at full resolution,
 * and only where 0 < α < 1.
 */

const WORK_SIDE = 1024;

/** Separable box blur (running sums), clamped borders, `channels` interleaved. */
export function boxBlur(values, width, height, channels, radius) {
  if (radius < 1) return Float32Array.from(values);
  const temporary = new Float32Array(values.length);
  const result = new Float32Array(values.length);
  const window = radius * 2 + 1;
  for (let y = 0; y < height; y++) {
    const row = y * width;
    for (let c = 0; c < channels; c++) {
      let sum = 0;
      for (let k = -radius; k <= radius; k++) sum += values[(row + Math.min(width - 1, Math.max(0, k))) * channels + c];
      for (let x = 0; x < width; x++) {
        temporary[(row + x) * channels + c] = sum / window;
        const add = Math.min(width - 1, x + radius + 1);
        const remove = Math.max(0, x - radius);
        sum += values[(row + add) * channels + c] - values[(row + remove) * channels + c];
      }
    }
  }
  for (let x = 0; x < width; x++) {
    for (let c = 0; c < channels; c++) {
      let sum = 0;
      for (let k = -radius; k <= radius; k++) sum += temporary[(Math.min(height - 1, Math.max(0, k)) * width + x) * channels + c];
      for (let y = 0; y < height; y++) {
        result[(y * width + x) * channels + c] = sum / window;
        const add = Math.min(height - 1, y + radius + 1);
        const remove = Math.max(0, y - radius);
        sum += temporary[(add * width + x) * channels + c] - temporary[(remove * width + x) * channels + c];
      }
    }
  }
  return result;
}

function blurFusionStep(image, foreground, background, alpha, width, height, radius) {
  const pixels = width * height;
  const weightedForeground = new Float32Array(pixels * 3);
  const weightedBackground = new Float32Array(pixels * 3);
  for (let p = 0; p < pixels; p++) {
    for (let c = 0; c < 3; c++) {
      weightedForeground[p * 3 + c] = foreground[p * 3 + c] * alpha[p];
      weightedBackground[p * 3 + c] = background[p * 3 + c] * (1 - alpha[p]);
    }
  }
  const blurredAlpha = boxBlur(alpha, width, height, 1, radius);
  const blurredForegroundSum = boxBlur(weightedForeground, width, height, 3, radius);
  const blurredBackgroundSum = boxBlur(weightedBackground, width, height, 3, radius);
  const blurredForeground = new Float32Array(pixels * 3);
  const blurredBackground = new Float32Array(pixels * 3);
  const estimate = new Float32Array(pixels * 3);
  for (let p = 0; p < pixels; p++) {
    const a = alpha[p];
    for (let c = 0; c < 3; c++) {
      const i = p * 3 + c;
      const f = blurredForegroundSum[i] / (blurredAlpha[p] + 1e-5);
      const b = blurredBackgroundSum[i] / (1 - blurredAlpha[p] + 1e-5);
      blurredForeground[i] = f;
      blurredBackground[i] = b;
      estimate[i] = Math.min(1, Math.max(0, f + a * (image[i] - a * f - (1 - a) * b)));
    }
  }
  return { estimate, blurredForeground, blurredBackground };
}

/**
 * Two-pass blur fusion on a small image. `image` is RGB in [0, 1] (3 floats
 * per pixel), `alpha` one float per pixel. Returns the blurred F̂ and B̂ of the
 * final pass, which `applyForegroundEstimate` evaluates at full resolution.
 */
export function blurFusionEstimates(image, alpha, width, height) {
  const longSide = Math.max(width, height);
  const wide = Math.max(2, Math.round(longSide * 0.045));
  const narrow = Math.max(1, Math.round(longSide * 0.006));
  const first = blurFusionStep(image, image, image, alpha, width, height, wide);
  const second = blurFusionStep(image, first.estimate, first.blurredBackground, alpha, width, height, narrow);
  return { foreground: second.blurredForeground, background: second.blurredBackground };
}

function sample(values, width, height, x, y, c) {
  const x0 = Math.max(0, Math.min(width - 1, Math.floor(x)));
  const y0 = Math.max(0, Math.min(height - 1, Math.floor(y)));
  const x1 = Math.min(width - 1, x0 + 1);
  const y1 = Math.min(height - 1, y0 + 1);
  const tx = Math.max(0, Math.min(1, x - x0));
  const ty = Math.max(0, Math.min(1, y - y0));
  const top = values[(y0 * width + x0) * 3 + c] * (1 - tx) + values[(y0 * width + x1) * 3 + c] * tx;
  const bottom = values[(y1 * width + x0) * 3 + c] * (1 - tx) + values[(y1 * width + x1) * 3 + c] * tx;
  return top * (1 - ty) + bottom * ty;
}

/**
 * Rewrites RGB of `rgba` (full-resolution ImageData.data) where 0 < α < 1 and
 * writes α into the alpha channel. `alpha` holds one float per pixel.
 */
export function decontaminateForeground(rgba, alpha, width, height) {
  const scale = Math.min(1, WORK_SIDE / Math.max(width, height));
  const workWidth = Math.max(1, Math.round(width * scale));
  const workHeight = Math.max(1, Math.round(height * scale));
  const workImage = new Float32Array(workWidth * workHeight * 3);
  const workAlpha = new Float32Array(workWidth * workHeight);
  // Area-average downsample keeps thin soft structures in the estimates.
  const counts = new Float32Array(workWidth * workHeight);
  for (let y = 0; y < height; y++) {
    const wy = Math.min(workHeight - 1, Math.floor(y * scale));
    for (let x = 0; x < width; x++) {
      const wx = Math.min(workWidth - 1, Math.floor(x * scale));
      const w = wy * workWidth + wx;
      const p = y * width + x;
      counts[w] += 1;
      workAlpha[w] += alpha[p];
      for (let c = 0; c < 3; c++) workImage[w * 3 + c] += rgba[p * 4 + c] / 255;
    }
  }
  for (let w = 0; w < counts.length; w++) {
    const n = Math.max(1, counts[w]);
    workAlpha[w] /= n;
    for (let c = 0; c < 3; c++) workImage[w * 3 + c] /= n;
  }
  const estimates = blurFusionEstimates(workImage, workAlpha, workWidth, workHeight);
  let changed = 0;
  for (let y = 0; y < height; y++) {
    const sy = (y + 0.5) * scale - 0.5;
    for (let x = 0; x < width; x++) {
      const p = y * width + x;
      const a = Math.min(1, Math.max(0, alpha[p]));
      rgba[p * 4 + 3] = Math.round(a * 255);
      if (a <= 0.004 || a >= 0.996) continue;
      const sx = (x + 0.5) * scale - 0.5;
      for (let c = 0; c < 3; c++) {
        const f = sample(estimates.foreground, workWidth, workHeight, sx, sy, c);
        const b = sample(estimates.background, workWidth, workHeight, sx, sy, c);
        const value = f + a * (rgba[p * 4 + c] / 255 - a * f - (1 - a) * b);
        rgba[p * 4 + c] = Math.round(Math.min(1, Math.max(0, value)) * 255);
      }
      changed++;
    }
  }
  return changed;
}
