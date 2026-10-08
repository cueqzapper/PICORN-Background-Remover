/** Conservative flat-ground graphics matting. No weights or dependencies.
 * The ground is white paper or, since 2026-10-07, any other flat colour
 * measured on the image border (a logo on navy, a sticker on mint, ...).
 * Enclosed ground-coloured artwork is ambiguous: only neural background
 * evidence may remove it. Border-connected ground can override a confidently
 * wrong model.
 */
function white(data, i) {
  return Math.min(data[i], data[i + 1], data[i + 2]) >= 247
    && Math.max(data[i], data[i + 1], data[i + 2]) - Math.min(data[i], data[i + 1], data[i + 2]) <= 6;
}

const WHITE = [255, 255, 255];
const COLOURED_GROUND_TOLERANCE = 10;

function borderSamples(width, height) {
  const samples = [];
  for (let t = 0; t < 64; t++) {
    const x = Math.floor(t * (width - 1) / 63), y = Math.floor(t * (height - 1) / 63);
    samples.push([x, (height - 1) * width + x, y * width, y * width + width - 1]);
  }
  return samples;
}

/** The flat ground colour and its membership test, or null if the border is not flat. */
export function detectGround(data, width, height) {
  const colours = [[], [], []];
  for (const points of borderSamples(width, height)) {
    for (const p of points) for (let c = 0; c < 3; c++) colours[c].push(data[p * 4 + c]);
  }
  const ground = colours.map(values => values.sort((a, b) => a - b)[values.length >> 1]);
  const probe = Uint8ClampedArray.from([...ground, 255]);
  if (white(probe, 0)) return { colour: WHITE, isGround: (d, i) => white(d, i), white: true };
  const isGround = (d, i) => Math.abs(d[i] - ground[0]) <= COLOURED_GROUND_TOLERANCE
    && Math.abs(d[i + 1] - ground[1]) <= COLOURED_GROUND_TOLERANCE
    && Math.abs(d[i + 2] - ground[2]) <= COLOURED_GROUND_TOLERANCE;
  return { colour: ground, isGround, white: false };
}

export function isWhiteGraphic(data, width, height, options = {}) {
  if (width < 8 || height < 8) return false;
  // Private production may explicitly confirm a white-paper logo after
  // inspecting the original. Thin antialiased type is not sufficiently flat
  // for the conservative automatic photo/graphic classifier below.
  if (options.verifiedWhitePaper === true) {
    if (data.length !== width * height * 4) return false;
    const corners = [0, width - 1, (height - 1) * width, width * height - 1];
    if (corners.some(p => !white(data, p * 4))) return false;
    let paper = 0, ink = 0;
    for (let p = 0; p < width * height; p++) {
      if (data[p * 4 + 3] !== 255) return false;
      if (white(data, p * 4)) paper++; else ink++;
    }
    return paper / (width * height) >= 0.2 && ink >= 16;
  }
  const ground = detectGround(data, width, height);
  const sides = [0, 0, 0, 0];
  for (const points of borderSamples(width, height)) {
    points.forEach((p, side) => {
      if (data[p * 4 + 3] === 255 && ground.isGround(data, p * 4)) sides[side]++;
    });
  }
  if (sides.some(count => count < 61)) return false;
  let samples = 0, paper = 0, ink = 0, flat = 0;
  const palette = new Map();
  const step = Math.max(1, Math.ceil(Math.sqrt(width * height / 65536)));
  for (let y = 1; y < height - 1; y += step) {
    for (let x = 1; x < width - 1; x += step) {
      const i = (y * width + x) * 4;
      if (data[i + 3] !== 255) return false;
      samples++;
      if (ground.isGround(data, i)) { paper++; continue; }
      ink++;
      const key = (data[i] >> 4) * 256 + (data[i + 1] >> 4) * 16 + (data[i + 2] >> 4);
      palette.set(key, (palette.get(key) || 0) + 1);
      if ([0, 1, 2].every(c => Math.abs(data[i + c] - data[i + 4 + c]) <= 8
        && Math.abs(data[i + c] - data[i + width * 4 + c]) <= 8)) flat++;
    }
  }
  const dominant = [...palette.values()].sort((a, b) => b - a).slice(0, 8).reduce((a, b) => a + b, 0);
  return paper / samples >= 0.25 && paper / samples <= 0.995
    && ink >= 16 && flat / ink >= 0.85 && dominant / ink >= 0.90;
}

function groundDistance(data, i, colour) {
  return Math.abs(data[i] - colour[0]) + Math.abs(data[i + 1] - colour[1]) + Math.abs(data[i + 2] - colour[2]);
}

export function refineWhiteGraphic(data, width, height, neuralAlpha, options = {}) {
  const pixels = width * height;
  if (!Number.isInteger(width) || !Number.isInteger(height) || width <= 0 || height <= 0
    || data.length !== pixels * 4 || neuralAlpha.length !== pixels) {
    throw new Error('White graphic refinement needs matching RGBA and alpha dimensions.');
  }
  if (!isWhiteGraphic(data, width, height, options)) return null;
  const ground = options.verifiedWhitePaper === true
    ? { colour: WHITE, isGround: (d, i) => white(d, i), white: true }
    : detectGround(data, width, height);
  const paperColour = ground.colour;
  // Lossy website logos often contain near-white compression ringing. In the
  // inspected white-paper mode only, include that paper halo in the same
  // connected-component treatment; coloured ink and enclosed white artwork
  // still follow the existing preservation rules.
  const isPaper = options.verifiedWhitePaper === true
    ? i => Math.min(data[i], data[i + 1], data[i + 2]) >= 235
      && Math.max(data[i], data[i + 1], data[i + 2]) - Math.min(data[i], data[i + 1], data[i + 2]) <= 20
    : i => ground.isGround(data, i);
  // The classifier samples large images; never overwrite transparency that
  // fell between its samples.
  for (let p = 0; p < pixels; p++) if (data[p * 4 + 3] !== 255) return null;
  const visited = new Uint8Array(pixels);
  const removed = new Uint8Array(pixels);
  const queue = new Int32Array(pixels);
  let enclosedRemoved = 0, preserved = 0;
  for (let start = 0; start < pixels; start++) {
    if (visited[start] || !isPaper(start * 4)) continue;
    let head = 0, tail = 1, border = false, backgroundVotes = 0;
    queue[0] = start; visited[start] = 1;
    while (head < tail) {
      const p = queue[head++], x = p % width, y = Math.floor(p / width);
      border ||= x === 0 || y === 0 || x === width - 1 || y === height - 1;
      if (neuralAlpha[p] < 0.2) backgroundVotes++;
      const visit = q => {
        if (!visited[q] && isPaper(q * 4)) { visited[q] = 1; queue[tail++] = q; }
      };
      if (x > 0) visit(p - 1);
      if (x + 1 < width) visit(p + 1);
      if (y > 0) visit(p - width);
      if (y + 1 < height) visit(p + width);
    }
    if (border || backgroundVotes / tail >= 0.8) {
      for (let n = 0; n < tail; n++) removed[queue[n]] = 1;
      if (!border) enclosedRemoved++;
    } else preserved++;
  }
  const alpha = new Float32Array(pixels).fill(1);
  const foreground = new Uint8ClampedArray(data);
  for (let p = 0; p < pixels; p++) {
    if (removed[p]) { alpha[p] = 0; continue; }
    const x = p % width, y = Math.floor(p / width), i = p * 4;
    // Solve C = alpha * F + (1-alpha) * white at a narrow paper edge.
    // A nearby darker ink sample supplies F. Interiors, including grey logos,
    // stay opaque; this is not a global luminance-to-alpha conversion.
    let nearPaper = false, best = p;
    const edgeRadius = options.verifiedWhitePaper === true ? 4 : 2;
    for (let dy = -edgeRadius; dy <= edgeRadius; dy++) {
      for (let dx = -edgeRadius; dx <= edgeRadius; dx++) {
        if (x + dx < 0 || x + dx >= width || y + dy < 0 || y + dy >= height) continue;
        const q = (y + dy) * width + x + dx;
        const distance = Math.abs(dx) + Math.abs(dy);
        if (distance > 0 && distance <= (options.verifiedWhitePaper === true ? 2 : 1) && removed[q]) nearPaper = true;
        if (!removed[q] && groundDistance(data, q * 4, paperColour) > groundDistance(data, best * 4, paperColour)) best = q;
      }
    }
    if (!nearPaper || best === p) continue;
    let numerator = 0, denominator = 0;
    for (let c = 0; c < 3; c++) {
      const delta = paperColour[c] - data[best * 4 + c];
      numerator += (paperColour[c] - data[i + c]) * delta; denominator += delta * delta;
    }
    const a = Math.min(1, numerator / Math.max(1, denominator));
    if (a < 0.02 || a > 0.98) continue;
    let residual = 0;
    for (let c = 0; c < 3; c++) residual = Math.max(residual,
      Math.abs(data[i + c] - (a * data[best * 4 + c] + (1 - a) * paperColour[c])));
    if (residual > (options.verifiedWhitePaper === true ? 18 : 3)) continue;
    alpha[p] = a;
    for (let c = 0; c < 3; c++) foreground[i + c] = (data[i + c] - (1 - a) * paperColour[c]) / a;
  }
  return { alpha, foreground, diagnostics: { enclosedRemoved, preserved } };
}

/** Canvas adapter shared by Studio and the standalone demo. Returns null for
 * photos/ambiguous inputs. Work is capped at 4 MP / 4096 px; model stays tiny. */
export function whiteGraphicCanvases(source, neuralMatte, options = {}) {
  const scale = Math.min(1, 4096 / Math.max(source.naturalWidth, source.naturalHeight),
    Math.sqrt(4_000_000 / (source.naturalWidth * source.naturalHeight)));
  const width = Math.max(1, Math.round(source.naturalWidth * scale));
  const height = Math.max(1, Math.round(source.naturalHeight * scale));
  const canvas = document.createElement('canvas'); canvas.width = width; canvas.height = height;
  const context = canvas.getContext('2d', { willReadFrequently: true });
  context.drawImage(source, 0, 0, width, height);
  const original = context.getImageData(0, 0, width, height);
  if (!isWhiteGraphic(original.data, width, height, options)) return null;
  context.clearRect(0, 0, width, height);
  context.drawImage(neuralMatte, 0, 0, width, height);
  const mask = context.getImageData(0, 0, width, height);
  const neural = Float32Array.from({ length: width * height }, (_, p) => mask.data[p * 4] / 255);
  const result = refineWhiteGraphic(original.data, width, height, neural, options);
  if (!result) return null;
  context.putImageData(new ImageData(result.foreground, width, height), 0, 0);
  const matte = document.createElement('canvas'); matte.width = width; matte.height = height;
  for (let p = 0; p < result.alpha.length; p++) {
    mask.data[p * 4] = mask.data[p * 4 + 1] = mask.data[p * 4 + 2] = Math.round(result.alpha[p] * 255);
    mask.data[p * 4 + 3] = 255;
  }
  matte.getContext('2d').putImageData(mask, 0, 0);
  return { matte, foreground: canvas, diagnostics: result.diagnostics };
}
