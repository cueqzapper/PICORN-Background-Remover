function clamp01(value) {
  return Math.min(1, Math.max(0, value));
}

class MinimumHeap {
  constructor() {
    this.indices = [];
    this.costs = [];
  }

  get length() {
    return this.indices.length;
  }

  push(index, cost) {
    let position = this.indices.length;
    this.indices.push(index);
    this.costs.push(cost);
    while (position > 0) {
      const parent = (position - 1) >> 1;
      if (this.costs[parent] <= cost) break;
      this.indices[position] = this.indices[parent];
      this.costs[position] = this.costs[parent];
      position = parent;
    }
    this.indices[position] = index;
    this.costs[position] = cost;
  }

  pop() {
    if (!this.indices.length) return null;
    const index = this.indices[0];
    const cost = this.costs[0];
    const lastIndex = this.indices.pop();
    const lastCost = this.costs.pop();
    if (this.indices.length) {
      let position = 0;
      while (true) {
        const left = position * 2 + 1;
        if (left >= this.indices.length) break;
        const right = left + 1;
        const child = right < this.indices.length && this.costs[right] < this.costs[left] ? right : left;
        if (this.costs[child] >= lastCost) break;
        this.indices[position] = this.indices[child];
        this.costs[position] = this.costs[child];
        position = child;
      }
      this.indices[position] = lastIndex;
      this.costs[position] = lastCost;
    }
    return { index, cost };
  }
}

const NEIGHBOURS = [
  [-1, 0, 1], [1, 0, 1], [0, -1, 1], [0, 1, 1],
  [-1, -1, Math.SQRT2], [1, -1, Math.SQRT2], [-1, 1, Math.SQRT2], [1, 1, Math.SQRT2],
];

function seedPrototype(features, seeds) {
  const prototype = { red: 0, green: 0, blue: 0, entropy: 0, alpha: 0, count: 0 };
  for (let index = 0; index < seeds.length; index++) {
    if (!seeds[index]) continue;
    prototype.count += 1;
    prototype.red += features.red[index];
    prototype.green += features.green[index];
    prototype.blue += features.blue[index];
    prototype.entropy += features.entropy[index];
    prototype.alpha += features.alpha[index];
  }
  if (!prototype.count) return prototype;
  for (const key of ["red", "green", "blue", "entropy", "alpha"]) prototype[key] /= prototype.count;
  return prototype;
}

function spatialDistance(grid, seeds, maximumDistance) {
  const distances = new Float32Array(seeds.length).fill(Number.POSITIVE_INFINITY);
  const heap = new MinimumHeap();
  for (let index = 0; index < seeds.length; index++) {
    if (!seeds[index]) continue;
    distances[index] = 0;
    heap.push(index, 0);
  }
  while (heap.length) {
    const current = heap.pop();
    if (!current || current.cost > distances[current.index] + 1e-6 || current.cost > maximumDistance) continue;
    const x = current.index % grid.width;
    const y = Math.floor(current.index / grid.width);
    for (const [dx, dy, step] of NEIGHBOURS) {
      const nx = x + dx;
      const ny = y + dy;
      if (nx < 0 || nx >= grid.width || ny < 0 || ny >= grid.height) continue;
      const next = ny * grid.width + nx;
      const candidate = current.cost + step;
      if (candidate >= distances[next] || candidate > maximumDistance) continue;
      distances[next] = candidate;
      heap.push(next, candidate);
    }
  }
  return distances;
}

/**
 * Detects a small, detached false-positive island or an enclosed mask hole.
 * These regions should be corrected as a whole; a large component is left to
 * the local geodesic solver so a click can never erase the main subject.
 */
function boundedWrongLabelComponent(alpha, grid, seeds, mode) {
  const candidate = mode === "keep"
    ? (value) => value <= 0.45
    : (value) => value >= 0.55;
  const maximumCells = Math.max(48, Math.round(alpha.length * 0.08));
  const queued = new Uint8Array(alpha.length);
  const component = new Uint8Array(alpha.length);
  const queue = [];
  for (let index = 0; index < seeds.length; index++) {
    if (!seeds[index] || !candidate(alpha[index]) || queued[index]) continue;
    queued[index] = 1;
    queue.push(index);
  }
  if (!queue.length) return { component, cells: 0 };

  let head = 0;
  while (head < queue.length) {
    const index = queue[head++];
    component[index] = 1;
    if (queue.length > maximumCells) return { component: null, cells: 0 };
    const x = index % grid.width;
    const y = Math.floor(index / grid.width);
    // Four-connectivity avoids merging diagonal wisps that merely touch.
    for (const [dx, dy] of NEIGHBOURS.slice(0, 4)) {
      const nx = x + dx;
      const ny = y + dy;
      if (nx < 0 || nx >= grid.width || ny < 0 || ny >= grid.height) continue;
      const next = ny * grid.width + nx;
      if (queued[next] || !candidate(alpha[next])) continue;
      queued[next] = 1;
      queue.push(next);
    }
  }
  return { component, cells: queue.length };
}

function geodesicDistance(features, grid, anchors, roi, prototype = null) {
  const distances = new Float32Array(anchors.length).fill(Number.POSITIVE_INFINITY);
  const heap = new MinimumHeap();
  for (let index = 0; index < anchors.length; index++) {
    if (!anchors[index]) continue;
    distances[index] = 0;
    heap.push(index, 0);
  }
  while (heap.length) {
    const current = heap.pop();
    if (!current || current.cost > distances[current.index] + 1e-6) continue;
    const x = current.index % grid.width;
    const y = Math.floor(current.index / grid.width);
    for (const [dx, dy, step] of NEIGHBOURS) {
      const nx = x + dx;
      const ny = y + dy;
      if (nx < 0 || nx >= grid.width || ny < 0 || ny >= grid.height) continue;
      const next = ny * grid.width + nx;
      if (!roi[next]) continue;
      const colourEdge = Math.sqrt(
        (features.red[current.index] - features.red[next]) ** 2
        + (features.green[current.index] - features.green[next]) ** 2
        + (features.blue[current.index] - features.blue[next]) ** 2,
      ) / Math.sqrt(3);
      const entropyEdge = Math.abs(features.entropy[current.index] - features.entropy[next]);
      const alphaEdge = Math.abs(features.alpha[current.index] - features.alpha[next]);
      let appearance = 0;
      if (prototype?.count) {
        appearance = Math.sqrt(
          (features.red[next] - prototype.red) ** 2
          + (features.green[next] - prototype.green) ** 2
          + (features.blue[next] - prototype.blue) ** 2,
        ) / Math.sqrt(3);
      }
      const edgeCost = 0.018
        + colourEdge * 2.25
        + entropyEdge * 0.28
        + features.gradient[next] * 0.22
        + alphaEdge * 0.20
        + appearance * 0.12;
      const candidate = current.cost + edgeCost * step;
      if (candidate >= distances[next]) continue;
      distances[next] = candidate;
      heap.push(next, candidate);
    }
  }
  return distances;
}

/**
 * Builds a local trimap from the current alpha and the user's scribble, then
 * solves its unknown band with foreground/background geodesic distances.
 */
export function geodesicTrimapCorrection(features, grid, userSeeds, mode, brushRadius) {
  const desiredLabel = mode === "keep" ? 1 : 0;
  const unknownRadius = Math.min(
    Math.max(grid.width, grid.height) * 0.105,
    Math.max(5, brushRadius * 2.2 + 3),
  );
  const maximumDistance = Math.min(
    Math.max(grid.width, grid.height) * 0.16,
    Math.max(unknownRadius + 6, 10),
  );
  const distanceToBrush = spatialDistance(grid, userSeeds, maximumDistance);
  const roi = new Uint8Array(userSeeds.length);
  const targetAnchors = new Uint8Array(userSeeds.length);
  const oppositeAnchors = new Uint8Array(userSeeds.length);
  let targetCount = 0;
  let oppositeCount = 0;

  for (let index = 0; index < userSeeds.length; index++) {
    if (!Number.isFinite(distanceToBrush[index]) || distanceToBrush[index] > maximumDistance) continue;
    roi[index] = 1;
    if (userSeeds[index]) {
      targetAnchors[index] = 1;
      targetCount += 1;
      continue;
    }
    if (distanceToBrush[index] < unknownRadius) continue;
    const alpha = features.alpha[index];
    const currentLabel = alpha >= 0.5 ? 1 : 0;
    const confident = alpha >= 0.82 || alpha <= 0.18;
    // The outer ring is the known part of the trimap. Confident pixels keep
    // their previous label; uncertain ones still contain the correction by
    // using the current binary side as a weak anchor.
    if (currentLabel === desiredLabel && confident) {
      targetAnchors[index] = 1;
      targetCount += 1;
    } else if (currentLabel !== desiredLabel || distanceToBrush[index] > unknownRadius + 2) {
      oppositeAnchors[index] = 1;
      oppositeCount += 1;
    }
  }

  // A correction with no opposite class would be unconstrained. Anchor the
  // farthest ROI ring to the previous mask instead of flooding the component.
  if (!oppositeCount) {
    for (let index = 0; index < roi.length; index++) {
      if (!roi[index] || distanceToBrush[index] < maximumDistance - 1.5) continue;
      oppositeAnchors[index] = 1;
      oppositeCount += 1;
    }
  }

  const userPrototype = seedPrototype(features, userSeeds);
  const oppositePrototype = seedPrototype(features, oppositeAnchors);
  const topology = boundedWrongLabelComponent(features.alpha, grid, userSeeds, mode);
  const targetDistance = geodesicDistance(features, grid, targetAnchors, roi, userPrototype);
  const oppositeDistance = geodesicDistance(features, grid, oppositeAnchors, roi, oppositePrototype);
  const influence = new Float32Array(userSeeds.length);
  let selected = 0;
  let soft = 0;
  for (let index = 0; index < influence.length; index++) {
    let value = 0;
    // Small disconnected islands and holes are a single topological mistake,
    // not a circular brush mark. Correct the full component, then let guided
    // matting soften its real image boundary on the full-resolution grid.
    if (topology.component?.[index]) {
      value = 1;
    } else {
      if (!roi[index]) continue;
      const target = targetDistance[index];
      const opposite = oppositeDistance[index];
      if (!Number.isFinite(target) || !Number.isFinite(opposite)) continue;
      const targetProbability = opposite / Math.max(1e-6, target + opposite);
      const solvedAlpha = desiredLabel ? targetProbability : 1 - targetProbability;
      const currentAlpha = features.alpha[index];
      value = desiredLabel
        ? Math.max(0, (solvedAlpha - currentAlpha) / Math.max(1e-4, 1 - currentAlpha))
        : Math.max(0, (currentAlpha - solvedAlpha) / Math.max(1e-4, currentAlpha));
      const spatial = clamp01(1 - distanceToBrush[index] / maximumDistance);
      value *= 0.35 + spatial * 0.65;
      if (userSeeds[index]) value = 1;
    }
    influence[index] = clamp01(value);
    if (influence[index] > 0.04) selected += 1;
    if (influence[index] > 0.05 && influence[index] < 0.95) soft += 1;
  }
  return {
    influence,
    selected,
    soft,
    unknownRadius,
    maximumDistance,
    targetAnchors: targetCount,
    oppositeAnchors: oppositeCount,
    componentCells: topology.cells,
  };
}

function boxMean(values, width, height, radius) {
  const horizontal = new Float32Array(values.length);
  const output = new Float32Array(values.length);
  for (let y = 0; y < height; y++) {
    const row = y * width;
    let sum = 0;
    for (let x = 0; x <= Math.min(width - 1, radius); x++) sum += values[row + x];
    for (let x = 0; x < width; x++) {
      horizontal[row + x] = sum;
      const removeX = x - radius;
      const addX = x + radius + 1;
      if (removeX >= 0) sum -= values[row + removeX];
      if (addX < width) sum += values[row + addX];
    }
  }
  for (let x = 0; x < width; x++) {
    let sum = 0;
    for (let y = 0; y <= Math.min(height - 1, radius); y++) sum += horizontal[y * width + x];
    for (let y = 0; y < height; y++) {
      const horizontalCount = Math.min(width - 1, x + radius) - Math.max(0, x - radius) + 1;
      const verticalCount = Math.min(height - 1, y + radius) - Math.max(0, y - radius) + 1;
      output[y * width + x] = sum / (horizontalCount * verticalCount);
      const removeY = y - radius;
      const addY = y + radius + 1;
      if (removeY >= 0) sum -= horizontal[removeY * width + x];
      if (addY < height) sum += horizontal[addY * width + x];
    }
  }
  return output;
}

export function guidedFilter(guide, input, width, height, radius, epsilon = 0.003) {
  const guideSquared = new Float32Array(guide.length);
  const guideInput = new Float32Array(guide.length);
  for (let index = 0; index < guide.length; index++) {
    guideSquared[index] = guide[index] * guide[index];
    guideInput[index] = guide[index] * input[index];
  }
  const meanGuide = boxMean(guide, width, height, radius);
  const meanInput = boxMean(input, width, height, radius);
  const correlationGuide = boxMean(guideSquared, width, height, radius);
  const correlationGuideInput = boxMean(guideInput, width, height, radius);
  const a = new Float32Array(guide.length);
  const b = new Float32Array(guide.length);
  for (let index = 0; index < guide.length; index++) {
    const variance = correlationGuide[index] - meanGuide[index] * meanGuide[index];
    const covariance = correlationGuideInput[index] - meanGuide[index] * meanInput[index];
    a[index] = covariance / (variance + epsilon);
    b[index] = meanInput[index] - a[index] * meanGuide[index];
  }
  const meanA = boxMean(a, width, height, radius);
  const meanB = boxMean(b, width, height, radius);
  const output = new Float32Array(guide.length);
  for (let index = 0; index < output.length; index++) output[index] = clamp01(meanA[index] * guide[index] + meanB[index]);
  return output;
}

/** Refines the rough geodesic proposal into a local, edge-aware alpha matte. */
export function guidedRefineHints(image, baseAlpha, size, keep, remove, hardKeep, hardRemove) {
  const plane = size * size;
  const guide = new Float32Array(plane);
  const proposal = new Float32Array(plane);
  const support = new Float32Array(plane);
  for (let index = 0; index < plane; index++) {
    guide[index] = image[index] * 0.2126 + image[plane + index] * 0.7152 + image[plane * 2 + index] * 0.0722;
    proposal[index] = Math.min(1 - remove[index], Math.max(keep[index], baseAlpha[index]));
    support[index] = Math.max(keep[index], remove[index]);
  }
  const radius = Math.max(2, Math.round(size / 80));
  const filteredAlpha = guidedFilter(guide, proposal, size, size, radius, 0.0025);
  const filteredSupport = guidedFilter(guide, support, size, size, radius, 0.008);
  const refinedKeep = new Float32Array(plane);
  const refinedRemove = new Float32Array(plane);
  for (let index = 0; index < plane; index++) {
    const localSupport = clamp01(Math.max(support[index], filteredSupport[index]));
    const alpha = baseAlpha[index] * (1 - localSupport) + filteredAlpha[index] * localSupport;
    const delta = alpha - baseAlpha[index];
    if (delta > 0) refinedKeep[index] = clamp01(delta / Math.max(1e-4, 1 - baseAlpha[index]));
    if (delta < 0) refinedRemove[index] = clamp01(-delta / Math.max(1e-4, baseAlpha[index]));
    if (hardKeep[index]) {
      refinedKeep[index] = 1;
      refinedRemove[index] = 0;
    } else if (hardRemove[index]) {
      refinedRemove[index] = 1;
      refinedKeep[index] = 0;
    }
  }
  return { keep: refinedKeep, remove: refinedRemove, radius };
}
