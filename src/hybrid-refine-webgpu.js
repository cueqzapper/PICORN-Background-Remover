import {
  HYBRID_REFINER_VERSION,
  refineAlphaHybridCpu,
} from "./hybrid-refine.js";

const JACOBI_SHADER = /* wgsl */ `
struct Dimensions { width: u32, height: u32, lowWidth: u32, lowHeight: u32 };
@group(0) @binding(0) var<storage, read> rgb: array<f32>;
@group(0) @binding(1) var<storage, read> originalAlpha: array<f32>;
@group(0) @binding(2) var<storage, read> currentAlpha: array<f32>;
@group(0) @binding(3) var<storage, read_write> nextAlpha: array<f32>;
@group(0) @binding(4) var<uniform> dimensions: Dimensions;

fn adjacent(x: i32, y: i32) -> u32 {
  let safeX = clamp(x, 0, i32(dimensions.width) - 1);
  let safeY = clamp(y, 0, i32(dimensions.height) - 1);
  return u32(safeY) * dimensions.width + u32(safeX);
}

fn edgeWeight(index: u32, neighbour: u32, plane: u32) -> f32 {
  let dr = rgb[index] - rgb[neighbour];
  let dg = rgb[plane + index] - rgb[plane + neighbour];
  let db = rgb[plane * 2u + index] - rgb[plane * 2u + neighbour];
  let distance = (dr * dr + dg * dg + db * db) / 3.0;
  return exp(-distance / (2.0 * 0.08 * 0.08));
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x >= dimensions.width || id.y >= dimensions.height) { return; }
  let index = id.y * dimensions.width + id.x;
  let original = originalAlpha[index];
  if (original <= 0.06 || original >= 0.94) {
    nextAlpha[index] = original;
    return;
  }
  let plane = dimensions.width * dimensions.height;
  let left = adjacent(i32(id.x) - 1, i32(id.y));
  let right = adjacent(i32(id.x) + 1, i32(id.y));
  let top = adjacent(i32(id.x), i32(id.y) - 1);
  let bottom = adjacent(i32(id.x), i32(id.y) + 1);
  let wl = edgeWeight(index, left, plane);
  let wr = edgeWeight(index, right, plane);
  let wt = edgeWeight(index, top, plane);
  let wb = edgeWeight(index, bottom, plane);
  let weightSum = wl + wr + wt + wb;
  let weightedAlpha = wl * currentAlpha[left] + wr * currentAlpha[right]
    + wt * currentAlpha[top] + wb * currentAlpha[bottom];
  let transition = max(0.0, 4.0 * original * (1.0 - original));
  let dataConfidence = 0.12 + 3.0 * (1.0 - transition) * (1.0 - transition);
  let candidate = (dataConfidence * original + 1.25 * weightedAlpha)
    / max(0.000001, dataConfidence + 1.25 * weightSum);
  nextAlpha[index] = clamp(
    currentAlpha[index] + transition * 0.65 * (candidate - currentAlpha[index]),
    0.0,
    1.0,
  );
}
`;

const DOWNSAMPLE_SHADER = /* wgsl */ `
struct Dimensions { width: u32, height: u32, lowWidth: u32, lowHeight: u32 };
@group(0) @binding(0) var<storage, read> rgb: array<f32>;
@group(0) @binding(1) var<storage, read> alpha: array<f32>;
@group(0) @binding(2) var<storage, read_write> lowLuma: array<f32>;
@group(0) @binding(3) var<storage, read_write> lowAlpha: array<f32>;
@group(0) @binding(4) var<uniform> dimensions: Dimensions;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x >= dimensions.lowWidth || id.y >= dimensions.lowHeight) { return; }
  let plane = dimensions.width * dimensions.height;
  var luma = 0.0;
  var matte = 0.0;
  for (var dy = 0u; dy < 4u; dy = dy + 1u) {
    for (var dx = 0u; dx < 4u; dx = dx + 1u) {
      let x = id.x * 4u + dx;
      let y = id.y * 4u + dy;
      let index = y * dimensions.width + x;
      luma = luma + rgb[index] * 0.2126
        + rgb[plane + index] * 0.7152
        + rgb[plane * 2u + index] * 0.0722;
      matte = matte + alpha[index];
    }
  }
  let lowIndex = id.y * dimensions.lowWidth + id.x;
  lowLuma[lowIndex] = luma / 16.0;
  lowAlpha[lowIndex] = matte / 16.0;
}
`;

const COEFFICIENT_SHADER = /* wgsl */ `
struct Dimensions { width: u32, height: u32, lowWidth: u32, lowHeight: u32 };
@group(0) @binding(0) var<storage, read> lowLuma: array<f32>;
@group(0) @binding(1) var<storage, read> lowAlpha: array<f32>;
@group(0) @binding(2) var<storage, read_write> coefficient: array<f32>;
@group(0) @binding(3) var<storage, read_write> intercept: array<f32>;
@group(0) @binding(4) var<storage, read_write> foregroundSupport: array<f32>;
@group(0) @binding(5) var<storage, read_write> backgroundSupport: array<f32>;
@group(0) @binding(6) var<uniform> dimensions: Dimensions;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x >= dimensions.lowWidth || id.y >= dimensions.lowHeight) { return; }
  var guide = 0.0;
  var matte = 0.0;
  var guideSquared = 0.0;
  var guideMatte = 0.0;
  var maximum = 0.0;
  var minimum = 1.0;
  var count = 0.0;
  for (var dy = -2; dy <= 2; dy = dy + 1) {
    let y = i32(id.y) + dy;
    if (y < 0 || y >= i32(dimensions.lowHeight)) { continue; }
    for (var dx = -2; dx <= 2; dx = dx + 1) {
      let x = i32(id.x) + dx;
      if (x < 0 || x >= i32(dimensions.lowWidth)) { continue; }
      let index = u32(y) * dimensions.lowWidth + u32(x);
      let imageValue = lowLuma[index];
      let alphaValue = lowAlpha[index];
      guide = guide + imageValue;
      matte = matte + alphaValue;
      guideSquared = guideSquared + imageValue * imageValue;
      guideMatte = guideMatte + imageValue * alphaValue;
      maximum = max(maximum, alphaValue);
      minimum = min(minimum, alphaValue);
      count = count + 1.0;
    }
  }
  let meanGuide = guide / count;
  let meanAlpha = matte / count;
  let variance = guideSquared / count - meanGuide * meanGuide;
  let covariance = guideMatte / count - meanGuide * meanAlpha;
  let index = id.y * dimensions.lowWidth + id.x;
  coefficient[index] = covariance / (variance + 0.0025);
  intercept[index] = meanAlpha - coefficient[index] * meanGuide;
  foregroundSupport[index] = select(0.0, 1.0, maximum >= 0.94);
  backgroundSupport[index] = select(0.0, 1.0, minimum <= 0.06);
}
`;

const SMOOTH_COEFFICIENT_SHADER = /* wgsl */ `
struct Dimensions { width: u32, height: u32, lowWidth: u32, lowHeight: u32 };
@group(0) @binding(0) var<storage, read> coefficient: array<f32>;
@group(0) @binding(1) var<storage, read> intercept: array<f32>;
@group(0) @binding(2) var<storage, read_write> meanCoefficient: array<f32>;
@group(0) @binding(3) var<storage, read_write> meanIntercept: array<f32>;
@group(0) @binding(4) var<uniform> dimensions: Dimensions;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x >= dimensions.lowWidth || id.y >= dimensions.lowHeight) { return; }
  var sumCoefficient = 0.0;
  var sumIntercept = 0.0;
  var count = 0.0;
  for (var dy = -2; dy <= 2; dy = dy + 1) {
    let y = i32(id.y) + dy;
    if (y < 0 || y >= i32(dimensions.lowHeight)) { continue; }
    for (var dx = -2; dx <= 2; dx = dx + 1) {
      let x = i32(id.x) + dx;
      if (x < 0 || x >= i32(dimensions.lowWidth)) { continue; }
      let index = u32(y) * dimensions.lowWidth + u32(x);
      sumCoefficient = sumCoefficient + coefficient[index];
      sumIntercept = sumIntercept + intercept[index];
      count = count + 1.0;
    }
  }
  let index = id.y * dimensions.lowWidth + id.x;
  meanCoefficient[index] = sumCoefficient / count;
  meanIntercept[index] = sumIntercept / count;
}
`;

const SLICE_SHADER = /* wgsl */ `
struct Dimensions { width: u32, height: u32, lowWidth: u32, lowHeight: u32 };
@group(0) @binding(0) var<storage, read> rgb: array<f32>;
@group(0) @binding(1) var<storage, read> referenceAlpha: array<f32>;
@group(0) @binding(2) var<storage, read> meanCoefficient: array<f32>;
@group(0) @binding(3) var<storage, read> meanIntercept: array<f32>;
@group(0) @binding(4) var<storage, read> foregroundSupport: array<f32>;
@group(0) @binding(5) var<storage, read> backgroundSupport: array<f32>;
@group(0) @binding(6) var<storage, read_write> outputAlpha: array<f32>;
@group(0) @binding(7) var<uniform> dimensions: Dimensions;

fn sampleCoefficient(x: f32, y: f32) -> f32 {
  let rawX = i32(floor(x));
  let rawY = i32(floor(y));
  let x0 = clamp(rawX, 0, i32(dimensions.lowWidth) - 1);
  let y0 = clamp(rawY, 0, i32(dimensions.lowHeight) - 1);
  let x1 = clamp(rawX + 1, 0, i32(dimensions.lowWidth) - 1);
  let y1 = clamp(rawY + 1, 0, i32(dimensions.lowHeight) - 1);
  let tx = clamp(x - f32(rawX), 0.0, 1.0);
  let ty = clamp(y - f32(rawY), 0.0, 1.0);
  let top = mix(
    meanCoefficient[u32(y0) * dimensions.lowWidth + u32(x0)],
    meanCoefficient[u32(y0) * dimensions.lowWidth + u32(x1)],
    tx,
  );
  let bottom = mix(
    meanCoefficient[u32(y1) * dimensions.lowWidth + u32(x0)],
    meanCoefficient[u32(y1) * dimensions.lowWidth + u32(x1)],
    tx,
  );
  return mix(top, bottom, ty);
}

fn sampleIntercept(x: f32, y: f32) -> f32 {
  let rawX = i32(floor(x));
  let rawY = i32(floor(y));
  let x0 = clamp(rawX, 0, i32(dimensions.lowWidth) - 1);
  let y0 = clamp(rawY, 0, i32(dimensions.lowHeight) - 1);
  let x1 = clamp(rawX + 1, 0, i32(dimensions.lowWidth) - 1);
  let y1 = clamp(rawY + 1, 0, i32(dimensions.lowHeight) - 1);
  let tx = clamp(x - f32(rawX), 0.0, 1.0);
  let ty = clamp(y - f32(rawY), 0.0, 1.0);
  let top = mix(
    meanIntercept[u32(y0) * dimensions.lowWidth + u32(x0)],
    meanIntercept[u32(y0) * dimensions.lowWidth + u32(x1)],
    tx,
  );
  let bottom = mix(
    meanIntercept[u32(y1) * dimensions.lowWidth + u32(x0)],
    meanIntercept[u32(y1) * dimensions.lowWidth + u32(x1)],
    tx,
  );
  return mix(top, bottom, ty);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x >= dimensions.width || id.y >= dimensions.height) { return; }
  let index = id.y * dimensions.width + id.x;
  let reference = referenceAlpha[index];
  if (reference <= 0.06 || reference >= 0.94) {
    outputAlpha[index] = reference;
    return;
  }
  let plane = dimensions.width * dimensions.height;
  let luma = rgb[index] * 0.2126
    + rgb[plane + index] * 0.7152
    + rgb[plane * 2u + index] * 0.0722;
  let lowX = (f32(id.x) + 0.5) * f32(dimensions.lowWidth) / f32(dimensions.width) - 0.5;
  let lowY = (f32(id.y) + 0.5) * f32(dimensions.lowHeight) / f32(dimensions.height) - 0.5;
  var candidate = sampleCoefficient(lowX, lowY) * luma + sampleIntercept(lowX, lowY);
  let nearestX = min(dimensions.lowWidth - 1u, id.x * dimensions.lowWidth / dimensions.width);
  let nearestY = min(dimensions.lowHeight - 1u, id.y * dimensions.lowHeight / dimensions.height);
  let lowIndex = nearestY * dimensions.lowWidth + nearestX;
  if (reference >= 0.5 && foregroundSupport[lowIndex] < 0.5) {
    candidate = max(candidate, reference);
  } else if (reference < 0.5 && backgroundSupport[lowIndex] < 0.5) {
    candidate = min(candidate, reference);
  }
  candidate = clamp(candidate, 0.0, 1.0);
  let transition = max(0.0, 4.0 * reference * (1.0 - reference));
  var refined = reference + transition * 0.75 * (candidate - reference);
  if (reference >= 0.5) {
    refined = max(0.5, refined);
  } else {
    refined = min(0.499999, refined);
  }
  outputAlpha[index] = clamp(refined, 0.0, 1.0);
}
`;

async function checkedPipeline(device, label, code) {
  const module = device.createShaderModule({ label, code });
  const information = await module.getCompilationInfo?.();
  const errors = information?.messages?.filter((message) => message.type === "error") ?? [];
  if (errors.length) {
    throw new Error(`${label}: ${errors.map((error) => error.message).join("; ")}`);
  }
  return device.createComputePipelineAsync({
    label,
    layout: "auto",
    compute: { module, entryPoint: "main" },
  });
}

function storageBuffer(device, label, byteLength, data = null) {
  const buffer = device.createBuffer({
    label,
    size: Math.max(4, Math.ceil(byteLength / 4) * 4),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  });
  if (data) {
    device.queue.writeBuffer(buffer, 0, data.buffer, data.byteOffset, data.byteLength);
  }
  return buffer;
}

function binding(binding, buffer) {
  return { binding, resource: { buffer } };
}

export class HybridAlphaRefiner {
  constructor({ preferWebGpu = true, onFallback = null } = {}) {
    this.preferWebGpu = preferWebGpu;
    this.onFallback = onFallback;
    this.device = null;
    this.pipelines = null;
    this.initializing = null;
    this.webGpuDisabled = false;
    this.backend = "CPU";
  }

  async initialize() {
    if (this.device && this.pipelines) return true;
    if (this.initializing) return this.initializing;
    if (
      !this.preferWebGpu
      || this.webGpuDisabled
      || !globalThis.navigator?.gpu
    ) return false;
    this.initializing = this.#initializeWebGpu();
    try {
      return await this.initializing;
    } catch (error) {
      this.webGpuDisabled = true;
      this.backend = "CPU";
      this.onFallback?.(error);
      return false;
    } finally {
      this.initializing = null;
    }
  }

  async #initializeWebGpu() {
    const adapter = await globalThis.navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) throw new Error("WebGPU adapter unavailable for PICORN refinement.");
    const device = await adapter.requestDevice({ label: `PICORN ${HYBRID_REFINER_VERSION}` });
    const [jacobi, downsample, coefficient, smoothCoefficient, slice] = await Promise.all([
      checkedPipeline(device, "PICORN edge-aware Jacobi", JACOBI_SHADER),
      checkedPipeline(device, "PICORN guided downsample", DOWNSAMPLE_SHADER),
      checkedPipeline(device, "PICORN guided coefficients", COEFFICIENT_SHADER),
      checkedPipeline(device, "PICORN guided coefficient smoothing", SMOOTH_COEFFICIENT_SHADER),
      checkedPipeline(device, "PICORN guided slicing", SLICE_SHADER),
    ]);
    this.device = device;
    this.pipelines = { jacobi, downsample, coefficient, smoothCoefficient, slice };
    this.backend = "WebGPU";
    device.lost.then((information) => {
      this.device = null;
      this.pipelines = null;
      this.webGpuDisabled = true;
      this.backend = "CPU";
      this.onFallback?.(new Error(`PICORN WebGPU device lost: ${information.message}`));
    });
    return true;
  }

  async refine(image, alpha, width, height = width) {
    const started = globalThis.performance?.now?.() ?? Date.now();
    const canUseWebGpu = width % 4 === 0 && height % 4 === 0 && await this.initialize();
    if (canUseWebGpu) {
      try {
        const refined = await this.#refineWebGpu(image, alpha, width, height);
        return {
          alpha: refined,
          backend: "WebGPU",
          milliseconds: (globalThis.performance?.now?.() ?? Date.now()) - started,
          version: HYBRID_REFINER_VERSION,
        };
      } catch (error) {
        this.webGpuDisabled = true;
        this.backend = "CPU";
        this.onFallback?.(error);
      }
    }
    return {
      alpha: refineAlphaHybridCpu(image, alpha, width, height),
      backend: "CPU",
      milliseconds: (globalThis.performance?.now?.() ?? Date.now()) - started,
      version: HYBRID_REFINER_VERSION,
    };
  }

  async #refineWebGpu(image, alpha, width, height) {
    const device = this.device;
    const pipelines = this.pipelines;
    if (!device || !pipelines) throw new Error("PICORN WebGPU refiner is not initialized.");
    const pixels = width * height;
    if (image.length !== pixels * 3 || alpha.length !== pixels) {
      throw new Error("PICORN WebGPU input dimensions do not match.");
    }
    const lowWidth = width / 4;
    const lowHeight = height / 4;
    const lowBytes = lowWidth * lowHeight * 4;
    const buffers = [];
    const make = (label, byteLength, data = null) => {
      const buffer = storageBuffer(device, label, byteLength, data);
      buffers.push(buffer);
      return buffer;
    };
    const rgb = make("PICORN RGB", image.byteLength, image);
    const original = make("PICORN original alpha", alpha.byteLength, alpha);
    const alphaA = make("PICORN alpha A", alpha.byteLength, alpha);
    const alphaB = make("PICORN alpha B", alpha.byteLength);
    const lowLuma = make("PICORN low luma", lowBytes);
    const lowAlpha = make("PICORN low alpha", lowBytes);
    const coefficient = make("PICORN coefficient", lowBytes);
    const intercept = make("PICORN intercept", lowBytes);
    const foregroundSupport = make("PICORN foreground support", lowBytes);
    const backgroundSupport = make("PICORN background support", lowBytes);
    const meanCoefficient = make("PICORN mean coefficient", lowBytes);
    const meanIntercept = make("PICORN mean intercept", lowBytes);
    const uniform = device.createBuffer({
      label: "PICORN hybrid dimensions",
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    buffers.push(uniform);
    device.queue.writeBuffer(
      uniform,
      0,
      new Uint32Array([width, height, lowWidth, lowHeight]),
    );
    const readback = device.createBuffer({
      label: "PICORN refined alpha readback",
      size: alpha.byteLength,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    buffers.push(readback);

    try {
      const encoder = device.createCommandEncoder({ label: "PICORN hybrid refinement" });
      let current = alphaA;
      let next = alphaB;
      for (let iteration = 0; iteration < 8; iteration += 1) {
        const pass = encoder.beginComputePass({ label: `PICORN Jacobi ${iteration + 1}` });
        pass.setPipeline(pipelines.jacobi);
        pass.setBindGroup(0, device.createBindGroup({
          layout: pipelines.jacobi.getBindGroupLayout(0),
          entries: [
            binding(0, rgb),
            binding(1, original),
            binding(2, current),
            binding(3, next),
            binding(4, uniform),
          ],
        }));
        pass.dispatchWorkgroups(Math.ceil(width / 16), Math.ceil(height / 16));
        pass.end();
        [current, next] = [next, current];
      }

      let pass = encoder.beginComputePass({ label: "PICORN guided downsample" });
      pass.setPipeline(pipelines.downsample);
      pass.setBindGroup(0, device.createBindGroup({
        layout: pipelines.downsample.getBindGroupLayout(0),
        entries: [
          binding(0, rgb),
          binding(1, current),
          binding(2, lowLuma),
          binding(3, lowAlpha),
          binding(4, uniform),
        ],
      }));
      pass.dispatchWorkgroups(Math.ceil(lowWidth / 16), Math.ceil(lowHeight / 16));
      pass.end();

      pass = encoder.beginComputePass({ label: "PICORN guided coefficients" });
      pass.setPipeline(pipelines.coefficient);
      pass.setBindGroup(0, device.createBindGroup({
        layout: pipelines.coefficient.getBindGroupLayout(0),
        entries: [
          binding(0, lowLuma),
          binding(1, lowAlpha),
          binding(2, coefficient),
          binding(3, intercept),
          binding(4, foregroundSupport),
          binding(5, backgroundSupport),
          binding(6, uniform),
        ],
      }));
      pass.dispatchWorkgroups(Math.ceil(lowWidth / 16), Math.ceil(lowHeight / 16));
      pass.end();

      pass = encoder.beginComputePass({ label: "PICORN guided coefficient smoothing" });
      pass.setPipeline(pipelines.smoothCoefficient);
      pass.setBindGroup(0, device.createBindGroup({
        layout: pipelines.smoothCoefficient.getBindGroupLayout(0),
        entries: [
          binding(0, coefficient),
          binding(1, intercept),
          binding(2, meanCoefficient),
          binding(3, meanIntercept),
          binding(4, uniform),
        ],
      }));
      pass.dispatchWorkgroups(Math.ceil(lowWidth / 16), Math.ceil(lowHeight / 16));
      pass.end();

      pass = encoder.beginComputePass({ label: "PICORN guided slicing" });
      pass.setPipeline(pipelines.slice);
      pass.setBindGroup(0, device.createBindGroup({
        layout: pipelines.slice.getBindGroupLayout(0),
        entries: [
          binding(0, rgb),
          binding(1, current),
          binding(2, meanCoefficient),
          binding(3, meanIntercept),
          binding(4, foregroundSupport),
          binding(5, backgroundSupport),
          binding(6, next),
          binding(7, uniform),
        ],
      }));
      pass.dispatchWorkgroups(Math.ceil(width / 16), Math.ceil(height / 16));
      pass.end();
      encoder.copyBufferToBuffer(next, 0, readback, 0, alpha.byteLength);
      device.queue.submit([encoder.finish()]);
      await readback.mapAsync(GPUMapMode.READ);
      const result = new Float32Array(readback.getMappedRange().slice(0));
      readback.unmap();
      return result;
    } finally {
      for (const buffer of buffers) buffer.destroy();
    }
  }
}
