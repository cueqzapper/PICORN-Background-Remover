import { copyFile, mkdir } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const project = dirname(dirname(fileURLToPath(import.meta.url)));
const source = join(project, "node_modules", "onnxruntime-web", "dist");
const destination = join(project, "public", "vendor");
const files = [
  "ort.webgpu.bundle.min.mjs",
  "ort.jspi.bundle.min.mjs",
  "ort-wasm-simd-threaded.asyncify.wasm",
  "ort-wasm-simd-threaded.jspi.wasm",
  "ort.wasm.bundle.min.mjs",
  "ort-wasm-simd-threaded.wasm",
];

await mkdir(destination, { recursive: true });
await Promise.all(files.map((file) => copyFile(join(source, file), join(destination, file))));
console.log(`ONNX Runtime Browser-Assets: ${destination}`);
