import { readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { refineAlphaHybridCpu } from '../src/hybrid-refine.js';
import { refineWhiteGraphic } from '../src/white-graphic.js';
const [mode, directory, w, h] = process.argv.slice(2);
const floats = name => new Float32Array(new Uint8Array(readFileSync(join(directory, name))).buffer);
if (mode === 'hybrid') {
  const result = refineAlphaHybridCpu(floats('image.f32'), floats('model.f32'), 512, 512);
  writeFileSync(join(directory, 'hybrid.f32'), new Uint8Array(result.buffer));
} else {
  const data = new Uint8ClampedArray(readFileSync(join(directory, 'source.rgba')));
  const before = performance.now();
  const result = refineWhiteGraphic(data, Number(w), Number(h), floats('baseline.f32'));
  const milliseconds = performance.now() - before;
  writeFileSync(join(directory, 'result.json'), JSON.stringify({ accepted: !!result, milliseconds, ...result?.diagnostics }));
  if (result) {
    writeFileSync(join(directory, 'improved.f32'), new Uint8Array(result.alpha.buffer));
    writeFileSync(join(directory, 'foreground.rgba'), result.foreground);
  }
}
