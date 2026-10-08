# White-background graphic refinement

## Current model and failure mechanism

The shipped V9 ONNX is the V8 MicroAlpha architecture with fine-tuned weights,
not `training/model_v9.py`'s experimental Coordinate Attention architecture.
It has 312,878 fused inference parameters and occupies 1,417,388 bytes.
The training source is in `G:/SEEZ/IMAGE-SEGMENTATION/training`.
See `model_v7.py`, `model_v8.py`, `dataset_v7.py`, `losses_v8.py`, and
this repository's `docs/model-v9.md` for the architecture and release evidence.

The encoder uses compact inverted residual blocks, multiscale context and
skip-connected decoding. RGB gradients, entropy/colour fingerprints and learned
regional affinities help the detail decoder. V8 adds outward gradient attention
and auxiliary training supervision. Guidance and explicit keep/remove inputs
support interactive corrections. Runtime defaults to WebGPU with a WASM fallback.

Studio runs the model on a 512-pixel letterboxed input. V10 follows it with
eight edge-aware Jacobi iterations and a guided filter. Values <=0.06 and >=0.94
are anchors: confident mistakes survive. The filter runs at inference resolution;
export previously enlarged that matte, so missing tiny letters stayed missing.

The recorded full V9 validation IoU is 0.592943 across 6,489 images, with P3M
portraits considerably stronger than DIS5K objects. Those are historical release
measurements, not a new full validation run. Weights were not changed here.

## Implemented change

`src/white-graphic.js` is shared byte-for-byte with the Studio integration at
`G:/Upload_Post_Engine/src/web/lib/ai/picorn/white-graphic.js`.

1. Require near-white borders on all four sides, substantial white area,
   predominantly flat ink and a small dominant palette. Reject existing alpha.
2. Work from source pixels up to 4 million pixels / 4096 pixels on the long side.
   The neural input and model size do not grow.
3. Flood-fill white components with four-neighbour connectivity. Remove those
   touching the image border, even if the model labelled them foreground.
4. Remove an enclosed white component only when at least 80% of its pixels have
   neural alpha below 0.2. Otherwise preserve it as potentially intentional art.
5. Recover nonwhite graphic pixels, including disconnected fine lettering.
6. At the immediate paper edge, solve the white compositing equation against
   nearby ink. Accept only a low-residual colour fit; keep flat interiors opaque.
   Export corrected foreground RGB with the new alpha to reduce white fringes.

The demo previews and exports the refined result. Manual brush correction starts
from the corrected mask and bypasses automatic graphics masking so explicit
edits retain priority. Studio and its image-tools demo use the same service.

## Validation, 2026-09-09

Run `npm test` and `npm run build` here. In Upload_Post_Engine run
`npx vitest run tests/unit/web/picorn` and `npm run build:web`.

Both suites passed 25 tests and both frontend builds passed. The regression
cases cover connected paper, disconnected ink, enclosed holes and white artwork,
pale ink, edge colour reconstruction, textured negatives, transparency and
invalid inputs. The ONNX hash remains checked by the existing tests.

Run `python scripts/evaluate-white-graphics.py` with numpy, Pillow and
onnxruntime installed to execute the actual ONNX + V10 CPU baseline and compare
the new stage against known synthetic alpha truth. It writes images and metrics
under `.verification/white-graphics/`, including `comparison.png`.

| Fixture | Baseline IoU | New IoU | Baseline alpha MAE | New alpha MAE |
| --- | ---: | ---: | ---: | ---: |
| Navy wordmark | 0.8737 | 0.9671 | 0.02239 | 0.00295 |
| Pale wordmark | 0.3787 | 0.8450 | 0.13659 | 0.01585 |
| White artwork in coloured badge | 0.7569 | 1.0000 | 0.07777 | 0.00003 |

Added stage: approximately 56–89 ms for these 1024x512 fixtures on this machine.
These are three synthetic diagnostics, not a diverse held-out quality benchmark.
The comparison sheet was visually inspected. Browser discovery returned no
available browser, so interactive upload, brush and download QA remain unverified.
No publication or deployment was performed.

## Known limits and next model improvement

Enclosed white is inherently ambiguous. The navy wordmark retains one letter
counter; the pale wordmark retains three. A global white key would fix these but
also erase the white star in the badge. This implementation preserves ambiguity.
Near-white ink (roughly 247–255), shadows, gradients, textured paper, tight crops,
smooth product renders and JPEG artefacts need a broader false-positive audit.
This heuristic does not guarantee that every accepted image is a logo.

The next training change should address component semantics, without increasing
the deployed network: fine-tune the existing architecture using licensed graphic
composites with exact alpha labels, letter counters, disconnected text, intentional
white artwork, pale ink, varied paper whites and photo negatives. Add supervision
for enclosed background regions and balance it against white-foreground retention.
Split by source design/font before generating variants to avoid train/test leakage.
Validate on separately held-out real logos plus the existing portrait/object splits.
Teacher segmentation alone is not reliable ground truth for white holes.
Promote new weights only after hole recall improves without white-artwork or
photo regressions, and check ONNX parity, model bytes and browser latency.
