# PICORN V12 model notes

Released 8 October 2026 as demo version 0.4.0.

## What changed

V7 to V11 learned their encoder from scratch on about 67k images. An error
analysis on 240 validation images showed that about 58% of V11's matte error
lay outside the edge band: wrong objects or extra background, not soft edges.
Edge-only post-processing could not fix that (V10 added +0.001 IoU).

V12 therefore changes the semantics, not the edge stage:

- **Encoder:** ImageNet-pretrained `mobilenetv4_conv_small_050` (timm) with 1×1
  projections to the previous channel widths. The V11 decoder, entropy
  fingerprint, dynamic prototypes and guidance inputs are unchanged, so the
  browser contract (`image`, `guidance`, `guidance_weight`, `keep_hint`,
  `remove_hint` → `alpha`) is identical. ReLU activations only.
- **Data:** the existing corpus plus on-the-fly sticker composites (1–3 RGBA
  stickers on photos or procedural grounds, with shadows, colour
  harmonisation, defocus, noise and JPEG; split by motif) and 24,353 COCO
  train2017 images whose two teacher mattes (Lucida, BRIA RMBG-2.0) agree at
  IoU ≥ 0.85.
- **Training:** about 1.7M samples instead of about 0.6M (60k steps mixed
  320/384/512 px, then 8k steps at 512 px), followed by a ×1.5 alpha-logit
  calibration chosen on validation only.
- **Export:** float initializers are stored as FP16 with a Cast node. ONNX
  Runtime folds the cast at session creation; all arithmetic stays FP32. On a
  240-image full-resolution check, FP16-stored and FP32 weights gave the same
  IoU, and the file is 45% smaller.

## Results

Original 6,489-image holdout (DIS5K, DUTS-TE, P3M-10K) at 512 px, network only:

| Metric | V11 | V12 |
| --- | ---: | ---: |
| IoU | 0.617 | **0.773** |
| Alpha MAE | 0.073 | **0.036** |
| Boundary F1 | 0.395 | **0.596** |

| IoU by source | V11 | V12 |
| --- | ---: | ---: |
| DUTS-TE (5,019) | 0.575 | 0.753 |
| DIS5K (470) | 0.430 | 0.571 |
| P3M-10K portraits (1,000) | 0.917 | 0.971 |
| Exact procedural graphics (400) | 0.940 | 0.962 |
| Held-out test, COCO (244) / graphics (400) | 0.617 / 0.930 | 0.728 / 0.954 |

CPU latency (ONNX Runtime 1.28, i9-11900K, 4 threads, 40 interleaved runs):
76.7 ms at 512 px and 27.4 ms at 320 px for V12, against 77.0 and 28.0 ms for
V11. A headless Chromium run on WebGPU matched native ONNX Runtime output.

## Runtime additions

- `inferAutomatic`: a second inference on the first pass's subject box, at
  most 80% of the image area. +0.9 IoU points overall and +2.2 for subjects
  under 15% of the frame (measured with V11).
- `snapConfidentAlpha`: far from the 0.5 contour (at least 1% of the long
  side), alpha above 0.75 becomes 1 and alpha below 0.25 becomes 0. Alpha MAE
  fell 1.5%, and no source got worse.
- `foreground.js`: two-pass blur-fusion foreground estimation (Forte &
  Pitié, ICIP 2021) for soft edge pixels on PNG export.
- `white-graphic.js`: border-connected ground removal now works for any flat
  ground colour, not only white paper.

## Limits

Transparent glass, scenes without a clear subject and very small secondary
subjects still fail. About 55% of the remaining error now lies in the edge
band, so edges are the next target. The weights remain under the separate
[model terms](../MODEL_LICENSE.md).

SHA-256: `119203edd80c2907e5c8c81967b1c594fb44b1a9a9ed969d7fb57fa97dc6ce91`
