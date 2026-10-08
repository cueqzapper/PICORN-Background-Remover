# Small background-removal networks: local benchmark

Measured on 2026-09-09. No production model was replaced or deployed.

## Finding

Two relevant small networks were found and executed:

- **U²-NetP**: the small general-purpose U²-Net model. Its nested residual
  U-blocks learn salient foreground without a large pretrained backbone.
  The original authors distribute a roughly 4.7 MB checkpoint; the rembg ONNX
  tested here is **4,574,861 bytes**.
  [Original project](https://github.com/xuebinqin/U-2-Net),
  [paper](https://arxiv.org/abs/2005.09007),
  [rembg inference adapter](https://github.com/danielgatis/rembg/blob/47ac53f593ac1f7e6b17c3531ee4a455de092bcb/rembg/sessions/u2netp.py).
- **SINet**: an extremely small **portrait** segmentation network. Its spatial
  squeeze modules and information-blocking decoder use about 86.9k trainable
  parameters. The tested model is the publisher's Aisegment/Baidu-trained
  ONNX, **438,815 bytes**, rather than the original paper's EG1800 checkpoint.
  [Original research code](https://github.com/clovaai/ext_portrait_segmentation),
  [paper](https://arxiv.org/abs/1911.09099),
  [tested model publisher](https://github.com/anilsathyan7/Portrait-Segmentation/tree/dbf69b043cf70d3362bc500ee620f20807e622d2/SINet).

MediaPipe Selfie Segmentation is another small portrait candidate, but was not
benchmarked here. Its human segmentation scope is not a solution to logo holes.
[Google's model documentation](https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/selfie_segmentation.md).

**Conclusion:** U²-NetP offers better general-object masks on this sample, with
larger weights and higher latency. SINet is a plausible fast portrait option,
but is not a general background-removal replacement. Neither solves the logo
counter problem. Keeping Picorn's tiny network and improving its learned
foreground/background semantics remains justified by these results.

## Speed and download size

Windows, Intel Core i9-11900K (8 cores / 16 threads), ONNX Runtime 1.28.0 CPU
execution provider. Batch size 1. Sequential execution, 1 inter-op thread;
intra-op threads set explicitly to 1 or 4 for every model. Five warmups after
the first run, then 50 measured repetitions on the same image for each model.
Input resolution follows each model's inference contract; Picorn is also
tested at 320 for the equal-resolution comparison.

Sizes below use decimal MB (1 MB = 1,000,000 bytes). Timing covers inference
only, not download, image decoding, preprocessing, V10, graphic masking or PNG
export. These are native CPU measurements, **not browser WASM/WebGPU results**.

| Model | Input | ONNX MB | 1-thread median ms | 4-thread median ms | 4-thread p95 ms |
| --- | --- | ---: | ---: | ---: | ---: |
| Picorn V9 | 512 × 512 | 1.417 | 155.68 | 88.67 | 103.10 |
| Picorn V9 | 320 × 320 | 1.417 | 62.03 | 33.22 | 38.51 |
| U²-NetP | 320 × 320 | 4.575 | 445.86 | 190.57 | 232.11 |
| SINet, original export | 320 × 320 | 0.439 | 36.11 | 17.06 | 22.59 |
| SINet, fixed input declarations | 320 × 320 | 0.421 | 29.37 | 15.30 | 18.61 |

U²-NetP is 3.23 times Picorn's download size and 2.15 times its 512px inference
latency in the four-thread test. The fixed SINet is 70% smaller and 5.80 times
faster than Picorn at 512px; at the same 320px resolution it is 2.17 times faster.
These comparisons do not imply equal quality or equal task coverage.

First-run and session-load timings are retained in the raw results. Memory
and GPU/browser latency were not measured.

### SINet export repair

The supplied ONNX redundantly declares its fixed initializers as overridable
inputs. ONNX Runtime warns that this prevents some constant folding. The
benchmark removes only those input declarations in a separate file, preserving
the original download and every weight. The result is 421,320 bytes.
Original and repaired outputs were compared across all 99 images: maximum
absolute difference **0.00001520**, attributable to optimized floating-point
execution; mask metrics are practically unchanged. This is not retraining.

## Quality

Seed 20260909, 32 randomly selected validation images each from DIS5K, DUTS-TE
and P3M-10K: **96 real images**, plus **3 synthetic logos** from the preceding
graphic-refinement work. Selection is saved before inference. No thresholds or
weights were tuned on this sample. These are local validation subsets, not
full published benchmarks; external-model training overlap is not audited.

All models receive the same original images with their own documented
preprocessing. Picorn uses letterboxing and RGB/255; U²-NetP uses rembg's
320px Lanczos resize, per-image maximum normalization and ImageNet mean/std;
SINet uses its ONNX notebook's BGR, dataset mean/std and final /255 scaling.
SINet's foreground softmax channel is used as a soft mask. Real-image metrics
are computed on a shared aspect-preserving canvas with a maximum side of 512;
the synthetic logos are evaluated at their 1024 × 512 source resolution.

**Mean per-image foreground IoU**, threshold 0.5, higher is better:

| Model / pipeline | DIS5K, 32 | DUTS-TE, 32 | P3M portraits, 32 | Logos, 3 |
| --- | ---: | ---: | ---: | ---: |
| Picorn V9 512, raw | 0.3885 | 0.6285 | 0.9044 | 0.6598 |
| Picorn V9 320, raw | 0.2798 | 0.5468 | 0.8431 | 0.3878 |
| Picorn V9 512 + V10 | 0.3874 | 0.6293 | **0.9052** | 0.6698 |
| U²-NetP 320 | **0.4209** | **0.7477** | 0.8980 | **0.7728** |
| SINet 320, fixed export | 0.1604 | 0.3127 | 0.8622 | 0.3039 |

Alpha MAE and a one-pixel-tolerant boundary F1 are also retained in results.json.
The boundaries expose an important distinction: U²-NetP's DIS5K IoU is higher,
but its boundary F1 (0.3373) is below Picorn V10 (0.3715). It is not uniformly
better. On DUTS its boundary F1 is higher as well.

The comparison sheet uses the predetermined first two images in each group
and all three logos; images were not selected afterward to favour a model.
Both comparison sheets were visually inspected.

## Does another model fix the enclosed logo holes?

No. Each candidate was also tested with the **same white-graphic stage** from
the previous change, without altering that stage's thresholds:

| Neural source + shared graphic masking | Navy wordmark IoU | Pale wordmark IoU | White-artwork badge IoU |
| --- | ---: | ---: | ---: |
| Picorn 512 + V10 | **0.9671** | 0.8450 | **1.0000** |
| U²-NetP | 0.8453 | 0.8450 | 0.9640 |
| SINet, fixed | 0.8453 | 0.8450 | 0.9640 |

U²-NetP and SINet both retain all three enclosed wordmark holes. Picorn
removes two of three on the navy wordmark. All retain the intended white star.
General foreground detection and deciding whether an enclosed white region is
paper are distinct tasks; swapping in either downloaded network does not fix
the latter on these fixtures.

## Reproduction and artifacts

Run from `G:/SEEZ/PICORN-Background-Remover` with Python packages numpy, Pillow,
opencv-python, onnx, onnxruntime and Node installed. The scripts use the local
training manifests under `G:/SEEZ/IMAGE-SEGMENTATION/training/data`.

```powershell
python scripts/download-small-models.py
python scripts/evaluate-white-graphics.py
python scripts/benchmark-small-models.py
python scripts/compare-logo-candidates.py
```

The downloads are hash checked. No new runtime dependency is added to Picorn.
`benchmark-small-models.py` validates U²-NetP's upstream checksum; the downloader
verifies both exact SHA-256 hashes. All model files stay in the ignored local
verification folder rather than the production/public model directory.

Artifacts in `.verification/small-models/`:

- `results.json`: environment, sizes, hashes, latency, per-image metrics and averages.
- `selection.json`: exact images and ground-truth paths.
- `comparison.png`: real-photo and raw-model logo comparisons.
- `logo-pipelines.json`, `logo-pipelines.png`: same-refiner comparison.
- `source-revisions.json`: source repository revisions at investigation time.
- Original model files, repaired SINet model and local license snapshots.

The U²-Net project is Apache-2.0 licensed; the tested SINet publisher and
original SINet research repository contain MIT-style license texts. Source
notices were inspected and saved locally. No claim is made here about an
independent audit of their training-data rights or suitability for redistribution.

The next useful experiment is a compact semantic model trained with exact
graphic alpha and explicit white-foreground versus enclosed-background labels,
with portrait and general-object negatives retained. U²-NetP is worth considering
as an additional general-object training reference, but its own logo mistakes
should not become the training target.
