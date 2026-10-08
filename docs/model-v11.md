# V11 local model evaluation

Evaluated on 9 September 2026. This is a local integration, not a deployment
or a claim that the public demo has changed.

V11 improves object masks, portraits and graphics while retaining 312,878
inference parameters. The guided ONNX export is 1,417,525 bytes, 137 bytes
larger than V9. The existing hybrid refinement and correction brushes remain.

## Paired validation

Both versions were evaluated on the same images at 512 px. The following
numbers measure the network before browser postprocessing. Higher IoU is
better; it measures foreground overlap, not the percentage of perfect cutouts.

| Validation group | Images | V9 IoU | V11 IoU |
| --- | ---: | ---: | ---: |
| Original combined holdout | 6,489 | 59.29% | 61.73% |
| General objects (DUTS-TE) | 5,019 | 54.98% | 57.52% |
| Difficult objects (DIS5K) | 470 | 40.33% | 43.00% |
| Portraits (P3M-10K) | 1,000 | 89.89% | 91.66% |
| Exact procedural graphics | 400 | 60.64% | 94.04% |

On the original holdout, mean alpha error falls from 0.07449 to 0.06794
(8.8% lower), and boundary F1 rises from 0.38433 to 0.41737. The paired
bootstrap 95% interval for the IoU gain is +2.24 to +2.64 percentage points.
The full validation contains 7,123 images; the original holdout is reported
separately so added graphics cannot inflate the comparison on existing data.

A separate 644-image test contains 400 procedural graphics and 244 natural
images with object masks. Graphics IoU improves from 53.24% to 93.01%; natural
image IoU improves from 57.18% to 61.65%. These groups are not interchangeable
with the original holdout or a customer photo collection.

## Product pipeline and performance

A fixed 99-image panel compares the network together with the existing hybrid
refinement and the white-graphic stage. Its groups contain 32 images each from
DIS5K, DUTS-TE and P3M-10K, plus three synthetic logo examples.

| Product output | V9 with current refiners | V11 with current refiners |
| --- | ---: | ---: |
| Difficult objects IoU | 38.74% | 41.63% |
| General objects IoU | 62.93% | 64.66% |
| Portraits IoU | 90.52% | 91.86% |
| Three logo examples IoU | 93.74% | 98.89% |

CPU inference was also timed in 60 randomized, interleaved rounds per model,
using four ONNX Runtime threads on an Intel i9-11900K. Median 512 px inference
was 77.20 ms for V9 and 77.40 ms for V11; at 320 px it was 28.25 and 27.98 ms.
These are warm network timings, excluding download, image decoding, refinement
and export. They are not browser WebGPU or mobile measurements.

## Export checks and limitations

ONNX and PyTorch agree within 0.000002 over 320-square, 256x384 and 512-square
inputs, with and without guidance. Explicit keep/remove hints still produce
exact alpha 1/0. Both local applications use a versioned V11 file and cache.

Visual review covered fixed samples, actual product cutouts, and the largest
per-image improvements and regressions. Small subjects, glass, lattice-like
structures and complex backgrounds can still fail badly. Some individual
photos regress even though every evaluated group improves on average. Pale
graphics on nonwhite backgrounds can lose opacity. V11 is an incremental
improvement, not an exceptionally reliable general-purpose model.

SHA-256:
`f59ac457a9612bccae8587d2956acaa75b58020130c931d561a5cdf434392987`.

The inference/model license remains unchanged. Training data, checkpoints and
the training pipeline are not included in this repository.
