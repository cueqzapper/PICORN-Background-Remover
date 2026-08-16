# PICORN V9 model notes

V9 is a measured update to the existing compact model. The browser runtime,
ONNX graph shape, parameter count and file size remain unchanged.

## What changed

The selected model received one additional low-learning-rate fine-tuning epoch
on the existing 49,789-image training set. It still uses the V8 architecture:
312,878 fused inference parameters in a 1,417,388-byte FP32 ONNX file.

Several architectural and teacher-guidance ideas were tested first, including
semantic residual heads, calibrated teacher targets, safe-region teacher
guidance and Coordinate Attention. None beat V8 consistently enough to justify
extra runtime logic or parameters, so they are not part of V9.

## Full validation results

All measurements use the same 512 px evaluation path. Higher IoU and boundary
F1 are better; lower MAE is better.

| Dataset | Images | Metric | V8 | V9 |
| --- | ---: | --- | ---: | ---: |
| Combined | 6,489 | IoU | 0.592329 | **0.592943** |
| Combined | 6,489 | MAE | 0.075149 | **0.074491** |
| Combined | 6,489 | Boundary F1 | 0.382060 | **0.384325** |
| DIS5K | 470 | IoU | **0.404517** | 0.403265 |
| DIS5K | 470 | MAE | 0.116449 | **0.116031** |
| DIS5K | 470 | Boundary F1 | 0.447665 | **0.449401** |
| DUTS-TE | 5,019 | IoU | 0.548874 | **0.549755** |
| DUTS-TE | 5,019 | MAE | 0.080121 | **0.079333** |
| DUTS-TE | 5,019 | Boundary F1 | 0.332511 | **0.334881** |
| P3M | 1,000 | IoU | 0.898705 | **0.898852** |
| P3M | 1,000 | MAE | 0.030785 | **0.030664** |
| P3M | 1,000 | Boundary F1 | 0.599912 | **0.601897** |

The only regression is DIS5K IoU (-0.001252). DIS5K MAE and boundary F1 still
improve, as do all three combined metrics and all three metrics on DUTS-TE and
P3M.

## Runtime and export checks

- RTX 3090, 512 px, 200 repeats: V8 median 12.33 ms; V9 median 12.19 ms.
  The difference is within benchmark noise and shows no speed regression.
- ONNX parity passed at 96x128, 97x129 and 512x512 inputs.
- Maximum PyTorch-to-ONNX absolute error: 0.00001484.
- Guided keep and remove locks remain exact.
- SHA-256: `fb2d32ee2c07c9b7bc8f2adc2b5947d11c901c4bb2a8f1d23859c5107bc01295`

## Why V9 stays simple

The most promising papers and implementations pointed toward spatial attention
and stronger detail supervision. They were useful research directions, but the
tested variants did not clear the release gate at this model size. V9 therefore
ships the better weights and leaves the runtime alone.

Research references:

- [Coordinate Attention for Efficient Mobile Network Design](https://arxiv.org/abs/2103.02907)
- [Highly Efficient Natural Image Matting](https://www.bmva-archive.org.uk/bmvc/2021/assets/papers/1642.pdf)
- [GAPNet: Guided Attention Prior for Efficient Image Matting](https://arxiv.org/abs/2508.07585)
