# PICORN V10 hybrid alpha refinement

Date: 2026-08-16

Release candidate: 0.3.0

Refiner: `v10-j8-topology-s75`

## Decision

Keep the 312,878-parameter V9 network unchanged. It owns object semantics: what
belongs to the foreground and the approximate soft matte. A deterministic,
weight-free second stage owns the narrow uncertain band: where the visible
boundary lies and how alpha changes across it.

This is the useful split for a very small browser model. Classical propagation
cannot decide whether a bicycle, a chair behind a person or a second person is
the intended object. A small neural model can. Conversely, the neural model
does not need to spend parameters reproducing local edge-aware optimisation
that a few WebGPU stencils can compute directly from the source pixels.

## Research translated into this implementation

| Work | Relevant idea | V10 decision |
| --- | --- | --- |
| [Fast Guided Filter](https://arxiv.org/abs/1505.00996) | Subsample coefficient estimation and recover a full-resolution edge-aware result at much lower cost. | Fit alpha against luminance at quarter resolution, smooth the coefficients, then slice them at full resolution. |
| [Fast Bilateral Solver](https://arxiv.org/abs/1511.03296) | Optimise a dense result while respecting image edges. | Use a much smaller fixed four-neighbour colour graph instead of a bilateral-grid linear solver. |
| [KNN Matting](https://dingzeyu.li/projects/knn/) | Propagate known opacity through local and non-local affinities. | Retain local colour affinity; reject global KNN search because browser cost and accidental cross-object links are too high. |
| [Information-Flow Matting](https://openaccess.thecvf.com/content_cvpr_2017/html/Aksoy_Designing_Effective_Inter-Pixel_CVPR_2017_paper.html) | Treat known/unknown propagation and affinity design explicitly. | Confident alpha values become immutable anchors; propagation happens only in the unknown band. |
| [PointRend](https://openaccess.thecvf.com/content_CVPR_2020/html/Kirillov_PointRend_Image_Segmentation_As_Rendering_CVPR_2020_paper.html) | Spend high-resolution work at uncertain boundaries instead of uniform image regions. | Every pass is weighted by `4a(1-a)` and therefore becomes a no-op in confident regions. |
| [SegFix](https://arxiv.org/abs/2007.04269) | Boundary pixels are less reliable than interior pixels; correct them from trusted interiors. | Protect the class side selected by the neural model and require nearby same-class support before erosion. |
| [IndexNet](https://openaccess.thecvf.com/content_ICCV_2019/html/Lu_Indices_Matter_Learning_to_Index_for_Deep_Image_Matting_ICCV_2019_paper.html) | Ordinary bilinear upsampling loses boundary locations. | Do not upsample alpha alone: upsample guided coefficients and evaluate them against full-resolution luminance. |
| [A2U](https://openaccess.thecvf.com/content/CVPR2021/html/Dai_Learning_Affinity-Aware_Upsampling_for_Deep_Image_Matting_CVPR_2021_paper.html) | Affinity-aware upsampling can add detail with negligible model capacity. | Use source-colour affinity as a zero-parameter analogue outside the network. |
| [Matting Anything](https://openaccess.thecvf.com/content/CVPR2024W/MMFM/html/Li_Matting_Anything_CVPRW_2024_paper.html) | A semantic mask can feed a small iterative mask-to-matte stage. | Preserve the same separation, but make the second stage deterministic and weight-free rather than adding a 2.7M-parameter module. |
| [Mask2Alpha](https://arxiv.org/abs/2502.17093) | Refine a coarse mask progressively and spend sparse high-resolution work on critical detail. | Run a short fixed iteration budget and a single high-resolution guided slice. |
| [alphaMatte4K and muMatting](https://openaccess.thecvf.com/content/CVPR2026/html/Chen_alphaMatte4K__muMatting_Dataset_and_Model_for_Ultra-Micro_Precision_Alpha_CVPR_2026_paper.html) | Separate coarse localisation from sparse refinement of critical regions. | Confirms the two-stage direction; its 3D/video model is not suitable for this tiny still-image browser runtime. |
| [MODNet](https://arxiv.org/abs/2011.11961) | Decompose semantics, detail and fusion objectives for efficient matting. | Keep semantic ownership in the model and make detail refinement an explicit runtime stage. |

The papers are design evidence, not a claim that PICORN implements their full
systems. The shipped algorithm was selected by direct ablation on PICORN V9.

## Shipped algorithm

Input is the model's planar RGB tensor and soft alpha at 320, 384 or 512 px.

1. Mark alpha at or below 0.06 as trusted background and alpha at or above 0.94
   as trusted foreground.
2. Precompute four RGB Gaussian affinities per pixel with sigma 0.08.
3. Run eight Jacobi updates. The original model alpha remains the data term;
   uncertainty controls update strength, so trusted anchors never move.
4. Downsample RGB luminance and the Jacobi matte by four.
5. Fit and box-smooth guided-filter coefficients with radius 8 and epsilon
   0.0025, then evaluate them against full-resolution luminance.
6. Preserve the Jacobi foreground/background class. An uncertain foreground
   pixel can only be eroded when a trusted foreground core exists nearby; the
   inverse rule protects small holes and background gaps.

WebGPU executes the passes as WGSL compute shaders. The CPU fallback follows
the same equations and is also the numeric reference used by tests.

## Full validation result

The fixed candidate was evaluated once across all 6,489 images: DIS5K (470),
DUTS-TE (5,019) and P3M-10K (1,000), at 512 px. Lower is better for alpha error,
SAD and gradient L1; higher is better for IoU and Boundary F1.

| Source | IoU change | MAE change | MSE change | Boundary F1 change |
| --- | ---: | ---: | ---: | ---: |
| DIS5K | +0.00010 | -0.00032 | -0.00031 | +0.00387 |
| DUTS-TE | +0.00129 | -0.00029 | -0.00044 | +0.00779 |
| P3M-10K | +0.00115 | -0.00009 | -0.00022 | +0.00849 |
| **All** | **+0.00118** | **-0.00026** | **-0.00040** | **+0.00762** |

Aggregate V9 baseline and V10 result:

| Metric | V9 | V10 hybrid |
| --- | ---: | ---: |
| IoU | 0.592944 | **0.594125** |
| Alpha MAE | 0.074490 | **0.074226** |
| Alpha MSE | 0.051745 | **0.051347** |
| SAD | 19.5272 | **19.4580** |
| Gradient L1 | 3.6762 | **3.2369** |
| Boundary F1 | 0.384324 | **0.391940** |

The PyTorch/CUDA research implementation took 2.46 ms per image for the
post-process. In Chrome at 512 px, six WebGPU measurements after loading were
60.8, 11.4, 10.0, 6.9, 18.4 and 9.7 ms; excluding the first pipeline warm-up,
the mean was 11.25 ms. The JavaScript CPU fallback was about 31 ms warm on the
same workstation. Timings vary by browser and GPU.

Numeric parity was checked two ways:

- Python and JavaScript CPU output matched on a synthetic edge case after
  Float32 rounding.
- Chrome WebGPU and JavaScript CPU differed by at most
  `1.1920928955078125e-7` over the same alpha field.

## Ablations and rejected paths

| Candidate | Result | Why it did not ship |
| --- | --- | --- |
| Guided filter alone | Fast, but weaker boundary recovery. | Lacks iterative connectivity along thin, colour-consistent structures. |
| Jacobi alone | Improves class boundary, but leaves less natural soft transitions. | The guided slice gives better alpha and gradient metrics. |
| Local foreground/background colour projection | Sometimes improves translucent alpha. | Reduced aggregate Boundary F1 and was fragile when foreground and background colours overlap. |
| Unprotected guided cascade | Highest raw edge gain in the first sweep. | Worst cases erased thin opaque spokes, racks and wires. |
| Strict monotonic/dual-anchor gate | 80.9% per-image MAE win rate and smaller worst regression. | Safer but leaves measurable edge quality unused; topology protection offered the better Pareto point. |
| Six Jacobi iterations | About 2.04 ms in the CUDA reference. | Slightly lower IoU, MAE and Boundary F1 improvement than eight passes. |
| Full KNN, closed-form matting or bilateral solver | Powerful general optimisation. | More memory, neighbourhood construction and solve complexity than justified for a tiny browser runtime. |
| Another neural refinement network | Could learn difficult material priors. | Increases model/download size and duplicates work that the measured deterministic stage already improves. |

## Release boundary and remaining limits

- The ONNX model remains V9, 1,417,388 bytes, with the same SHA-256 checksum.
- Automatic inference receives V10 hybrid refinement. Brush-guided inference
  keeps its existing local trimap pipeline so user intent is never overwritten.
- WebGPU failure is non-fatal: the same image is refined on the CPU.
- The method cannot invent semantics that the model missed completely. Glass,
  camouflage, severe motion blur, crowded overlaps and multiple plausible
  foreground instances remain model/training problems.
- Future training should use boundary-band sampling and the physically coherent
  alpha data direction demonstrated by alphaMatte4K. Any model change still has
  to beat this fixed hybrid baseline at the same size.
