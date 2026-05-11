# Technical Specifications: VGGT + GaussianPro 3D Lifting Pipeline

**Status:** Implemented and validated  
**Date:** 2026-05-12  
**PRD:** [`../vggt_gaussianpro_prd.md`](../vggt_gaussianpro_prd.md)  
**Scope:** This document describes the pipeline as **built and verified**, recording actual implementation decisions, data contracts, CLI interfaces, and observed deviations from the PRD.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Directory Layout](#2-directory-layout)
3. [Stage A — Frame Extraction (`get_frame.py`)](#3-stage-a--frame-extraction-get_framepy)
4. [Stage B — VGGT Geometry Estimation (`vggt_inference.py`)](#4-stage-b--vggt-geometry-estimation-vggt_inferencepy)
5. [Stage C — GaussianPro Optimization (`gaussianpro_train.py`)](#5-stage-c--gaussianpro-optimization-gaussianpro_trainpy)
6. [Stage D — Rendering (`gaussianpro_render.py`)](#6-stage-d--rendering-gaussianpro_renderpy)
7. [Utility: `utils/depth_to_normal.py`](#7-utility-utilsdepth_to_normalpy)
8. [Orchestration Scripts](#8-orchestration-scripts)
9. [Data Contracts](#9-data-contracts)
10. [Environment and Dependencies](#10-environment-and-dependencies)
11. [Observed Run: `index_0003`](#11-observed-run-index_0003)
12. [Deviations from PRD](#12-deviations-from-prd)
13. [Known Issues and Quirks](#13-known-issues-and-quirks)

---

## 1. Overview

The pipeline takes a CogVideoX-generated MP4 and produces a trained 3D Gaussian Splatting scene in four sequential stages:

```
CogVideoX MP4
      │
      ▼  [Stage A]  get_frame.py
      │  Evenly samples N frames → 0.png … (N-1).png
      │  Output: data/images/{dataset}/
      │
      ▼  [Stage B]  vggt_inference.py
      │  VGGT single-pass → camera extrinsics/intrinsics, depth maps,
      │  depth confidence maps, surface normals, dense 3D point map.
      │  Exports COLMAP binary (cameras.bin, images.bin, points3D.bin)
      │  + colored PLY for visualisation.
      │  Output: data/scenes/{dataset}/
      │
      ▼  [Stage C]  gaussianpro_train.py
      │  GaussianPro optimization with progressive propagation.
      │  Optionally injects VGGT depth maps and normals as priors.
      │  Output: data/scenes/{dataset}/output_{iter}_gp[_depth_prior]/
      │
      ▼  [Stage D]  gaussianpro_render.py  (optional)
         Renders train and/or test views from trained Gaussians.
         Output: output_{iter}_gp[_depth_prior]/train/ and/or test/
```

This pipeline is **self-contained** under `vggt_gaussianpro/` and does not modify any file under `instantsplat/`, `cogvideo/`, or `src/`.

---

## 2. Directory Layout

```
vggt_gaussianpro/
├── pipeline.sh                 single-scene end-to-end runner
├── batch_pipeline.sh           multi-scene batch runner
├── get_frame.py                Stage A: video → PNG frames
├── vggt_inference.py           Stage B: VGGT geometry estimation
├── gaussianpro_train.py        Stage C: GaussianPro optimization wrapper
├── gaussianpro_render.py       Stage D: GaussianPro render wrapper
├── utils/
│   ├── __init__.py
│   └── depth_to_normal.py      surface normal computation from point maps
├── third_party/
│   ├── vggt/                   git clone of facebookresearch/vggt
│   └── GaussianPro/            git clone of kcheng1021/GaussianPro
├── data/                       runtime data (git-ignored)
│   ├── images/                 input frames per dataset
│   └── scenes/                 output scene data per dataset
├── checkpoints/                optional local VGGT weights (git-ignored)
├── environment.yml             conda environment definition
└── README.md
```

---

## 3. Stage A — Frame Extraction (`get_frame.py`)

### Purpose

Extracts evenly-spaced frames from a CogVideoX-generated MP4 and saves them as a zero-indexed PNG sequence.

### Location

`get_frame.py` is **bundled directly** in `vggt_gaussianpro/` (not called from `../instantsplat/` as the PRD originally specified). The implementation is identical in behavior to the instantsplat version.

### Behavior

- Opens the video with OpenCV and reads `total_frames` via `CAP_PROP_FRAME_COUNT`.
- When `num_frames` is `None` or `>= total_frames`: extracts every frame.
- When `num_frames < total_frames`: samples indices via `np.linspace(0, total_frames - 1, num_frames)`.
- Frames are saved as `{extracted_count}.png` (0, 1, 2, …), NOT named by source frame index.
- Output directory is created if it doesn't exist.

### CLI

```bash
python get_frame.py <video_path> <output_dir> [num_frames]
```

| Argument | Type | Required | Default | Notes |
|---|---|---|---|---|
| `video_path` | str | yes | — | Input MP4 |
| `output_dir` | str | yes | — | Frame output directory |
| `num_frames` | int | no | None (all frames) | Evenly-spaced subsample |

### Output

```
data/images/{dataset}/
└── {0..N-1}.png    # uint8 RGB PNG, native video resolution
```

---

## 4. Stage B — VGGT Geometry Estimation (`vggt_inference.py`)

### Purpose

Runs VGGT on the extracted frames to estimate per-frame camera poses, depth maps, depth confidence maps, and surface normals, then writes COLMAP binary files consumable by GaussianPro.

### Resolution Pipeline

Images are processed in two stages before being fed to VGGT:

| Step | Resolution | Why |
|---|---|---|
| Load and square-pad | `IMG_LOAD_RESOLUTION = 1024` | Preserve detail; `load_and_preprocess_images_square` pads to square |
| Interpolate before VGGT | `VGGT_FIXED_RESOLUTION = 518` | VGGT's internal patch-token grid is tuned for 518 px |

VGGT outputs (depth maps, confidence maps, normals) are all at **518 × 518** resolution. Camera intrinsics are later rescaled to the original input resolution via `rename_colmap_recons_and_rescale_camera`.

### Two Execution Paths

#### Feedforward Path (default, `--use_ba` not set)

1. Runs `model.aggregator` + `model.camera_head` + `model.depth_head` with `torch.bfloat16` (or `float16` on pre-Ampere GPUs).
2. Builds 3D point map via `unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic)`.
3. **Confidence filtering:** pixels with `depth_conf >= CONF_THRES_VALUE` (default: 5.0) are kept.
   - **Fallback for OOD footage:** if fewer than `MIN_POINTS_BEFORE_FALLBACK = 1000` pixels pass the absolute threshold, automatically switches to keeping the top-`(100 − CONF_THRES_FALLBACK_PERCENTILE)%` = top-20% of pixels by confidence.
4. Points are randomly capped to `MAX_POINTS_FOR_COLMAP = 100_000` using `randomly_limit_trues`.
5. COLMAP reconstruction is built via `batch_np_matrix_to_pycolmap_wo_track` using camera type `PINHOLE`.

#### Bundle-Adjustment Path (`--use_ba`)

1. Runs the same VGGT forward pass, but also calls `predict_tracks` with the `aliked+sp` keypoint extractor to produce 2D feature tracks across frames.
2. Builds reconstruction via `batch_np_matrix_to_pycolmap` (with track information).
3. Runs `pycolmap.bundle_adjustment` on the result.
4. Operates at `IMG_LOAD_RESOLUTION = 1024`; intrinsics are scaled by `scale = 1024 / 518`.

### Frame Subsampling (`--max_frames`)

When the extracted frame count exceeds `--max_frames`, frames are uniformly subsampled **before** the VGGT forward pass:

```python
indices = [round(i * (total - 1) / (max_frames - 1)) for i in range(max_frames)]
```

Default in `pipeline.sh`: `MAX_FRAMES=48`. In `batch_pipeline.sh`, no default is set (unlimited). Recommended values:
- ≤ 48 frames for 48 GB VRAM
- ≤ 32 frames for 24 GB VRAM

### COLMAP Rescaling

After reconstruction, `rename_colmap_recons_and_rescale_camera` adjusts:
- `camera.params *= resize_ratio` where `resize_ratio = max(original_image_size) / reconstruction_resolution`.
- `camera.width`, `camera.height` set to original image dimensions.
- `principal_point = original_image_size / 2`.
- 2D point observations are shifted to original-resolution pixel coordinates.

### Auxiliary Artifacts

Beyond COLMAP binary, these are written per dataset:

| Path | Shape | Dtype | Notes |
|---|---|---|---|
| `sparse/cameras.bin` | — | COLMAP binary | Camera intrinsics |
| `sparse/images.bin` | — | COLMAP binary | World-to-camera poses (OpenCV convention, Z-forward) |
| `sparse/points3D.bin` | — | COLMAP binary | 3D point cloud |
| `sparse/points.ply` | (M, 6) | ASCII PLY | Coloured point cloud for visualisation |
| `images/{i}.png` | (H, W, 3) | uint8 | Copies of input frames (for COLMAP self-containedness) |
| `depth_maps/{i}.npy` | (518, 518) | float32 | Metric depth in world units |
| `confidence_maps/{i}.npy` | (518, 518) | float32 | Per-pixel depth confidence (VGGT `expp1` activation; typical in-distribution range: 5–50+) |
| `normals/{i}.npy` | (3, 518, 518) | float32, [0, 1] | Surface normals stored channels-first, remapped from [-1, 1] to [0, 1] for GaussianPro |

> **Normal storage convention:** GaussianPro's `loadCam` expects normals as `(3, H, W)` in `[0, 1]` and internally applies `(n − 0.5) × 2` to recover `[-1, 1]` unit vectors. Therefore `vggt_inference.py` saves `(normal + 1) / 2` transposed to `(3, H, W)`.

### CLI

```bash
python vggt_inference.py \
    --dataset <name>                  # required; reads data/images/<name>/
    [--images_dir <path>]             # override input images directory
    [--scenes_dir <path>]             # override output scenes root
    [--device cuda:0]                 # default: cuda if available, else cpu
    [--checkpoint <path.pt>]          # local VGGT checkpoint; downloads from HF if absent
    [--use_ba]                        # enable bundle adjustment (slower)
    [--max_reproj_error 8.0]          # BA: COLMAP reprojection error threshold
    [--shared_camera]                 # BA: all frames share one camera
    [--camera_type SIMPLE_PINHOLE]    # BA: COLMAP camera model
    [--vis_thresh 0.2]                # BA: visibility score threshold for tracks
    [--query_frame_num 8]             # BA: number of query frames for track prediction
    [--max_query_pts 4096]            # BA: maximum tracked points per query frame
    [--fine_tracking]                 # BA: use fine-grained tracking (default: True)
    [--conf_thres_value 5.0]          # feedforward: absolute confidence filter threshold
    [--max_frames <N>]                # uniformly subsample to N frames before VGGT
    [--seed 42]
    [--save_glb]                      # export scene.glb via trimesh
```

---

## 5. Stage C — GaussianPro Optimization (`gaussianpro_train.py`)

### Purpose

A wrapper around `third_party/GaussianPro/train.py` that:
1. Validates the COLMAP sparse output.
2. Creates a `sparse/0/` subdirectory with symlinks (GaussianPro always looks under `sparse/0/` but VGGT writes to `sparse/`).
3. Creates a `metricdepth/` symlink pointing to `depth_maps/` (GaussianPro's depth loader expects `metricdepth/`).
4. Invokes GaussianPro `train.py` via `subprocess.run` with the correct flags.

### Pre-flight Checks

- `_validate_sparse(scene_dir)` exits with a clear message if `sparse/images.bin` is absent or contains 0 images (≤ 8 bytes).
- `ensure_sparse_subdir(scene_dir)` creates `sparse/0/` populated with symlinks to all files in `sparse/`.
- `ensure_metricdepth_link(scene_dir)` creates/updates the `metricdepth/ → depth_maps/` symlink.

### GaussianPro Subprocess Invocation

The subprocess runs with:
- `cwd = third_party/GaussianPro/`
- `PYTHONPATH` prepended with the GaussianPro root so relative imports (`scene`, `gaussian_renderer`, …) resolve.
- `CUDA_VISIBLE_DEVICES` set from `--device` index.
- A `cost/` directory pre-created in the GaussianPro root (GaussianPro's propagation step writes debug images there; it doesn't create the directory itself).

Checkpoint save iterations: `{1, 7000, min(20000, iter), iter}` (always includes 1 and 7000).
Test render iterations: `{1, 2000, 7000, min(20000, iter), iter}`.

### Depth Prior Mode (`--use_depth_prior`)

When enabled, adds the following flags to the GaussianPro subprocess:
```
--load_depth --load_normal --depth_loss --normal_loss
```

GaussianPro reads:
- Depth from `metricdepth/{stem}.npy` (symlinked from `depth_maps/{i}.npy`)
- Normals from `normals/{stem}.npy` (channels-first `(3, H, W)` in `[0, 1]`)

Output model path: `output_{iter}_gp_depth_prior/`

### Propagation Schedule

| Parameter | Flag | Default |
|---|---|---|
| Propagation interval | `--propagation_interval` → `--propagation_interval` | 20 |
| Propagation start | `--propagation_start` → `--propagated_iteration_begin` | 1000 |
| Propagation end | `--propagation_end` → `--propagated_iteration_after` | 12000 |
| Patch size | `--patch_size` | 20 |

> Note: The PRD specified `propagation_interval=500`. The actual GaussianPro upstream default is 20, which is what the implementation uses.

### LLFF Eval Split (`--eval`)

When `--eval` is set, GaussianPro holds out every 8th frame for test renders (LLFF-style). This flag is forwarded to `train.py`. When not set (default), all frames participate in training.

### CLI

```bash
python gaussianpro_train.py \
    --dataset <name>                   # required; reads data/scenes/<name>/
    [--scenes_dir <path>]              # override scenes root
    [--iter 30000]                     # total training iterations
    [--lambda_lpips 0.3]               # accepted for CLI parity; not forwarded to GaussianPro
    [--use_depth_prior]                # inject VGGT depth + normals
    [--confidence_threshold 0.3]       # informational; not forwarded as a native GP flag
    [--propagation_interval 20]
    [--propagation_start 1000]
    [--propagation_end 12000]
    [--patch_size 20]
    [--eval]                           # LLFF-style test split during training
    [--export_ply]                     # accepted for CLI parity; GaussianPro always writes PLY
    [--device cuda:0]
    [--port 6099]                      # GaussianPro network viewer port
```

### Output Structure

```
data/scenes/{dataset}/output_{iter}_gp[_depth_prior]/
├── input.ply                        initial point cloud (copy of sparse/points.ply)
├── point_cloud/
│   ├── iteration_1/point_cloud.ply
│   ├── iteration_7000/point_cloud.ply
│   ├── iteration_20000/point_cloud.ply    (when iter >= 20000)
│   └── iteration_{iter}/point_cloud.ply
├── cameras.json
├── cfg_args                         Namespace dump of all GaussianPro train arguments
└── events.out.tfevents.*            TensorBoard logs
```

---

## 6. Stage D — Rendering (`gaussianpro_render.py`)

### Purpose

A wrapper around `third_party/GaussianPro/render.py` that resolves paths, sets environment variables, and runs the render subprocess.

### Behavior

- Runs `render.py` with `cwd = third_party/GaussianPro/` and `PYTHONPATH` prepended.
- `--iteration -1` (default) loads the latest checkpoint from `point_cloud/` subdirectories.
- `--eval` loads the LLFF test split (every 8th frame as test); should match the split used at training time for consistent metrics.
- Produces both RGB renders and depth/normal renders.

### CLI

```bash
python gaussianpro_render.py \
    --model_path  data/scenes/<dataset>/output_30000_gp_depth_prior/ \
    --source_path data/scenes/<dataset>/ \
    [--iteration -1]      # -1 = latest checkpoint; or specify an iteration number
    [--skip_train]        # skip rendering training views
    [--skip_test]         # skip rendering test views
    [--eval]              # use LLFF test split
    [--device cuda:0]
    [--quiet]             # suppress GaussianPro render.py output
```

### Output Structure

```
output_{iter}_gp[_depth_prior]/
├── train/
│   └── ours_{iter}/
│       ├── renders/          rendered RGB PNGs (00000.png, 00001.png, …)
│       ├── gt/               ground-truth RGB PNGs
│       ├── render_depth/     depth visualisation PNGs
│       └── render_normal/    normal visualisation PNGs + raw .npy files
└── test/
    └── ours_{iter}/
        ├── renders/
        ├── gt/
        ├── render_depth/
        └── render_normal/
```

---

## 7. Utility: `utils/depth_to_normal.py`

### `point_map_to_normals(points_3d: np.ndarray) → np.ndarray`

Computes per-pixel surface normals from a world-space 3D point map using central finite differences.

- Input: `(H, W, 3)` float32, world-space XYZ per pixel.
- Output: `(H, W, 3)` float32, unit surface normals.
- Horizontal tangent: `dX[:, 1:-1] = points_3d[:, 2:] - points_3d[:, :-2]` (central difference).
- Vertical tangent: `dY[1:-1, :] = points_3d[2:, :] - points_3d[:-2, :]`.
- Normal: `cross(dX, dY)`, normalised.
- Degenerate pixels (‖normal‖ < 1e-8): assigned `[0, 0, 1]`.
- Edge rows/columns use one-sided differences.

### `batch_point_map_to_normals(points_3d_batch: np.ndarray) → np.ndarray`

Vectorised wrapper over `point_map_to_normals`.

- Input: `(N, H, W, 3)` float32.
- Output: `(N, H, W, 3)` float32.
- Implemented via `np.stack` over frame axis (pure NumPy, no GPU required).

---

## 8. Orchestration Scripts

### `pipeline.sh` (single scene)

All configuration is controlled via environment variables, with sensible defaults:

| Variable | Default | Notes |
|---|---|---|
| `VIDEO_PATH` | `./data/video/video.mp4` | Input MP4 |
| `DATASET` | auto-derived | `basename(dirname(VIDEO_PATH))` or `basename(VIDEO_PATH)` without extension |
| `NUM_FRAMES` | empty (all frames) | Passed to `get_frame.py` as positional arg; empty = omitted = all frames |
| `DEVICE` | `cuda:0` | Torch device |
| `USE_BA` | empty (disabled) | Set to `--use_ba` to enable bundle adjustment |
| `SAVE_GLB` | empty (disabled) | Set to `--save_glb` to write scene.glb |
| `MAX_FRAMES` | `48` | VGGT frame budget; passed as `--max_frames` |
| `CONF_THRES` | empty (uses code default 5.0) | Absolute confidence threshold |
| `GP_ITER` | `30000` | GaussianPro training iterations |
| `GP_LAMBDA_LPIPS` | `0.3` | Perceptual loss weight (CLI parity) |
| `USE_DEPTH_PRIOR` | `1` (enabled) | Set to empty to disable |
| `SKIP_RENDER` | empty (render enabled) | Set to `1` to skip Stage D |
| `SKIP_EVAL` | empty (eval enabled) | Set to `1` to omit `--eval` from Stage D |
| `CUDA_VISIBLE_DEVICES` | `0` | Exported to environment |

Stage D (render) is skipped if `SKIP_RENDER=1`. When render runs, `--eval` is passed unless `SKIP_EVAL=1`.

### `batch_pipeline.sh` (multi-scene)

Iterates over `${DATAROOT}/${TYPE}/index_*/video.mp4` for each type in `${TYPES}`.

Key differences from `pipeline.sh`:

| Variable | Batch Default | Notes |
|---|---|---|
| `DATAROOT` | `./workspace/DimensionX/data/dimensionx_batch_sat360` | Override per batch |
| `TYPES` | `orbit dolly` | Space-separated |
| `SKIP_RENDER` | `1` (skipped) | Render disabled by default in batch mode |
| `MAX_FRAMES` | not set (unlimited) | No frame cap in batch mode |

Per-scene dataset name: `{TYPE}_{ID}` where `ID = basename(dirname(VIDEO_PATH))`.

Error handling: per-scene failures increment a `FAIL` counter and `continue` to the next scene; the batch does not abort on a single failure. Final summary prints `success=N failed=M`.

---

## 9. Data Contracts

### 9.1 Frame Extraction Output

```
data/images/{dataset}/
└── {0..N-1}.png    # uint8 RGB PNG, native CogVideoX resolution (e.g. 720×480)
                    # sorted by integer stem for temporal ordering
```

### 9.2 VGGT Scene Output (`data/scenes/{dataset}/`)

```
data/scenes/{dataset}/
├── images/
│   └── {0..N-1}.png          # uint8 RGB, copies of input frames
├── sparse/
│   ├── cameras.bin           # COLMAP binary; PINHOLE model; rescaled to original resolution
│   ├── images.bin            # COLMAP binary; world-to-camera extrinsics (OpenCV convention)
│   ├── points3D.bin          # COLMAP binary; 3D points from depth unprojection
│   └── points.ply            # ASCII PLY; coloured point cloud (visualisation)
├── depth_maps/
│   └── {i}.npy               # float32 (518, 518); metric depth in world units
├── confidence_maps/
│   └── {i}.npy               # float32 (518, 518); depth confidence (expp1 activation; typical 5–50+)
└── normals/
    └── {i}.npy               # float32 (3, 518, 518); surface normals channels-first,
                              # values in [0, 1] (remapped from [-1, 1])
```

> **Camera convention:** VGGT uses OpenCV (Z-forward, Y-down). GaussianPro uses the same convention. No axis flip is required between stages.

### 9.3 GaussianPro Training Input (auto-provisioned by `gaussianpro_train.py`)

```
data/scenes/{dataset}/
├── images/                   PNG frames
├── sparse/
│   ├── cameras.bin  …        (VGGT output)
│   └── 0/                    ← symlinks created by gaussianpro_train.py
│       ├── cameras.bin → ../cameras.bin
│       ├── images.bin  → ../images.bin
│       ├── points3D.bin → ../points3D.bin
│       └── points.ply  → ../points.ply
├── depth_maps/ {i}.npy       (VGGT output)
├── normals/    {i}.npy       (VGGT output)
└── metricdepth/              ← symlink → depth_maps/, created by gaussianpro_train.py
```

### 9.4 GaussianPro Training Output

```
data/scenes/{dataset}/output_{iter}_gp[_depth_prior]/
├── input.ply
├── point_cloud/
│   ├── iteration_1/point_cloud.ply
│   ├── iteration_7000/point_cloud.ply
│   ├── iteration_20000/point_cloud.ply
│   └── iteration_{iter}/point_cloud.ply
├── cameras.json
├── cfg_args
└── events.out.tfevents.*
```

### 9.5 GaussianPro Render Output

```
output_{iter}_gp[_depth_prior]/
├── train/ours_{iter}/
│   ├── renders/     {00000..N-1}.png    rendered RGB
│   ├── gt/          {00000..N-1}.png    ground-truth RGB
│   ├── render_depth/{00000..N-1}.png    depth visualisation
│   └── render_normal/
│       ├── {00000..N-1}.png             normal visualisation
│       └── {00000..N-1}.png.npy         raw normal arrays
└── test/ours_{iter}/                    (populated only if --eval was passed to render)
    ├── renders/     {00000..K-1}.png
    ├── gt/          {00000..K-1}.png
    ├── render_depth/{00000..K-1}.png
    └── render_normal/
```

---

## 10. Environment and Dependencies

### Conda Environment (`environment.yml`)

```
name: vggt_gp
```

Key packages:

| Package | Notes |
|---|---|
| Python 3.10 | |
| `torch >= 2.9` | CUDA 12.8 wheels via `--index-url https://download.pytorch.org/whl/cu128` |
| `torchvision` | |
| `pycolmap` | COLMAP binary I/O; required by `vggt.dependency.np_to_pycolmap` |
| `einops`, `safetensors` | VGGT core deps |
| `huggingface_hub` | Automatic weight download |
| `LightGlue` | BA path only; `git+https://github.com/cvg/LightGlue.git` |
| `trimesh` | Optional GLB export |
| `open3d`, `plyfile` | 3D utilities |
| `opencv` (conda-forge) | |
| `imageio`, `imageio-ffmpeg` | GaussianPro frame I/O |
| `tensorboard` | GaussianPro training logs |
| `tqdm`, `matplotlib`, `scipy` | |

### VGGT Installation

```bash
conda env create -f environment.yml
conda activate vggt_gp
pip install -e third_party/vggt/
```

### GaussianPro CUDA Extensions

Must be built manually after environment creation:

```bash
pip install --no-build-isolation \
    third_party/GaussianPro/submodules/diff-gaussian-rasterization \
    third_party/GaussianPro/submodules/simple-knn \
    third_party/GaussianPro/submodules/Propagation
```

> **Patches required for CUDA 12+ / C++17 toolchains (already applied in the bundled submodule source):**
> - `diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.h`: add `#include <cstdint>`
> - `simple-knn/simple_knn.cu`: add `#include <cfloat>`

### VGGT Model Weights

Downloaded automatically on first run:
```
https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt
```

For air-gapped environments, pre-download to `checkpoints/vggt_1b.pt` and pass `--checkpoint checkpoints/vggt_1b.pt` to `vggt_inference.py`.

### GPU Memory Requirements (VGGT Stage)

| Frames | VGGT VRAM |
|--------|-----------|
| 24     | ~6 GB     |
| 35     | ~8 GB     |
| 50     | ~12 GB    |
| 100    | ~21 GB    |

---

## 11. Observed Run: `index_0003`

A complete end-to-end run was performed on dataset `index_0003`:

| Item | Value |
|---|---|
| Input frames | 48 (extracted from source video, all frames) |
| VGGT resolution | 518 × 518 |
| Depth prior | enabled (`--use_depth_prior`) |
| Training iterations | 30,000 |
| Training eval split | `eval=False` (all 48 frames used for training) |
| Render eval split | `eval=True` (6 test frames rendered — every 8th of 48) |

Confirmed output files present:
- `data/images/index_0003/` — 48 PNG frames (0.png … 47.png)
- `data/scenes/index_0003/depth_maps/` — 48 `.npy` files
- `data/scenes/index_0003/confidence_maps/` — 48 `.npy` files
- `data/scenes/index_0003/normals/` — 48 `.npy` files
- `data/scenes/index_0003/sparse/` — `cameras.bin`, `images.bin`, `points3D.bin`, `points.ply`
- `data/scenes/index_0003/output_30000_gp_depth_prior/point_cloud/` — checkpoints at iterations 1, 7000, 20000, 30000
- `data/scenes/index_0003/output_30000_gp_depth_prior/train/ours_30000/` — renders for 48 train frames
- `data/scenes/index_0003/output_30000_gp_depth_prior/test/ours_30000/` — renders for 6 test frames

An intermediate run at 1,100 iterations (`output_1100_gp_depth_prior/`) is also present with checkpoints at iterations 1 and 1,100.

---

## 12. Deviations from PRD

| # | PRD Specification | Actual Implementation |
|---|---|---|
| 1 | Call `../instantsplat/get_frame.py` by relative path | `get_frame.py` bundled locally; self-contained |
| 2 | `--num_frames` default: 24–49 | Default: all frames (`NUM_FRAMES` unset); `MAX_FRAMES=48` applied at VGGT stage |
| 3 | `VGGT.from_pretrained("facebook/VGGT-1B")` | `VGGT()` + `torch.hub.load_state_dict_from_url` from HuggingFace raw URL |
| 4 | Images loaded at VGGT-native resolution | Images loaded at 1024 px (`IMG_LOAD_RESOLUTION`), then downscaled to 518 px for VGGT |
| 5 | `load_and_preprocess_images` | `load_and_preprocess_images_square` (square-pad variant) |
| 6 | COLMAP export via `demo_colmap.save_colmap_output` | Inline implementation; no import from `demo_colmap.py` |
| 7 | Normals saved as `(H, W, 3)` float32 | Saved as `(3, H, W)` float32 in `[0, 1]` range for GaussianPro `loadCam` compatibility |
| 8 | No mention of `sparse/0/` adaptation | `gaussianpro_train.py` creates `sparse/0/` with symlinks (GaussianPro expects `sparse/0/`) |
| 9 | No mention of `metricdepth/` naming | `gaussianpro_train.py` creates `metricdepth/ → depth_maps/` symlink (GaussianPro's expected name) |
| 10 | `--propagation_interval 500` | Implementation default: 20 (matches GaussianPro upstream) |
| 11 | Propagation end: not specified | `--propagation_end 12000` (maps to `--propagated_iteration_after`) |
| 12 | No eval split mentioned | `--eval` flag added to train/render for LLFF-style held-out test set |
| 13 | `torch>=2.1`, `pytorch-cuda=11.8` | `torch>=2.9`, CUDA 12.8 wheels |
| 14 | `--lambda_lpips` forwarded to GaussianPro | Accepted for CLI parity only; GaussianPro uses `lambda_dssim=0.2` natively |
| 15 | `scene.glb` optional output | Implemented via `--save_glb`; uses `trimesh` |
| 16 | `output_{iter}_gp/tb_logs/` | TensorBoard log: `events.out.tfevents.*` at root of model dir |
| 17 | No `input.ply` mentioned | GaussianPro writes `input.ply` at model path root |
| 18 | Render outputs depth/normal | Actual render also produces `render_depth/` and `render_normal/` directories |

---

## 13. Known Issues and Quirks

### pycolmap 4.x Compatibility

A bug in pycolmap 4.x's `reconstruction.write()` can produce `images.bin` with 0 images (file size ≤ 8 bytes). This has been patched in `third_party/vggt/vggt/dependency/np_to_pycolmap.py`. If `gaussianpro_train.py` exits with the "0 images" error, re-run `vggt_inference.py` for the dataset.

### OOD Footage Confidence Collapse

For out-of-distribution footage (aerial, satellite, strongly distorted lenses), VGGT's depth confidence values collapse to near-1.0 due to the `expp1` activation. The automatic percentile fallback (top-20% of pixels by confidence) activates transparently, but the resulting COLMAP point cloud may be sparser or noisier than for in-distribution CogVideoX content.

### GaussianPro `sparse/0/` Expectation

GaussianPro's `readColmapSceneInfo` hardcodes `sparse/0/` as the COLMAP subdirectory. VGGT writes directly to `sparse/`. The `ensure_sparse_subdir` function creates `sparse/0/` symlinks automatically, but if the symlinks break (e.g. on a file system that doesn't support symlinks), training will fail with a COLMAP read error.

### GaussianPro `cost/` Directory

GaussianPro's propagation step writes debug images to a hardcoded relative path `cost/` from its working directory. The training wrapper pre-creates this directory at `third_party/GaussianPro/cost/`. If the GaussianPro submodule is updated, check that this path assumption still holds.

### Normal Channels-First Convention

VGGT normals are saved as `(3, H, W)` channels-first in `[0, 1]`. This is specific to GaussianPro's `loadCam` convention. If normals are consumed by any other downstream system, they must be transposed back to `(H, W, 3)` and rescaled from `[0, 1]` to `[-1, 1]`:
```python
normal_hw3 = (np.load("0.npy").transpose(1, 2, 0) - 0.5) * 2.0
```

### LLFF Eval Split vs Training Split

`pipeline.sh` defaults to `--eval` during Stage D (render) but **not** during Stage C (train). This means:
- Gaussians are trained on all frames (no held-out views).
- At render time, every 8th frame is designated as a "test" view and rendered under `test/ours_{iter}/`.
- Metrics computed from `test/` renders reflect novel-view quality on frames the Gaussians never directly supervised, but the comparison is not strictly fair (Gaussians were fit to those frames).
- For rigorous evaluation, use `--eval` for **both** training and rendering.

### Batch Mode Frame Cap

`batch_pipeline.sh` does not set `MAX_FRAMES`, so scenes with many frames (> 48) may trigger OOM on 24–48 GB GPUs. Set `MAX_FRAMES=48` in the environment before running batch mode on large frame-count videos.

---

*End of specifications.*
