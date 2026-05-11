# Product Requirement Document: VGGT + GaussianPro 3D Lifting Pipeline

**Status:** Draft  
**Authors:** DimensionX Engineering  
**Date:** 2026-05-11  
**Related Work:** [DUSt3R + InstantSplat pipeline](instantsplat/) (existing, untouched)

---

## 1. Executive Summary

This document specifies a new 3D scene lifting pipeline for the DimensionX project that replaces the existing DUSt3R + InstantSplat stack with **VGGT** (Visual Geometry Grounded Transformer, CVPR 2025 Best Paper) for geometry estimation and **GaussianPro** (ICML 2024) for Gaussian Splatting optimization. The new pipeline is additive — the existing `instantsplat/` directory and all its scripts remain completely untouched. The replacement delivers meaningfully faster geometry estimation, higher-fidelity pose/depth predictions, and better Gaussian densification for the texture-light synthetic scenes that CogVideoX tends to produce.

---

## 2. Background and Motivation

### 2.1 Current Pipeline (DUSt3R + InstantSplat / vanilla 3DGS)

The existing lifting pipeline under `instantsplat/` follows five sequential stages:

```
CogVideo MP4
  └─ get_frame.py ──────────── PNG sequence  (instantsplat/data/images/{dataset}/)
  └─ dust3r_inference.py ───── COLMAP-text + points3D.ply  (data/scenes/{dataset}/sparse/0/)
  └─ 3dgs.py ───────────────── Optimized Gaussians  (data/scenes/{dataset}/output_*/point_cloud/)
  └─ render.py ─────────────── Rendered views
```

**DUSt3R** (`instantsplat/dust3r/`) runs iterative pairwise inference followed by a `PointCloudOptimizer` global alignment (300 iterations, `make_pairs(..., scene_graph='complete')`). On a scene with 35 frames this produces O(N²) pairs and takes several minutes even on an A100.

**Vanilla 3DGS** (as used in `instantsplat/3dgs.py`) employs the original split-and-clone densification strategy. It works well on scenes with rich SfM point clouds, but CogVideoX-generated synthetic scenes frequently have large texture-less or smoothly-shaded regions where SfM-seeded point clouds are sparse, leading to floaters and under-reconstruction in those areas.

### 2.2 Why VGGT over DUSt3R

| Property | DUSt3R | VGGT |
|---|---|---|
| Approach | Iterative pairwise + global alignment | Single feed-forward transformer pass |
| Speed (35 frames, H100) | ~3–5 min | < 5 seconds |
| Complexity | O(N²) pairs + 300-iter optimization | O(N) with attention; no iterative refinement |
| Output format | COLMAP text (`.txt`) | COLMAP binary (`.bin`) — natively compatible with gsplat, GaussianPro |
| Pose accuracy | Good | State-of-the-art (AUC@30: 89.98 on Co3D) |
| Bundle adjustment | No | Optional (`--use_ba` flag) |
| Depth predictions | Yes | Yes (higher quality, per-pixel confidence) |
| License | Apache 2.0 | Non-commercial (research); commercial checkpoint available as `VGGT-1B-Commercial` |
| Maintenance | Stable, no active development | Actively maintained; CVPR 2025 Best Paper |

### 2.3 Why GaussianPro over Vanilla 3DGS

| Property | Vanilla 3DGS (InstantSplat) | GaussianPro |
|---|---|---|
| Densification | Split + clone from photometric gradient | Progressive propagation guided by rendered depth/normals + patch matching |
| Texture-less surfaces | Frequently under-reconstructed | Explicitly addressed via MVS-inspired propagation |
| Initialization dependency | High | Lower (propagation fills gaps in sparse init) |
| Temporal ordering requirement | None | Yes — images must be in temporal order (already satisfied by `get_frame.py`) |
| PSNR gain on typical scenes | Baseline | +0.5–1.15 dB (Waymo and YouTube evaluations) |
| Input format | COLMAP text or binary | COLMAP binary or text (compatible) |
| Depth prior support | No | Yes — depth maps can be injected for better normal estimation |

---

## 3. Goals and Non-Goals

### 3.1 Goals

- Implement a complete, standalone pipeline at `vggt_gaussianpro/` that takes a CogVideoX-generated MP4 as input and produces a trained 3D Gaussian scene as output.
- Replace DUSt3R with VGGT for camera pose and dense point cloud estimation.
- Replace vanilla 3DGS with GaussianPro for Gaussian optimization with progressive densification.
- Produce COLMAP-binary output from VGGT that GaussianPro can consume without conversion.
- Expose depth maps and confidence maps from VGGT as optional priors for GaussianPro's propagation step.
- Match or exceed the reconstruction quality of the existing pipeline, with significantly lower geometry-estimation latency.
- Provide a `pipeline.sh` and a `batch_pipeline.sh` with identical CLI conventions to the existing instantsplat equivalents.

### 3.2 Non-Goals

- Modifying any file under `instantsplat/`, `cogvideo/`, or `src/`.
- Replacing or modifying CogVideoX inference.
- Real-time or online 3D reconstruction.
- Training VGGT or GaussianPro from scratch.
- Supporting unordered image sets (GaussianPro's current public release requires temporal ordering).

---

## 4. Proposed Pipeline Architecture

### 4.1 High-Level Data Flow

```
CogVideo MP4  (outputs/{cnt}_{prompt}/{index}/000000.mp4)
      │
      ▼
[Stage A]  get_frame.py                 (shared utility, unchanged)
      │    Evenly samples N frames → 0.png, 1.png, ..., N-1.png
      │    Output: vggt_gaussianpro/data/images/{dataset}/
      │
      ▼
[Stage B]  vggt_inference.py            (NEW — replaces dust3r_inference.py)
      │    VGGT single-pass → extrinsics, intrinsics, depth maps, point maps,
      │    confidence maps → exports COLMAP binary + per-frame depth NPZ
      │    Output: vggt_gaussianpro/data/scenes/{dataset}/
      │               ├── images/          (copies of input frames)
      │               ├── depth_maps/      (float32 .npy, one per frame)
      │               ├── confidence_maps/ (float32 .npy, one per frame)
      │               ├── normals/         (float32 .npy derived from depth+extrinsics)
      │               └── sparse/
      │                   ├── cameras.bin
      │                   ├── images.bin
      │                   └── points3D.bin
      │
      ▼
[Stage C]  gaussianpro_train.py         (NEW — replaces 3dgs.py)
      │    Loads COLMAP binary → initializes Gaussians from points3D.bin
      │    Progressive propagation using depth_maps/ + confidence_maps/ as priors
      │    Output: vggt_gaussianpro/data/scenes/{dataset}/output_{iter}_gp/
      │               ├── point_cloud/iteration_*/point_cloud.ply
      │               ├── cameras.json
      │               └── cfg_args
      │
      ▼
[Stage D]  gaussianpro_render.py        (NEW — thin wrapper around GaussianPro render)
           Renders train/novel views from trained Gaussians
           Output: vggt_gaussianpro/data/scenes/{dataset}/output_{iter}_gp/train/  (renders)
```

### 4.2 Directory Layout

```
vggt_gaussianpro/
├── pipeline.sh               # single-scene end-to-end script
├── batch_pipeline.sh         # multi-scene batch runner
├── vggt_inference.py         # Stage B: VGGT geometry estimation
├── gaussianpro_train.py      # Stage C: GaussianPro optimization
├── gaussianpro_render.py     # Stage D: render trained scene
├── utils/
│   └── depth_to_normal.py    # convert metric depth + extrinsics → normal map
├── data/
│   ├── images/               # input frames per dataset (git-ignored)
│   └── scenes/               # output scenes per dataset (git-ignored)
├── third_party/
│   ├── vggt/                 # git submodule: facebookresearch/vggt
│   └── GaussianPro/          # git submodule: kcheng1021/GaussianPro
├── environment.yml           # conda env for this pipeline
└── README.md
```

---

## 5. Component Specifications

### 5.1 Stage A — Frame Extraction

**Script:** Reuse `instantsplat/get_frame.py` (no copy needed; call it by relative path from `pipeline.sh`).

**CLI:**
```bash
python ../instantsplat/get_frame.py \
    <VIDEO_PATH> \
    ./data/images/<DATASET> \
    <NUM_FRAMES>
```

**Constraints:**
- Frames are saved as `0.png, 1.png, ..., (N-1).png` (integer stems, zero-indexed). VGGT `load_and_preprocess_images` is agnostic to filename; the pipeline should sort by integer stem to preserve temporal order for GaussianPro.
- Recommended `NUM_FRAMES`: 24–49. VGGT benchmarks show <1 s inference and <12 GB VRAM for up to 50 frames on an H100; 24 frames fit in ~6 GB.

---

### 5.2 Stage B — VGGT Geometry Estimation (`vggt_inference.py`)

#### 5.2.1 Model Loading

```python
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map

model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
```

For reproducible offline environments, the checkpoint can be pre-downloaded to `vggt_gaussianpro/checkpoints/vggt_1b.pt` and loaded via `torch.hub.load_state_dict_from_url`.

#### 5.2.2 Inference

```python
images = load_and_preprocess_images(sorted_image_paths).to(device)  # (N, 3, H, W)
with torch.no_grad():
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        predictions = model(images)
        # predictions keys: extrinsic (N,4,4), intrinsic (N,3,3),
        #                   depth_map (N,H,W), depth_conf (N,H,W),
        #                   point_map (N,H,W,3), point_conf (N,H,W)
```

**Depth-based point map** (used for initialization, more accurate than direct point map branch):

```python
world_points = unproject_depth_map_to_point_map(
    predictions["depth_map"],   # (N, H, W)
    predictions["extrinsic"],   # (N, 4, 4), world-from-camera, OpenCV convention
    predictions["intrinsic"],   # (N, 3, 3)
)
```

#### 5.2.3 COLMAP Binary Export

VGGT's `demo_colmap.py` already implements full COLMAP binary export. `vggt_inference.py` should **import and call** that logic rather than re-implement it:

```python
# Pseudo-code; exact imports depend on vggt submodule layout
from third_party.vggt.demo_colmap import save_colmap_output

save_colmap_output(
    predictions=predictions,
    image_paths=sorted_image_paths,
    output_dir=scene_dir / "sparse",
    use_ba=args.use_ba,            # optional bundle adjustment
)
```

This produces:
```
data/scenes/{dataset}/sparse/
├── cameras.bin
├── images.bin
└── points3D.bin
```

#### 5.2.4 Auxiliary Outputs

Beyond COLMAP, `vggt_inference.py` should also write:

| Path | Format | Purpose |
|---|---|---|
| `data/scenes/{dataset}/depth_maps/{i}.npy` | float32 NumPy | Per-frame metric depth, passed to GaussianPro for normal estimation |
| `data/scenes/{dataset}/confidence_maps/{i}.npy` | float32 NumPy | Per-frame depth confidence, used as per-pixel loss weight in GaussianPro |
| `data/scenes/{dataset}/images/{i}.png` | uint8 RGB PNG | Aligned image copies (VGGT may crop/resize; save the preprocessed version for consistency) |
| `data/scenes/{dataset}/scene.glb` | GLB | Optional; visualisation aid |

**Normal maps** are derived from depth maps and extrinsics via `utils/depth_to_normal.py` using finite-difference cross-products on the unprojected point map. GaussianPro uses normals to orient newly-propagated Gaussians.

#### 5.2.5 CLI

```
python vggt_inference.py \
    --dataset <name>           # reads from data/images/<name>/
    --device cuda:0
    [--use_ba]                 # enable VGGT bundle adjustment (slower but more accurate)
    [--max_query_pts 4096]     # BA parameter
    [--query_frame_num 8]      # BA parameter
    [--save_glb]               # export scene.glb
```

---

### 5.3 Stage C — GaussianPro Optimization (`gaussianpro_train.py`)

#### 5.3.1 Overview

GaussianPro extends vanilla 3DGS with a **progressive propagation** phase that fires between densification steps. After an initial warm-up using the SfM-seeded Gaussians (from `points3D.bin`), the propagation module:

1. Renders depth and normal maps from the current Gaussian set.
2. Runs patch matching (ACMH-style) on rendered and reference frames to find adjacent surface patches.
3. Propagates existing Gaussians into poorly-covered regions using the matched patch geometry to assign accurate position and orientation to new Gaussians.

VGGT's depth maps and confidence maps serve as **priors** to the propagation module: they provide reference depth for the patch matching and can weight the propagation to focus on high-confidence regions.

#### 5.3.2 Data Input

GaussianPro expects a COLMAP scene at `data/scenes/{dataset}/`:

```
data/scenes/{dataset}/
├── images/          ← copies of input frames
└── sparse/
    ├── cameras.bin
    ├── images.bin
    └── points3D.bin
```

This is exactly what Stage B produces. No format conversion is needed.

#### 5.3.3 Depth Prior Integration

GaussianPro's `scene/dataset_readers.py` can optionally read a `depth_maps/` folder alongside the COLMAP files (in the GaussianPro fork). The depth files output by `vggt_inference.py` (`{i}.npy`, float32, metric depth) should be renamed/symlinked to match GaussianPro's expected naming convention (same stem as `images/{i}.png`). This mapping is handled inside `gaussianpro_train.py`.

The `--use_depth_prior` flag activates this path in GaussianPro:
- Depth maps are loaded alongside camera calibration.
- Normals derived from depth are used to orient propagated Gaussians.
- Confidence maps optionally gate which pixels participate in propagation (high-confidence regions seed propagation; low-confidence pixels are excluded from depth-normal supervision).

#### 5.3.4 Key Training Parameters

| Parameter | Recommended Default | Notes |
|---|---|---|
| `--iterations` | 30000 | Match existing `3dgs.py` default |
| `--lambda_lpips` | 0.3 | Match existing `3dgs.py` default |
| `--use_depth_prior` | True | Enable VGGT depth prior injection |
| `--propagation_interval` | 500 | Iterations between propagation steps |
| `--propagation_start` | 1000 | Warm-up before first propagation |
| `--max_propagation_pts` | 50000 | Cap on new Gaussians per step |
| `--confidence_threshold` | 0.3 | Exclude low-confidence pixels from normal supervision |

#### 5.3.5 Output Structure

```
data/scenes/{dataset}/output_{iter}_gp[_depth_prior]/
├── point_cloud/
│   └── iteration_*/point_cloud.ply
├── cameras.json
├── cfg_args
└── tb_logs/
```

The `_depth_prior` suffix is appended when `--use_depth_prior` is active, matching the existing pipeline's `_use_conf` suffix convention.

#### 5.3.6 CLI

```
python gaussianpro_train.py \
    --dataset <name>               # reads from data/scenes/<name>/
    --iter 30000
    --lambda_lpips 0.3
    [--use_depth_prior]            # inject VGGT depth maps and normals
    [--propagation_interval 500]
    [--propagation_start 1000]
    [--export_ply]                 # write final merged PLY after training
    [--device cuda:0]
```

---

### 5.4 Stage D — Rendering (`gaussianpro_render.py`)

A thin wrapper around GaussianPro's `render.py`:

```
python gaussianpro_render.py \
    --model_path data/scenes/<dataset>/output_30000_gp/ \
    --source_path data/scenes/<dataset>/
    [--skip_train] [--skip_test]
```

Outputs rendered frames under `output_30000_gp/train/` and/or `output_30000_gp/test/`.

---

### 5.5 Pipeline Orchestration Scripts

#### `pipeline.sh` (single scene)

```bash
#!/usr/bin/env bash
set -euo pipefail

DATASET="cafe"
NUM_FRAMES=35
VIDEO_PATH="./data/video/video.mp4"

export CUDA_VISIBLE_DEVICES=0

# Stage A: extract frames
python ../instantsplat/get_frame.py \
    "${VIDEO_PATH}" \
    "./data/images/${DATASET}_${NUM_FRAMES}" \
    "${NUM_FRAMES}"

# Stage B: VGGT geometry estimation
python vggt_inference.py \
    --dataset "${DATASET}_${NUM_FRAMES}" \
    --device cuda:0

# Stage C: GaussianPro optimization
python gaussianpro_train.py \
    --dataset "${DATASET}_${NUM_FRAMES}" \
    --iter 30000 \
    --lambda_lpips 0.3 \
    --use_depth_prior

# Stage D: render (optional)
# python gaussianpro_render.py \
#     --model_path "data/scenes/${DATASET}_${NUM_FRAMES}/output_30000_gp_depth_prior" \
#     --source_path "data/scenes/${DATASET}_${NUM_FRAMES}"
```

#### `batch_pipeline.sh` (multi-scene)

Mirrors the layout convention of `instantsplat/batch_pipeline.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

DATAROOT="./data/video"
NUM_FRAMES=35
TYPES=("orbit" "dolly")

export CUDA_VISIBLE_DEVICES=0

for TYPE in "${TYPES[@]}"; do
    for VIDEO_PATH in "${DATAROOT}/${TYPE}"/index_*/video.mp4; do
        ID=$(basename "$(dirname "${VIDEO_PATH}")")
        DATASET="${TYPE}_${ID}_${NUM_FRAMES}"

        echo "=== Processing ${DATASET} ==="

        python ../instantsplat/get_frame.py \
            "${VIDEO_PATH}" \
            "./data/images/${DATASET}" \
            "${NUM_FRAMES}"

        python vggt_inference.py \
            --dataset "${DATASET}" \
            --device cuda:0

        python gaussianpro_train.py \
            --dataset "${DATASET}" \
            --iter 30000 \
            --lambda_lpips 0.3 \
            --use_depth_prior
    done
done
```

---

## 6. Environment and Dependencies

A new conda environment `vggt_gp` is defined in `vggt_gaussianpro/environment.yml`:

```yaml
name: vggt_gp
channels:
  - pytorch
  - nvidia
  - conda-forge
  - defaults
dependencies:
  - python=3.10
  - pytorch>=2.1
  - torchvision
  - pytorch-cuda=11.8
  - pip
  - pip:
    - huggingface_hub
    - transformers
    - diffusers
    - opencv-python
    - pillow
    - numpy
    - scipy
    - trimesh
    - plyfile
    - tqdm
    - tensorboard
    # VGGT (install from submodule)
    # cd third_party/vggt && pip install -e .
    # GaussianPro submodule deps
    # cd third_party/GaussianPro/submodules/Propagation && cmake . && make
```

**Separate from the existing `instantsplat` conda env.** No changes to `instantsplat/requirements.txt`.

### Submodule Setup

```bash
# From vggt_gaussianpro/
git submodule add https://github.com/facebookresearch/vggt.git third_party/vggt
git submodule add https://github.com/kcheng1021/GaussianPro.git third_party/GaussianPro
git submodule update --init --recursive

# Install VGGT as a package
cd third_party/vggt && pip install -e . && cd -

# Build GaussianPro's CUDA propagation extension
cd third_party/GaussianPro/submodules/Propagation
# Edit CMakeLists.txt: set sm_XX to match your GPU (e.g. sm_86 for RTX 3090/A10)
cmake . && make
cd -
```

### Model Checkpoint

VGGT weights are downloaded automatically on first use via `VGGT.from_pretrained("facebook/VGGT-1B")`. For air-gapped environments, pre-download to:

```
vggt_gaussianpro/checkpoints/vggt_1b.pt
```

and load with:

```python
model = VGGT()
model.load_state_dict(torch.load("checkpoints/vggt_1b.pt"))
```

---

## 7. Interface Contracts and Data Formats

### 7.1 Frame Extraction Output

```
data/images/{dataset}/
└── {0..N-1}.png        # uint8 RGB, native CogVideoX resolution (typically 720×480 or 480×720)
```

### 7.2 VGGT Scene Output

```
data/scenes/{dataset}/
├── images/
│   └── {0..N-1}.png    # uint8 RGB, VGGT-resized (nearest power-of-2 ≥ input res, ≤ 518px long side)
├── depth_maps/
│   └── {i}.npy         # float32, shape (H, W), metric depth in world units
├── confidence_maps/
│   └── {i}.npy         # float32, shape (H, W), [0,1]-normalised depth confidence
├── normals/
│   └── {i}.npy         # float32, shape (H, W, 3), surface normals in world frame
└── sparse/
    ├── cameras.bin      # COLMAP binary cameras
    ├── images.bin       # COLMAP binary images (world-to-camera extrinsics)
    └── points3D.bin     # COLMAP binary 3D points (from depth unprojection)
```

> **Convention note:** VGGT uses OpenCV camera convention (Z forward, Y down). GaussianPro and the vendored `gaussian-splatting` also operate in this convention. No axis-flip is required between stages, unlike some NeRF frameworks which expect OpenGL convention.

### 7.3 GaussianPro Training Output

```
data/scenes/{dataset}/output_{iter}_gp[_depth_prior]/
├── point_cloud/
│   ├── iteration_7000/point_cloud.ply
│   ├── iteration_30000/point_cloud.ply
│   └── ...
├── cameras.json
├── cfg_args
└── tb_logs/
```

---

## 8. Performance Targets

| Stage | Current (DUSt3R+3DGS) | Target (VGGT+GaussianPro) |
|---|---|---|
| Frame extraction (35 frames) | ~5 s | ~5 s (unchanged) |
| Geometry estimation (35 frames) | ~3–5 min | < 30 s (VGGT forward pass + optional BA) |
| Gaussian optimization (30k iter) | ~15 min | ~15–20 min (+5 min overhead for propagation steps) |
| Reconstruction quality (PSNR) | Baseline | ≥ Baseline + 0.5 dB (target: match GaussianPro Waymo benchmark gain) |
| Peak VRAM (geometry stage) | ~12 GB (DUSt3R, 35 frames) | ~12 GB (VGGT, 35 frames; see runtime table in §2.2) |
| Peak VRAM (training stage) | ~10 GB | ~10–12 GB (propagation adds modest overhead) |

---

## 9. Testing and Validation Plan

### 9.1 Unit Tests

- `test_vggt_inference.py`: given a 5-frame sample clip from `instantsplat/data/`, assert that `sparse/cameras.bin`, `sparse/images.bin`, `sparse/points3D.bin`, and `depth_maps/0.npy` are produced and non-empty.
- `test_gaussianpro_train.py`: given the above sparse output, run training for 500 iterations and assert that `output_500_gp/point_cloud/iteration_500/point_cloud.ply` exists.

### 9.2 Integration Test

Run `pipeline.sh` on the existing test clip at `instantsplat/data/video/video.mp4` (35 frames, cafe scene). Compare PSNR/SSIM of `gaussianpro_render.py` output against the existing `3dgs.py` baseline on the same clip.

### 9.3 Regression Gate

The new pipeline must not be worse than the existing pipeline by more than 0.2 dB PSNR on the test clip. Geometry-estimation wall-clock time must be ≤ 60 s for a 35-frame scene on a single A100/H100.

---

## 10. Migration and Coexistence Notes

- The new pipeline lives entirely under `vggt_gaussianpro/` and shares only `instantsplat/get_frame.py` (called by relative path in the shell scripts, no modification to the original).
- The `vggt_gp` conda environment is separate from the `instantsplat` environment.
- Data directories are separate: `vggt_gaussianpro/data/` vs. `instantsplat/data/`. No path collision.
- The main `README.md` should be updated (in a separate PR) to document the new pipeline alongside the existing one, pointing to `vggt_gaussianpro/README.md` for details.

---

## 11. Open Questions and Risks

| # | Question / Risk | Owner | Resolution Path |
|---|---|---|---|
| 1 | VGGT's `demo_colmap.py` imports may need adaptation if the submodule version diverges from the README | Engineering | Pin submodule to a specific commit hash; wrap imports defensively |
| 2 | GaussianPro's public release note "does not support unordered image sets" — CogVideoX frames are temporally ordered, so this is satisfied, but edge cases (e.g. looping video) need verification | Engineering | Test with a 360° CogVideoX clip |
| 3 | GaussianPro's CUDA propagation extension (`submodules/Propagation`) requires manual CMake build and `sm_XX` editing per GPU — fragile for multi-GPU environments | DevOps | Provide a Docker image or document exact CMake flags per GPU family |
| 4 | VGGT attention memory scales as O(N²) — 100+ frame scenes may OOM on 24 GB GPUs | Engineering | Default to chunked/windowed inference mode; expose `--chunk_size` flag |
| 5 | VGGT non-commercial license restricts production use; commercial checkpoint (`VGGT-1B-Commercial`) requires an approval form | Legal/Product | Proceed with non-commercial for research; file commercial application if productising |
| 6 | GaussianPro's `--use_depth_prior` flag and depth file ingestion are in the `version 1.0` branch — the main branch may not expose this API directly | Engineering | Audit GaussianPro's `version 1.0` branch at integration time; may require a lightweight fork or patch |
| 7 | VGGT outputs depth in a normalised scene coordinate; GaussianPro's patch matching uses metric depth. VGGT's `unproject_depth_map_to_point_map` gives metric world-space depth when extrinsics are provided, so this should be resolved — but axis conventions must be validated end-to-end | Engineering | Add an assertion in `vggt_inference.py` that the mean scene depth is plausible (0.5–50 units) |

---

## 12. References

- VGGT paper: [Wang et al., CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_VGGT_Visual_Geometry_Grounded_Transformer_CVPR_2025_paper.pdf)
- VGGT repository: [facebookresearch/vggt](https://github.com/facebookresearch/vggt)
- GaussianPro paper: [Cheng et al., ICML 2024](https://arxiv.org/abs/2402.14650)
- GaussianPro repository: [kcheng1021/GaussianPro](https://github.com/kcheng1021/GaussianPro)
- Existing pipeline entry point: [`instantsplat/pipeline.sh`](instantsplat/pipeline.sh)
- Existing geometry script: [`instantsplat/dust3r_inference.py`](instantsplat/dust3r_inference.py)
- Existing GS training script: [`instantsplat/3dgs.py`](instantsplat/3dgs.py)
- DUSt3R: [Leroy et al., CVPR 2024](https://github.com/naver/dust3r)
- InstantSplat: [Fan et al., 2024](https://instantsplat.github.io/)
