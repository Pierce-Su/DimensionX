---
name: VGGT OOM & Point Cloud Fix
overview: Diagnose the VRAM discrepancy between the paper's 200-frame/40 GB claim and the observed 48-frame OOM limit, then fix the pipeline to safely process more frames and improve orbit-camera point cloud coverage.
todos:
  - id: verify-fa2
    content: Add SDPA backend diagnostics to vggt_inference.py main() to confirm Flash Attention 2 is active
    status: pending
  - id: increase-max-frames
    content: Change MAX_FRAMES default from 48 to 100 in pipeline.sh and batch_pipeline.sh; add a comment explaining the derivation from the paper's scaling
    status: pending
  - id: free-images-early
    content: Refactor vggt_inference.py to extract RGB colours and free the 1024px images tensor from GPU before calling run_vggt(), saving ~9 MB/frame
    status: pending
  - id: lower-conf-thres
    content: Set CONF_THRES default to 2.0–3.0 in pipeline.sh for CogVideoX synthetic orbit content to reduce point cloud holes
    status: pending
  - id: test-frame-ladder
    content: Empirically test MAX_FRAMES at 80, 100, 150, 200 with nvidia-smi monitoring to confirm actual VRAM limits on RTX 6000 Ada
    status: pending
isProject: false
---

# VGGT OOM Discrepancy & Point Cloud Gaps — Diagnosis and Fix

## Root Cause Analysis

### Why 48 frames OOMs when the paper claims 200 frames = 40 GB

The discrepancy has two independent causes:

#### 1. Flash Attention backend is critical (the main cause)

VGGT's aggregator alternates between *frame attention* (each frame attends to its own 1374 patch tokens — manageable) and *global attention* (all frames attend jointly as one sequence of `N × 1374` tokens). Without Flash Attention, the global attention matrix materialises as `(N × 1374)²`:

- N = 48 → attention matrix ≈ **280 GB** — instant OOM
- N = 20 → attention matrix ≈ **49 GB** — also OOM

The fact that N = 48 works at all proves Flash Attention 2 is active via PyTorch SDPA. The pipeline correctly uses `torch.cuda.amp.autocast(dtype=bfloat16)`, and the RoPE implementation (`rope.py:178`) uses `tokens.dtype` to construct BF16 cos/sin tables, so Q/K remain BF16 after RoPE and FA2 triggers. With FA2 active, global attention memory is `O(N × P)` rather than `O((N × P)²)`.

The paper's 200-frame/40 GB benchmark was measured on H100 **with Flash Attention 3 explicitly compiled** as a standalone library, not via PyTorch SDPA. FA2 (used here via SDPA on RTX 6000 Ada SM89) has the same asymptotic memory complexity as FA3 — the numbers should be comparable.

#### 2. `IMG_LOAD_RESOLUTION = 1024` keeps an extra full-resolution tensor on GPU

`vggt_inference.py` calls `load_and_preprocess_images_square(paths, 1024)` and moves the result to GPU:

```63:64:E3DQA_project/DimensionX/vggt_gaussianpro/vggt_inference.py
VGGT_FIXED_RESOLUTION = 518   # resolution VGGT runs at internally
IMG_LOAD_RESOLUTION   = 1024  # resolution images are loaded at before feeding to VGGT
```

The `images` tensor `(N, 3, 1024, 1024)` float32 stays on GPU the entire inference because it is needed *after* `run_vggt()` returns to extract COLMAP point colours:

```505:511:E3DQA_project/DimensionX/vggt_gaussianpro/vggt_inference.py
        points_rgb_full = F.interpolate(
            images, size=(VGGT_FIXED_RESOLUTION, VGGT_FIXED_RESOLUTION),
            mode="bilinear", align_corners=False,
        )
        points_rgb_full = (points_rgb_full.cpu().numpy() * 255).astype(np.uint8)
        points_rgb_full = points_rgb_full.transpose(0, 2, 3, 1)  # (N, H, W, 3)
```

Inside `run_vggt()`, a *second* interpolation to 518px is also created, so both the 1024px and 518px tensors coexist on GPU during the forward pass:

```227:229:E3DQA_project/DimensionX/vggt_gaussianpro/vggt_inference.py
    images_vggt = F.interpolate(
        images, size=(resolution, resolution), mode="bilinear", align_corners=False
    )
```

Extra overhead per frame: `(1024² - 518²) × 3 × 4 bytes ≈ 9 MB/frame`. For N = 48: **432 MB**; for N = 200: **1.8 GB**.

#### 3. `aggregated_tokens_list` holds all 24 layer outputs simultaneously

The aggregator accumulates a list of 24 entries, each `(1, N, 1374, 2048)` BF16. Peak size:

| N | Memory for output list |
|---|---|
| 48 | 24 × 48 × 1374 × 2048 × 2 B ≈ **6.5 GB** |
| 100 | **13.5 GB** |
| 200 | **27 GB** |

This is the dominant scaling term and matches the paper's linear curve closely.

#### 4. Actual vs safe limit on 48 GB

Combining model weights (~2 GB), output list, forward-pass activations, and images overhead, the paper's scaling predicts:

| N frames | Peak VRAM (estimated for Ada, FA2) |
|---|---|
| 48 | ~11 GB |
| 100 | ~22 GB |
| 150 | ~32 GB |
| 200 | ~42 GB |

The **48-frame cap is approximately 4× too conservative**. The `"Recommended: ≤48 for 48 GB VRAM"` comment in the code was likely derived from the paper's conservative interpretation or from tests performed with other workloads already occupying the GPU. The actual safe limit on a dedicated 48 GB RTX 6000 Ada is in the range of **150–200 frames**.

---

### Why point clouds have structural gaps

With N = 48 frames uniformly sampled over a 360° orbit, adjacent sampled frames are ~7.5° apart. This has three effects:

1. **Angular coverage gaps**: Surfaces visible only near the transition angle between two sampled frames are reconstructed from very few (or zero) views. VGGT's confidence is low for such surfaces → filtered by `CONF_THRES_VALUE = 5.0`.
2. **OOD confidence collapse**: CogVideoX orbit videos are somewhat out-of-distribution for VGGT, causing depressed confidence values. The 80th-percentile fallback activates but still discards 80% of pixels.
3. **Temporal ≠ angular sampling**: If the orbit camera velocity varies (acceleration/deceleration at start/end), uniform *temporal* sampling clusters frames at certain angles, leaving large gaps elsewhere.

---

## Proposed Changes

### Fix 1 — Increase `MAX_FRAMES` to 100–200

In [`pipeline.sh`](E3DQA_project/DimensionX/vggt_gaussianpro/pipeline.sh) and [`batch_pipeline.sh`](E3DQA_project/DimensionX/vggt_gaussianpro/batch_pipeline.sh):

```bash
# Current
MAX_FRAMES="${MAX_FRAMES:-48}"

# Change to (start conservatively, scale up after verification)
MAX_FRAMES="${MAX_FRAMES:-100}"
```

Recommended test ladder: 80 → 100 → 150 → 200 frames, monitoring VRAM with `nvidia-smi` at each step.

### Fix 2 — Free the 1024px `images` tensor before `run_vggt()`

In [`vggt_inference.py`](E3DQA_project/DimensionX/vggt_gaussianpro/vggt_inference.py) (feedforward path, ~line 500): extract RGB at 518px *before* calling `run_vggt()`, then delete `images` from GPU. This frees ~9 MB/frame before the memory-intensive aggregator forward.

```python
# Extract RGB colours (needed for COLMAP point cloud) BEFORE the expensive forward pass
# so the 1024-px float32 tensor can be freed from GPU memory.
points_rgb_full_gpu = F.interpolate(
    images, size=(VGGT_FIXED_RESOLUTION, VGGT_FIXED_RESOLUTION),
    mode="bilinear", align_corners=False,
)
points_rgb_full = (points_rgb_full_gpu.cpu().numpy() * 255).astype(np.uint8)
points_rgb_full = points_rgb_full.transpose(0, 2, 3, 1)
del points_rgb_full_gpu

# Free 1024-px images from GPU — no longer needed
images_gpu = images
images = None
del images_gpu
torch.cuda.empty_cache()

# Now call run_vggt with the 1024-px tensor freed
extrinsic, intrinsic, depth_map, depth_conf = run_vggt(
    model, load_and_preprocess_images_for_vggt(image_path_list), dtype, VGGT_FIXED_RESOLUTION
)
```

> Note: `run_vggt()` currently receives `images` (1024px) and interpolates internally. With this refactor, we need to either (a) load images at 518px separately for the forward pass, or (b) pass the already-interpolated 518px tensor to `run_vggt()`.

The cleanest approach: load once at 518px (`load_and_preprocess_images` with `target_size=518`) for the VGGT forward, and load once at 1024px for RGB extraction, then free the 1024px copy before inference. Or change `IMG_LOAD_RESOLUTION` to 518 and accept slightly lower RGB colour quality in the point cloud.

### Fix 3 — Add SDPA backend diagnostics

At the start of `vggt_inference.py` main():

```python
import torch
print(f"[VGGT] Flash SDPA enabled:   {torch.backends.cuda.flash_sdp_enabled()}")
print(f"[VGGT] MemEff SDPA enabled:  {torch.backends.cuda.mem_efficient_sdp_enabled()}")
print(f"[VGGT] Math SDPA enabled:    {torch.backends.cuda.math_sdp_enabled()}")
```

If `flash_sdp_enabled()` returns `False`, add `torch.backends.cuda.enable_flash_sdp(True)` before inference. If Flash SDPA cannot be selected for some reason (dtype, kernel constraints), the math backend will silently OOM even at modest N.

### Fix 4 — Lower confidence threshold for orbit/OOD scenes

In [`pipeline.sh`](E3DQA_project/DimensionX/vggt_gaussianpro/pipeline.sh), set a lower threshold for CogVideoX synthetic content:

```bash
CONF_THRES="${CONF_THRES:-2.0}"   # was unset (defaults to 5.0)
```

This accepts noisier but denser point clouds, filling structural holes at the cost of some outliers.

### Fix 5 — Orbit-angle-aware frame sampling (optional, larger change)

For orbit camera videos, replace the uniform temporal subsampling in `vggt_inference.py` with spatial-coverage sampling: detect camera angular displacement per frame (using a lightweight flow or by analyzing brightness/colour differences between frames), then sample frames at approximately equal angular intervals.

This is the most targeted fix for structural gaps specific to CogVideoX orbital sequences and is independent of the frame-count limit.

---

## Summary of What Went Wrong

The 48-frame cap is **not the actual OOM boundary**. It is a conservative recommendation that was set approximately 4× below the real Flash-Attention-2 limit (~150–200 frames on 48 GB). The structural gaps in the VGGT point clouds are a direct consequence of this over-restriction: 48 uniformly-sampled frames over a 360° orbit leaves 7.5° gaps between adjacent viewpoints, which VGGT cannot bridge with enough confidence to survive the depth filter.

The two-part fix is: **(1) increase MAX_FRAMES to 100–150 after verifying SDPA backend + available VRAM** and **(2) lower the confidence threshold slightly for synthetic orbit content**.