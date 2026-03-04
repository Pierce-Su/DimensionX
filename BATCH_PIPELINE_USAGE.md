# DimensionX Batch Pipeline Usage Guide

## Overview

`run_batch_pipeline.py` runs the DimensionX image-to-video pipeline on a curated dataset. It generates 145-frame 360° orbit videos from single images using the SAT-based model (same logic as `cogvideo/sample_video_lowR.py`). The main model lives under **checkpoints/1/** and T5/VAE under **sat_weights**.

Features (aligned with Matrix-3D batch pipeline):

- **Stage-wise execution**: `--only_stages 1` (Stage 1 = image-to-video; further stages can be added later)
- **Preserved file structure**: Output mirrors the dataset layout (`output_base/photorealistic/index_XXXX/`, `stylized/...`)
- **Enhanced prompts**: Prompts are augmented with metadata (style, scene_type, category) from `metadata.json`
- **Index filtering and continue-on-error**: `--indices`, `--continue_on_error`

## Prerequisites

1. **Weights**
   - **Main model**: Directory containing `1/mp_rank_00_model_states.pt` (e.g. `checkpoints/`). Pass as **`--sat_checkpoint_dir`** (required when running stage 1). The SAT loader also expects a file named **`latest`** in that directory (content: the rank subdir name, usually `1`). If `latest` is missing but `1/mp_rank_00_model_states.pt` exists, the batch script will create `latest` for you.
   - **T5**: Text encoder directory (e.g. `sat_weights/t5-v1_1-xxl`). Pass as `--sat_t5_dir`.
   - **VAE**: 3D VAE checkpoint file (e.g. `sat_weights/vae/3d-vae.pt`). Pass as `--sat_vae_ckpt`.
   - Use `scripts/setup_sat_360_weights.sh` or `scripts/setup_sat_360_weights.py` to download T5 and VAE; see `SETUP_SAT_360.md`.

2. **Dataset**
   - Directory with `metadata.json` and image folders (see below).

3. **Environment**
   - Run from the DimensionX repo root (or set `--cogvideo_root` to the directory that contains `sample_video_lowR.py`).
   - CUDA and dependencies as required by `cogvideo/sample_video_lowR.py`.

## Dataset Structure

### Directory Layout

Same as Matrix-3D curated_set:

```
curated_set/
├── metadata.json
├── photorealistic/
│   ├── image_0000.jpg
│   ├── image_0005.jpg
│   └── ...
└── stylized/
    ├── image_0005.jpg
    ├── image_0017.jpg
    └── ...
```

### metadata.json Format

```json
{
  "samples": [
    {
      "index": 0,
      "style": "photorealistic",
      "scene_type": "outdoor",
      "category": "landscape",
      "photorealistic": {
        "filename": "image_0000.jpg",
        "prompt": "A beautiful landscape with mountains"
      },
      "stylized": {
        "filename": "image_0000.jpg",
        "prompt": "A stylized landscape with mountains"
      }
    }
  ]
}
```

- **`index`** (required): Unique sample id.
- **`style`**, **`scene_type`**, **`category`** (optional): Used for prompt enhancement.
- **`photorealistic`** / **`stylized`** (optional): Each has **`filename`** (under the same-named folder) and **`prompt`**.

`camera_path` is not used by the current DimensionX batch pipeline (SAT 360° orbit is fixed).

## Basic Usage

### Process All Images (with explicit weight paths)

Main model at `checkpoints/`, T5 and VAE under `sat_weights`:

```bash
python run_batch_pipeline.py \
    --dataset_dir data/curated_set \
    --output_base output/batch \
    --sat_checkpoint_dir checkpoints \
    --sat_t5_dir sat_weights/t5-v1_1-xxl \
    --sat_vae_ckpt sat_weights/vae/3d-vae.pt \
    --cogvideo_root cogvideo
```

### Only Stage 1 (default)

```bash
python run_batch_pipeline.py \
    --dataset_dir data/curated_set \
    --output_base output/batch \
    --only_stages 1 \
    --sat_checkpoint_dir checkpoints \
    --sat_t5_dir sat_weights/t5-v1_1-xxl \
    --sat_vae_ckpt sat_weights/vae/3d-vae.pt \
    --cogvideo_root cogvideo
```

### Process Specific Indices

```bash
# Single indices
python run_batch_pipeline.py \
    --dataset_dir data/curated_set \
    --output_base output/batch \
    --indices 0 5 14 \
    --sat_checkpoint_dir checkpoints \
    --sat_t5_dir sat_weights/t5-v1_1-xxl \
    --sat_vae_ckpt sat_weights/vae/3d-vae.pt

# Range (e.g. for parallel workers)
python run_batch_pipeline.py \
    --dataset_dir data/curated_set \
    --output_base output/batch \
    --indices 0-24 \
    --sat_checkpoint_dir checkpoints \
    --sat_t5_dir sat_weights/t5-v1_1-xxl \
    --sat_vae_ckpt sat_weights/vae/3d-vae.pt
```

### Continue on Error

```bash
python run_batch_pipeline.py \
    --dataset_dir data/curated_set \
    --output_base output/batch \
    --sat_checkpoint_dir checkpoints \
    --sat_t5_dir sat_weights/t5-v1_1-xxl \
    --sat_vae_ckpt sat_weights/vae/3d-vae.pt \
    --cogvideo_root cogvideo \
    --continue_on_error
```

## Command-Line Arguments

### Required

- **`--dataset_dir`**: Path to the curated_set directory (contains `metadata.json` and `photorealistic/`, `stylized/`).

### Output and filtering

- **`--output_base`** (default: `output/batch`): Base output directory. Layout:
  ```
  output_base/
  ├── photorealistic/
  │   ├── index_0000/
  │   │   ├── video.mp4
  │   │   └── prompt.txt
  │   └── ...
  └── stylized/
      └── ...
  ```
- **`--indices`**: Only process these sample indices. Each item can be an integer or an inclusive range (`0-24` or `0:24`). Example: `--indices 0 2 5`, `--indices 0-9 20`.
- **`--only_stages`**: Only run these stages. Currently only **1** is supported (image-to-video). Example: `--only_stages 1`.

### Stage 1: SAT 360° image-to-video

- **`--seed`** (default: `42`): Random seed for the SAT pipeline.
- **`--sat_checkpoint_dir`** (required when running stage 1): Path to the main model checkpoint directory that contains `1/mp_rank_00_model_states.pt`. The SAT loader also expects a file named `latest` in that directory (containing the text `1`). If `latest` is missing, the batch script will create it automatically. Do not omit this when running stage 1—the default config points to an author path that will not exist on your machine.
- **`--sat_t5_dir`**: Path to the T5 text encoder directory (e.g. `sat_weights/t5-v1_1-xxl`). If omitted, the script uses the path from the model config.
- **`--sat_vae_ckpt`**: Path to the 3D VAE checkpoint file (e.g. `sat_weights/vae/3d-vae.pt`). If omitted, the script uses the path from the model config.
- **`--cogvideo_root`** (default: `cogvideo`): Directory containing `sample_video_lowR.py` (and configs under `configs/`).

### General

- **`--continue_on_error`**: Do not stop the batch when one sample fails; continue with the rest.

## Output Structure

```
output_base/
├── photorealistic/
│   ├── index_0000/
│   │   ├── video.mp4      # 145-frame orbit video
│   │   └── prompt.txt     # Enhanced prompt used
│   ├── index_0005/
│   └── ...
├── stylized/
│   └── ...
└── batch_summary.json
```

### batch_summary.json

After the run, a summary is written with:

- `total_samples`, `total_processed`, `total_failed`
- `failed_sample_indices`, `failed_details`
- `elapsed_time_seconds`
- `configuration`: seed, paths, dataset_dir, output_base, only_stages
- `results`: per-sample status (processed / failed)

Use this to resume by re-running with `--indices` for failed indices and `--continue_on_error` if needed.

## Prompt Enhancement

Prompts are enhanced with metadata in the same way as Matrix-3D:

- **Style**: `"Style: <style>"`
- **Scene type**: `"Scene type: <scene_type>"`
- **Category**: `"Category: <category>"`

Example:

- Base prompt: `"A beautiful landscape"`
- Enhanced: `"Style: Photorealistic, Scene type: Outdoor, Category: Landscape. A beautiful landscape"`

The enhanced prompt is saved as `prompt.txt` in each sample’s output directory.

## Weight Paths Summary

| Component      | Typical location              | CLI argument             |
|----------------|-------------------------------|--------------------------|
| Main model     | `checkpoints/` (contains `1/`) | `--sat_checkpoint_dir` |
| T5 text encoder| `sat_weights/t5-v1_1-xxl`     | `--sat_t5_dir`          |
| 3D VAE         | `sat_weights/vae/3d-vae.pt`    | `--sat_vae_ckpt`         |

If you use `scripts/setup_sat_360_weights.sh` without `--t5-vae-only`, the layout can be:

- `sat_weights/checkpoints/` (with `1/mp_rank_00_model_states.pt`) → `--sat_checkpoint_dir sat_weights/checkpoints`
- `sat_weights/t5-v1_1-xxl` → `--sat_t5_dir sat_weights/t5-v1_1-xxl`
- `sat_weights/vae/3d-vae.pt` → `--sat_vae_ckpt sat_weights/vae/3d-vae.pt`

If the main model is at `checkpoints/` (repo root), use `--sat_checkpoint_dir checkpoints` and only pass T5/VAE from `sat_weights`.

## Tips and Troubleshooting

1. **Test one sample first**: Use `--indices 0` to verify paths and GPU memory.
2. **Paths**: Use absolute paths if you run from a different working directory.
3. **OOM**: Each sample runs a separate `sample_video_lowR.py` process; ensure one run fits in GPU memory (see cogvideo README for requirements).
4. **Resume**: From `batch_summary.json`, take `failed_sample_indices` and re-run with `--indices <those>` and `--continue_on_error`.
5. **Debug a single sample**: Run the SAT pipeline manually, e.g.:
   ```bash
   cd cogvideo
   echo "/abs/path/to/image.jpg@@Your prompt" > /tmp/i2v.txt
   python sample_video_lowR.py --base configs/cogvideox_5b_i2v_lora_145.yaml configs/inference_145.yaml \
       --seed 42 --image2video --input-type txt --input-file /tmp/i2v.txt --output-dir /tmp/out
   ```

## See Also

- `SETUP_SAT_360.md` – Downloading T5, VAE, and optional checkpoint
- `cogvideo/README.md` – SAT pipeline and configs
- `cogvideo/sample_video_lowR.py` – Single-run image-to-video script
- Matrix-3D `BATCH_PIPELINE_USAGE.md` – Same dataset layout and prompt enhancement idea
