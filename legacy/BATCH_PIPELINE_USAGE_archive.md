## DimensionX Batch Pipeline Usage Guide

### Overview

`run_batch_pipeline.py` is a batch processing script that runs the **DimensionX** pipeline on multiple single-view images from a curated dataset and reconstructs corresponding 3D scenes.

It is designed to mirror the behavior of the Matrix-3D batch pipeline:

- **Stage 1**: DimensionX video generation (CogVideoX + S-Director LoRA) from a single image and an enhanced prompt.
- **Stage 2**: 3D reconstruction using **Dust3R + InstantSplat (Gaussian Splatting)**.
- **Enhanced prompts**: Prompts are automatically enriched using metadata fields (`style`, `scene_type`, `category`) from `metadata.json`.
- **Stage-wise execution**: You can run only a subset of stages with `--only_stages`.
- **Dataset-like outputs**: Results are organized to mirror the dataset structure by index and variant (photorealistic / stylized).

---

### Prerequisites

1. **Dataset structure** compatible with Matrix-3D:
   - `metadata.json` describing samples and prompts.
   - `photorealistic/` directory with photorealistic input images.
   - `stylized/` directory with stylized input images (optional).

2. **DimensionX environment** correctly installed (see the main `README.md` for:
   - CogVideoX / S-Director setup.
   - LoRA checkpoints.

3. **InstantSplat environment** set up under `instantsplat/`:
   - Dust3R checkpoint and InstantSplat dependencies installed (see the **3D Scene Optimization** section in `README.md`).

4. **Hardware**:
   - CUDA-capable GPU with sufficient VRAM.
   - For CogVideoX-5B I2V, you typically need a high-memory GPU (e.g., A6000/A100).

---

### Dataset Structure

#### Directory Layout

The batch script expects a dataset directory structured like this:

```text
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

#### `metadata.json` Format

`metadata.json` should contain a top-level `samples` array:

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
        "prompt": "A beautiful landscape with mountains",
        "camera_path": ["push_in"]
      },
      "stylized": {
        "filename": "image_0000.jpg",
        "prompt": "A stylized landscape with mountains",
        "camera_path": ["orbit_right"]
      }
    },
    {
      "index": 5,
      "style": "stylized",
      "scene_type": "indoor",
      "category": "room",
      "photorealistic": {
        "filename": "image_0005.jpg",
        "prompt": "A modern living room",
        "camera_path": ["pan_left", "push_in"]
      }
    }
  ]
}
```

##### Field descriptions

- **`index`** (required): Unique integer identifier for each sample.
- **`style`** (optional): High-level style label (`photorealistic`, `stylized`, `anime`, etc.).
- **`scene_type`** (optional): Scene type (`outdoor`, `indoor`, etc.).
- **`category`** (optional): Category (`landscape`, `room`, `portrait`, etc.).
- **`photorealistic`** (optional):
  - `filename`: Image filename under `photorealistic/`.
  - `prompt`: Base text prompt for the image.
  - `camera_path`: List of camera motion descriptors (currently not used for control in DimensionX batch script, but preserved for compatibility).
- **`stylized`** (optional): Same structure as `photorealistic`, but for images in `stylized/`.

---

### Prompt Enhancement

For each variant (photorealistic / stylized), the script:

1. Reads the base prompt from `metadata.json`.
2. Enhances it with metadata fields:
   - `Style: [Style Name]`
   - `Scene type: [Scene Type]`
   - `Category: [Category]`

Example:

- Base prompt: `"A beautiful landscape"`
- Metadata: `style="photorealistic"`, `scene_type="outdoor"`, `category="landscape"`
- **Enhanced prompt**:

```text
Style: Photorealistic, Scene type: Outdoor, Category: Landscape. A beautiful landscape
```

This enhanced prompt is:

- Passed to the DimensionX video generator (Stage 1).
- Saved as `prompt.txt` inside each output folder.

---

### Output Structure

Outputs are organized to mirror the dataset variant / index layout:

```text
output_base/
├── photorealistic/
│   ├── index_0000/
│   │   ├── video.mp4          # DimensionX-generated video
│   │   └── prompt.txt         # Enhanced prompt used for generation
│   ├── index_0005/
│   │   └── ...
│   └── ...
├── stylized/
│   ├── index_0005/
│   │   └── ...
│   └── ...
└── batch_summary.json         # Global summary of the batch run
```

> Note: **InstantSplat 3D outputs** (point clouds, Gaussian Splatting checkpoints, rendered videos) are stored under `instantsplat/data/scenes/<case_tag>/...` following the standard InstantSplat structure, where `case_tag` is a compact identifier such as `idx0005_photorealistic`.

---

### Stages

The DimensionX batch pipeline currently uses **two stages**:

- **Stage 1 – Video generation**
  - Input: Single-view image + enhanced prompt.
  - Output: Video file (`video.mp4`) stored in the corresponding `index_XXXX/` folder.
  - Implementations (selectable via `--video_backend`):
    - `diffusers` (default): `CogVideoXImageToVideoPipeline` + S-Director LoRA (short controllable camera-motion videos, e.g., ~6s / 48 frames).
    - `sat_360`: SAT-based **145-frame 360-degree orbit** pipeline (`cogvideo/sample_video_lowR.py` + `configs/*145.yaml` and the `DimensionX_360orbit` checkpoint).

- **Stage 2 – 3D reconstruction (Dust3R + InstantSplat)**
  - Input: `video.mp4` from Stage 1.
  - Steps:
    1. Frame extraction using `instantsplat/get_frame.py`.
    2. Camera / point cloud estimation using `instantsplat/dust3r_inference.py`.
    3. 3D Gaussian Splatting optimization using `instantsplat/3dgs.py`.
  - Output: 3DGS scene assets under `instantsplat/data/scenes/<case_tag>/...`.

You can restrict execution to a subset of stages with `--only_stages`.

---

### Basic Usage

#### Process all samples (both stages)

```bash
cd DimensionX

python run_batch_pipeline.py \
    --dataset_dir data/curated_set \
    --output_base output/dimensionx_batch \
    --lora_path path/to/orbit_lora.safetensors
```

#### Process specific indices

```bash
# Specific indices
python run_batch_pipeline.py \
    --dataset_dir data/curated_set \
    --output_base output/dimensionx_batch \
    --lora_path path/to/orbit_lora.safetensors \
    --indices 0 5 14

# Ranges (handy for parallel workers)
python run_batch_pipeline.py \
    --dataset_dir data/curated_set \
    --output_base output/dimensionx_batch \
    --lora_path path/to/orbit_lora.safetensors \
    --indices 0-24 30-39
```

#### Run only a subset of stages

```bash
# Only Stage 1 (video generation)
python run_batch_pipeline.py \
    --dataset_dir data/curated_set \
    --output_base output/dimensionx_batch \
    --lora_path path/to/orbit_lora.safetensors \
    --only_stages 1

# Only Stage 2 (3D reconstruction), assuming videos already exist
python run_batch_pipeline.py \
    --dataset_dir data/curated_set \
    --output_base output/dimensionx_batch \
    --only_stages 2
```

#### Continue on error

```bash
python run_batch_pipeline.py \
    --dataset_dir data/curated_set \
    --output_base output/dimensionx_batch \
    --lora_path path/to/orbit_lora.safetensors \
    --continue_on_error
```

---

### Command-Line Arguments

#### Required

- **`--dataset_dir`**: Path to the dataset directory containing `metadata.json`, `photorealistic/`, and (optionally) `stylized/`.

#### Recommended (Stage 1)

- **`--video_backend`** (default: `diffusers`, choices: `diffusers`, `sat_360`):
  - `diffusers`: Use Diffusers `CogVideoXImageToVideoPipeline` with an S-Director LoRA (short controllable videos, e.g., ~48 frames).
  - `sat_360`: Use the SAT-based 145-frame 360-degree orbit pipeline (via `cogvideo/sample_video_lowR.py` and the `DimensionX_360orbit` checkpoint).
- **`--lora_path`** (required when `--video_backend=diffusers`):
  - Path to the S-Director LoRA `.safetensors` file to use for camera-controlled video generation.
- **`--fps`** (default: `8`, used by the Diffusers backend):
  - Frames per second for exported videos when using `--video_backend=diffusers`.
- **`--sat_seed`** (default: `42`, used by the SAT backend):
  - Random seed for the SAT-based 145-frame 360° orbit pipeline when `--video_backend=sat_360`.
- **`--sat_t5_dir`**, **`--sat_vae_ckpt`**, **`--sat_checkpoint_dir`** (optional, for `sat_360`):
  - Override the T5 encoder dir, 3D VAE checkpoint file, and 360° checkpoint dir (the repo configs use hardcoded paths). See **[SETUP_SAT_360.md](SETUP_SAT_360.md)** for a Hugging Face–only setup script and current download links (THUDM → zai-org). If you already have the DimensionX checkpoint (e.g. under `DimensionX/checkpoints`), run the setup script with `--t5-vae-only` and pass that path as `--sat_checkpoint_dir`.

#### Stage Selection

- **`--only_stages`**:
  - Subset of stages to run (1 = video, 2 = 3D).
  - Examples:
    - `--only_stages 1`
    - `--only_stages 2`
    - `--only_stages 1 2` (default behavior if omitted).

#### Stage 2 – 3D Reconstruction

- **`--num_frames`** (default: `50`):
  - Number of evenly sampled frames extracted from each video for Dust3R.
- **`--gs_iter`** (default: `10000`):
  - Number of optimization iterations for 3D Gaussian Splatting.
- **`--lambda_lpips`** (default: `0.3`):
  - LPIPS weight in the 3DGS loss (as in `pipeline.sh`).
- **`--use_confidence`**:
  - Use Dust3R confidence maps during optimization to down-weight unreliable pixels.

#### General Settings

- **`--output_base`** (default: `output/dimensionx_batch`):

  Base directory for batch outputs. See **Output Structure** above.

- **`--device`** (default: `cuda:0`):

  CUDA device string used for both DimensionX and InstantSplat stages (e.g., `cuda:0`, `cuda:1`).

- **`--instantsplat_root`** (default: `instantsplat`):

  Path to the InstantSplat directory (relative to the DimensionX repo root by default).

- **`--cogvideo_root`** (default: `cogvideo`):

  Path to the CogVideoX SAT directory containing `sample_video_lowR.py` and the SAT configs (needed when `--video_backend=sat_360`).

- **`--indices`**:

  Subset of sample indices to process. Each value can be a single index or an inclusive range (`0-24`, `0:24`).

- **`--continue_on_error`**:

  Keep processing remaining samples even if one fails.

---

### Troubleshooting (SAT 360 backend)

When using `--video_backend=sat_360`, you may see the following. Here is what they mean and what to do.

#### “Deleting key loss.xxx from state_dict” (many lines)

**What’s happening:** The 3D VAE checkpoint was saved from a **training** run that included a full loss (discriminator, perceptual loss, etc.). For **inference**, only the encoder/decoder weights are loaded. The loader explicitly drops any key whose name starts with `loss` (see `ignore_keys: ['loss']` in the SAT config). So you get one “Deleting key …” line per such key.

**Is it a problem?** No. This is expected and harmless. The log has been trimmed to a single summary line: “Dropped N checkpoint keys (e.g. loss.…) — inference-only load.”

#### No videos produced + nvitop shows “No such process” but GPU VRAM still used

**What’s happening:**

1. **No videos:** The SAT pipeline runs in a **subprocess** (`sample_video_lowR.py`). If that process exits before writing the MP4 (e.g. out-of-memory kill, CUDA error, or Python exception during the 51-step diffusion or VAE decode), no file is written. The batch script then reports “No .mp4 files produced” or a non-zero exit code.

2. **“No such process” + VRAM in use:** The process that was using the GPU has **exited** (crashed or killed). nvitop still shows a PID that no longer exists (“No such process”). The driver may not release VRAM immediately when a process dies abnormally (e.g. OOM kill or SIGKILL), so VRAM can appear stuck until something clears it.

**What to do:**

- **See the real error:** Run the SAT script **directly** so you get the full traceback (e.g. OOM or CUDA). The batch script prints the exact command when Stage 1 fails; run it from the repo root, e.g.:
  ```bash
  cd cogvideo && python sample_video_lowR.py --base ... --input-file /abs/path/to/i2v_input.txt --output-dir /abs/path/to/raw_outputs ...
  ```
- **Free stuck VRAM:** Kill any remaining Python processes using the GPU (`nvidia-smi` → PIDs, then `kill <pid>`). If VRAM still doesn’t free, try logging out or rebooting; in some setups a driver reset is needed.
- **Reduce memory use:** Use a smaller batch or a GPU with more VRAM; the 145-frame SAT model is heavy (CogVideoX-5B + 3D VAE decode).

---

### Batch Summary

After completion, the script writes a summary file:

```text
output_base/
└── batch_summary.json
```

It contains:

- Number of samples processed.
- How many photorealistic / stylized variants were successfully processed.
- Indices and details for failed samples or variants.
- Elapsed time.
- Full configuration used for the run.

Example (abridged):

```json
{
  "total_samples": 10,
  "total_processed": 18,
  "total_failed": 2,
  "elapsed_time_seconds": 3600.5,
  "configuration": {
    "only_stages": [1, 2],
    "device": "cuda:0",
    "video": {
      "lora_path": "checkpoints/orbit_up_45_lora_weights.safetensors",
      "fps": 8
    },
    "reconstruction": {
      "num_frames": 50,
      "gs_iter": 10000,
      "lambda_lpips": 0.3,
      "use_confidence": true
    }
  },
  "results": [
    {
      "index": 0,
      "style": "photorealistic",
      "processed": ["photorealistic: image_0000.jpg"],
      "failed": []
    }
  ]
}
```

---

### Tips and Best Practices

- **Start small**: Use `--indices` to test on a few samples before running the full dataset.
- **Use `--only_stages`** to debug:
  - Run Stage 1, inspect `video.mp4` and `prompt.txt`.
  - Then run Stage 2 alone to verify 3D optimization.
- **Monitor VRAM usage** for CogVideoX; reduce concurrent jobs or use a smaller subset if you hit OOM errors.
- **Check InstantSplat logs** under `instantsplat/data/scenes/<case_tag>/...` for 3D reconstruction status.
- **Reuse the same dataset** across Matrix-3D and DimensionX, since the metadata format and directory layout are compatible.

---

### Troubleshooting

- **`Dataset directory not found`**:
  - Verify `--dataset_dir` exists and contains `metadata.json`.

- **`metadata.json not found`**:
  - Ensure the file exists and is valid JSON.

- **`Image not found` warnings**:
  - Check that `filename` entries in `metadata.json` match actual files in `photorealistic/` and `stylized/` (including case).

- **`InstantSplat directory not found` warning**:
  - Make sure `instantsplat/` is present at the expected path, or adjust `--instantsplat_root`.

- **Stage 1 failures**:
  - Confirm your `--lora_path` exists and matches the expected S-Director LoRA.
  - Verify that your GPU supports `bfloat16` for CogVideoX.

- **Stage 2 failures**:
  - Confirm Dust3R checkpoint and InstantSplat dependencies are installed.
  - Check that videos exist at `output_base/.../index_XXXX/video.mp4` when running only Stage 2.

---

### Related Files

- `run_batch_pipeline.py`: DimensionX batch pipeline implementation.
- `instantsplat/pipeline.sh`: Original reference pipeline for Dust3R + InstantSplat (used internally by our stage design).
- `README.md`: Main DimensionX documentation.

