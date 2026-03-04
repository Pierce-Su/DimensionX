# Setting up the SAT 360° pipeline (Hugging Face only)

The original DimensionX README points to **Tsinghua cloud** and **THUDM** for some weights. Those links are often down, and **THUDM models have moved to the `zai-org` organization** on Hugging Face. This guide uses **only Hugging Face** so you can get the 145-frame 360° orbit pipeline running.

---

## What you need for `--video_backend sat_360`

| Component | Purpose | Where to get it |
|-----------|--------|------------------|
| **Main model (360° checkpoint)** | 145-frame orbit model; `mp_rank_00_model_states.pt` + `latest` | **DimensionX authors**: `ShuoChen20/DimensionX_360orbit` on HF. You may already have it under e.g. `DimensionX/checkpoints/` if you followed the official README. |
| **T5 text encoder** | CogVideoX text encoder | **External (required):** `zai-org/CogVideoX1.5-5B-SAT` (subfolder `t5-v1_1-xxl`) – Tsinghua link is dead; use this. |
| **3D VAE** | CogVideoX 3D VAE | **External (required):** `zai-org/CogVideoX1.5-5B-SAT` (file `vae/3d-vae.pt`) – Tsinghua link is dead; use this. |

The **main model** for inference is the DimensionX 360° checkpoint (the file you may already have as `checkpoints/1/mp_rank_00_model_states.pt`). The **only** weights you must obtain from external sources are **T5** and **VAE** (CogVideoX components). You do **not** need a separate “base CogVideoX-5B-I2V” download for inference; the DimensionX checkpoint is the model the authors trained and distribute.

**Note (THUDM → zai-org):** Old links to `THUDM/CogVideoX-2b` or `THUDM/CogVideoX1.5-5B-SAT` should use **`zai-org/CogVideoX-2b`** and **`zai-org/CogVideoX1.5-5B-SAT`** instead.

---

## One-command setup (recommended)

From the **DimensionX repo root** you can use either:

**Option A – Python (recommended if you have `huggingface_hub`):**
```bash
python scripts/setup_sat_360_weights.py
```

**Option B – Shell:**
```bash
bash scripts/setup_sat_360_weights.sh
```

Both download into `./sat_weights` (or pass a path as first argument):

- `sat_weights/t5-v1_1-xxl/` – T5 text encoder (external)
- `sat_weights/vae/3d-vae.pt` – 3D VAE (external)
- `sat_weights/checkpoints/` – DimensionX 360° checkpoint (main model; `1/mp_rank_00_model_states.pt` + `latest`)

To use another directory:

```bash
python scripts/setup_sat_360_weights.py /path/to/my_weights
bash scripts/setup_sat_360_weights.sh /path/to/my_weights
```

### If you already have the DimensionX checkpoint

If you already downloaded the main model from the authors (e.g. under `DimensionX/checkpoints/` with `1/mp_rank_00_model_states.pt` and `latest`), you only need T5 and VAE from external sources. Run the script with **`--t5-vae-only`** so it does not re-download the 360° checkpoint:

```bash
python scripts/setup_sat_360_weights.py --t5-vae-only
# or
bash scripts/setup_sat_360_weights.sh . --t5-vae-only
```

Then point **`--sat_checkpoint_dir`** at your existing checkpoints directory (e.g. `checkpoints` or `DimensionX/checkpoints`).

**Requirements:** `huggingface_hub` (and `huggingface-cli` for the shell script). Install with:

```bash
pip install -U huggingface_hub
```

---

## Run the batch pipeline with SAT 360°

After the script finishes, run the batch pipeline and point it at the downloaded paths (adjust if you used a different output dir):

```bash
python run_batch_pipeline.py \
  --dataset_dir data/curated_set \
  --output_base output/dimensionx_batch_sat360 \
  --video_backend sat_360 \
  --sat_t5_dir sat_weights/t5-v1_1-xxl \
  --sat_vae_ckpt sat_weights/vae/3d-vae.pt \
  --sat_checkpoint_dir sat_weights/checkpoints \
  --cogvideo_root cogvideo \
  --only_stages 1
```

Use **absolute paths** if you run from another working directory:

```bash
--sat_t5_dir /workspace/DimensionX/sat_weights/t5-v1_1-xxl \
--sat_vae_ckpt /workspace/DimensionX/sat_weights/vae/3d-vae.pt \
--sat_checkpoint_dir /workspace/DimensionX/sat_weights/checkpoints
```

---

## Manual download (if you prefer)

1. **T5 and 3D VAE from zai-org** (only external components)

   ```bash
   mkdir -p sat_weights && cd sat_weights
   huggingface-cli download zai-org/CogVideoX1.5-5B-SAT \
     --local-dir . \
     --local-dir-use-symlinks False \
     --include "t5-v1_1-xxl/*" "vae/3d-vae.pt"
   ```

2. **360° checkpoint from ShuoChen20** (main model; or use your existing `checkpoints/` from the authors)

   ```bash
   mkdir -p checkpoints/1
   huggingface-cli download ShuoChen20/DimensionX_360orbit mp_rank_00_model_states.pt --local-dir ./checkpoints/1
   huggingface-cli download ShuoChen20/DimensionX_360orbit latest --local-dir ./checkpoints
   ```

Then use `--sat_t5_dir`, `--sat_vae_ckpt`, and `--sat_checkpoint_dir` as in the examples above.

---

## Summary of Hugging Face repos

| Repo | What it provides |
|------|--------------------|
| **zai-org/CogVideoX1.5-5B-SAT** | **T5** (`t5-v1_1-xxl`) and **3D VAE** (`vae/3d-vae.pt`) – the only external weights required for SAT 360° inference. Use instead of dead Tsinghua links. |
| **zai-org/CogVideoX-5b-I2V** | Diffusers-format I2V model (for `--video_backend diffusers` with LoRA). Not used by the SAT 360° pipeline. |
| **zai-org/CogVideoX-2b** | Alternative source for T5 (use `text_encoder` + `tokenizer` as `t5-v1_1-xxl`). No `3d-vae.pt` here. |
| **ShuoChen20/DimensionX_360orbit** | **Main model** for SAT 360°: 145-frame orbit checkpoint (`mp_rank_00_model_states.pt`, `latest`). Use as `--sat_checkpoint_dir`; you may already have it under `DimensionX/checkpoints/`. |
| **ShuoChen20/DimensionX_12_basic_camera_lora** etc. | Diffusers LoRAs for short (~48-frame) camera control; used with `--video_backend diffusers`, not SAT. |

If a link breaks, search Hugging Face for `CogVideoX` and `DimensionX`; prefer **zai-org** for T5/VAE and **ShuoChen20** for DimensionX checkpoints.
