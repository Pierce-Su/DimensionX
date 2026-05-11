# DimensionX — VGGT + GaussianPro 3D Lifting Pipeline

A replacement for the legacy `instantsplat/` pipeline that uses
**[VGGT](https://github.com/facebookresearch/vggt)** (CVPR 2025 Best Paper) for geometry
estimation and **[GaussianPro](https://github.com/kcheng1021/GaussianPro)** (ICML 2024) for
Gaussian Splatting optimisation.  The existing `instantsplat/` directory is untouched.

See [`../vggt_gaussianpro_prd.md`](../vggt_gaussianpro_prd.md) for the full Product Requirement
Document.

---

## Pipeline overview

```
CogVideoX MP4
  └─ get_frame.py          →  data/images/{dataset}/0.png … N.png
  └─ vggt_inference.py     →  data/scenes/{dataset}/sparse/{cameras,images,points3D}.bin
                               data/scenes/{dataset}/depth_maps/{i}.npy
                               data/scenes/{dataset}/confidence_maps/{i}.npy
                               data/scenes/{dataset}/normals/{i}.npy
  └─ gaussianpro_train.py  →  data/scenes/{dataset}/output_*/   (TODO next step)
  └─ gaussianpro_render.py →  rendered views                     (TODO next step)
```

---

## Quick start

### 1. Environment

```bash
conda env create -f environment.yml
conda activate vggt_gp
```

Install VGGT as a package from the bundled clone:

```bash
pip install -e third_party/vggt/
```

### 2. Model weights

VGGT weights download automatically on first run from HuggingFace.
For offline / air-gapped environments, pre-download the checkpoint:

```bash
python - <<'PY'
import torch
url = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
state = torch.hub.load_state_dict_from_url(url, map_location="cpu")
torch.save(state, "checkpoints/vggt_1b.pt")
PY
```

Then pass `--checkpoint checkpoints/vggt_1b.pt` to `vggt_inference.py`.

### 3. Run the pipeline (single scene)

```bash
# Place your CogVideoX-generated video at data/video/video.mp4
bash pipeline.sh
```

Or override defaults:

```bash
VIDEO_PATH=/path/to/your/video.mp4 DATASET=my_scene bash pipeline.sh
```

To enable bundle adjustment (slower, more accurate cameras):

```bash
USE_BA="--use_ba" bash pipeline.sh
```

### 4. Run on a batch of scenes

```bash
# Expected layout:  data/video/{type}/index_{id}/video.mp4
DATAROOT=./data/video TYPES="orbit dolly" bash batch_pipeline.sh
```

---

## Running stages individually

### Stage A — frame extraction

```bash
# Extract all frames (default):
python get_frame.py data/video/video.mp4 data/images/my_scene

# Extract a specific number of frames:
python get_frame.py data/video/video.mp4 data/images/my_scene 35
```

### Stage B — VGGT geometry estimation

```bash
python vggt_inference.py --dataset my_scene --device cuda:0
```

Key outputs under `data/scenes/my_scene/`:

| Path | Description |
|---|---|
| `sparse/cameras.bin` | COLMAP binary camera params |
| `sparse/images.bin`  | COLMAP binary camera poses |
| `sparse/points3D.bin`| COLMAP binary 3-D point cloud |
| `sparse/points.ply`  | Coloured PLY for quick visualisation |
| `depth_maps/{i}.npy` | Per-frame float32 depth (518×518) |
| `confidence_maps/{i}.npy` | Per-frame depth confidence (518×518) |
| `normals/{i}.npy`    | Per-frame surface normals (518×518, XYZ) |

---

## Directory layout

```
vggt_gaussianpro/
├── pipeline.sh              single-scene end-to-end runner
├── batch_pipeline.sh        multi-scene batch runner
├── get_frame.py             Stage A: video → PNG frames
├── vggt_inference.py        Stage B: VGGT geometry estimation
├── utils/
│   ├── __init__.py
│   └── depth_to_normal.py   compute surface normals from point maps
├── third_party/
│   └── vggt/                git clone of Pierce-Su/vggt
├── data/                    runtime data (git-ignored)
│   ├── images/
│   └── scenes/
├── checkpoints/             optional local model weights (git-ignored)
├── environment.yml
└── README.md
```

---

## GPU memory requirements

| Frames | VGGT VRAM | Notes |
|--------|-----------|-------|
| 24     | ~6 GB     | Fits on a 3090/24 GB in fp16 |
| 35     | ~8 GB     | Recommended default |
| 50     | ~12 GB    | Upper end for 24 GB cards |
| 100    | ~21 GB    | Needs A100/H100 |

---

## References

- VGGT: [Wang et al., CVPR 2025](https://vgg-t.github.io/) · [facebookresearch/vggt](https://github.com/facebookresearch/vggt)
- GaussianPro: [Cheng et al., ICML 2024](https://arxiv.org/abs/2402.14650) · [kcheng1021/GaussianPro](https://github.com/kcheng1021/GaussianPro)
- DimensionX PRD: [`../vggt_gaussianpro_prd.md`](../vggt_gaussianpro_prd.md)
