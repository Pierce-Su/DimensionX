#!/usr/bin/env bash
# =============================================================================
# DimensionX — VGGT + GaussianPro lifting pipeline (single scene)
#
# Usage:
#   bash pipeline.sh                        # uses defaults below
#   VIDEO_PATH=/path/to/video.mp4 bash pipeline.sh
#   VIDEO_PATH=... DATASET=myrun bash pipeline.sh
#
# Stages:
#   A) get_frame.py          — extract all frames from CogVideoX MP4
#   B) vggt_inference.py     — VGGT camera estimation + point cloud (COLMAP binary)
#   (C and D are GaussianPro — implemented in a future step)
# =============================================================================

set -euo pipefail

# --------------------------------------------------------------------------- #
# Configurable variables (override via environment)
# --------------------------------------------------------------------------- #
VIDEO_PATH="${VIDEO_PATH:-./data/video/video.mp4}"
DATASET="${DATASET:-}"          # auto-derived from video path when empty
NUM_FRAMES="${NUM_FRAMES:-}"    # leave empty to extract ALL frames (default)
DEVICE="${DEVICE:-cuda:0}"
USE_BA="${USE_BA:-}"            # set to "--use_ba" to enable bundle adjustment
SAVE_GLB="${SAVE_GLB:-}"        # set to "--save_glb" to write scene.glb

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# --------------------------------------------------------------------------- #
# Derive a dataset name from the video path when not explicitly provided
# --------------------------------------------------------------------------- #
if [[ -z "${DATASET}" ]]; then
    # e.g. /foo/bar/my_scene/000000.mp4  →  "my_scene"
    DATASET="$(basename "$(dirname "${VIDEO_PATH}")")"
    [[ "${DATASET}" == "." ]] && DATASET="$(basename "${VIDEO_PATH%.*}")"
fi

echo "============================================================"
echo "  DimensionX VGGT Pipeline"
echo "  video   : ${VIDEO_PATH}"
echo "  dataset : ${DATASET}"
echo "  frames  : ${NUM_FRAMES:-all}"
echo "  device  : ${DEVICE}"
echo "  BA      : ${USE_BA:-disabled}"
echo "============================================================"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --------------------------------------------------------------------------- #
# Stage A — frame extraction
# --------------------------------------------------------------------------- #
echo ""
echo "[Stage A] Extracting frames …"

IMAGES_DIR="${SCRIPT_DIR}/data/images/${DATASET}"
python "${SCRIPT_DIR}/get_frame.py" \
    "${VIDEO_PATH}" \
    "${IMAGES_DIR}" \
    ${NUM_FRAMES}   # empty → positional arg omitted → defaults to all frames

echo "[Stage A] Done. Frames in: ${IMAGES_DIR}"

# --------------------------------------------------------------------------- #
# Stage B — VGGT geometry estimation
# --------------------------------------------------------------------------- #
echo ""
echo "[Stage B] Running VGGT …"

python "${SCRIPT_DIR}/vggt_inference.py" \
    --dataset  "${DATASET}" \
    --device   "${DEVICE}"  \
    ${USE_BA}               \
    ${SAVE_GLB}

echo "[Stage B] Done. Scene in: ${SCRIPT_DIR}/data/scenes/${DATASET}/"

# --------------------------------------------------------------------------- #
# Stage C — GaussianPro optimisation  (TODO: implemented in next step)
# --------------------------------------------------------------------------- #
# python "${SCRIPT_DIR}/gaussianpro_train.py" \
#     --dataset "${DATASET}" \
#     --iter 30000 \
#     --lambda_lpips 0.3 \
#     --use_depth_prior

echo ""
echo "Pipeline complete (VGGT stage)."
echo "Outputs: ${SCRIPT_DIR}/data/scenes/${DATASET}/"
