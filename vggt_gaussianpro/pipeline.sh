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
#   C) gaussianpro_train.py  — GaussianPro optimization with progressive propagation
#   D) gaussianpro_render.py — render trained views (optional, set SKIP_RENDER=1 to skip;
#      set SKIP_EVAL=1 to omit --eval / train-style renders only)
# =============================================================================

set -euo pipefail

# --------------------------------------------------------------------------- #
# Configurable variables (override via environment)
# --------------------------------------------------------------------------- #
VIDEO_PATH="${VIDEO_PATH:-}"
DATASET="${DATASET:-}"          # auto-derived from video path when empty
NUM_FRAMES="${NUM_FRAMES:-}"    # leave empty to extract ALL frames (default)
DEVICE="${DEVICE:-cuda:0}"
USE_BA="${USE_BA:-}"            # set to "--use_ba" to enable bundle adjustment
SAVE_GLB="${SAVE_GLB:-}"        # set to "--save_glb" to write scene.glb
# Maximum frames fed to VGGT (prevents OOM on large frame counts).
# Frames are uniformly subsampled when the extracted count exceeds this value.
# Recommended: 48 for 48 GB VRAM | 32 for 24 GB VRAM | unset = no limit.
MAX_FRAMES="${MAX_FRAMES:-48}"
# Absolute confidence threshold for depth filtering (no-BA path).
# Leave unset to use the default (5.0, matches VGGT demo_colmap.py).
# For aerial / satellite / OOD footage the automatic percentile fallback
# will activate regardless, so you usually don't need to set this manually.
CONF_THRES="${CONF_THRES:-}"

# --- GaussianPro (Stage C/D) ---
GP_ITER="${GP_ITER:-30000}"         # total GaussianPro training iterations
GP_LAMBDA_LPIPS="${GP_LAMBDA_LPIPS:-0.3}"
USE_DEPTH_PRIOR="${USE_DEPTH_PRIOR:-1}"  # set to "" to disable VGGT depth prior injection
# Save only the final iteration's point cloud and checkpoint (default: yes).
# Set to "" to restore the full default schedule {1, 7000, 20000, 30000}.
GP_SAVE_ONLY_FINAL="${GP_SAVE_ONLY_FINAL:-1}"
SKIP_RENDER="${SKIP_RENDER:-}"      # set to "1" to skip Stage D rendering
SKIP_EVAL="${SKIP_EVAL:-}"            # set to "1" to omit --eval (default: pass --eval)

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
echo "  DimensionX VGGT + GaussianPro Pipeline"
echo "  video        : ${VIDEO_PATH}"
echo "  dataset      : ${DATASET}"
echo "  frames       : ${NUM_FRAMES:-all}"
echo "  max_frames (VGGT): ${MAX_FRAMES:-unlimited}"
echo "  device       : ${DEVICE}"
echo "  BA           : ${USE_BA:-disabled}"
echo "  GP iter      : ${GP_ITER}"
echo "  depth prior  : ${USE_DEPTH_PRIOR:-disabled}"
echo "  save only final : ${GP_SAVE_ONLY_FINAL:-no (full schedule)}"
echo "  render       : ${SKIP_RENDER:+skipped}"
if [[ -z "${SKIP_EVAL}" ]]; then
    echo "  eval render  : yes (--eval)"
else
    echo "  eval render  : no (--eval omitted)"
fi
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
    --dataset    "${DATASET}"  \
    --device     "${DEVICE}"   \
    ${USE_BA}                  \
    ${SAVE_GLB}                \
    ${MAX_FRAMES:+--max_frames "${MAX_FRAMES}"} \
    ${CONF_THRES:+--conf_thres_value "${CONF_THRES}"}

echo "[Stage B] Done. Scene in: ${SCRIPT_DIR}/data/scenes/${DATASET}/"

# --------------------------------------------------------------------------- #
# Stage C — GaussianPro optimisation
# --------------------------------------------------------------------------- #
echo ""
echo "[Stage C] Running GaussianPro …"

# Build save/checkpoint args: restrict to GP_ITER only when
# GP_SAVE_ONLY_FINAL is set, otherwise let gaussianpro_train.py use
# its default schedule {1, 7000, 20000, GP_ITER}.
SAVE_ITER_ARGS=()
if [[ -n "${GP_SAVE_ONLY_FINAL}" ]]; then
    SAVE_ITER_ARGS+=(
        --save_iterations        "${GP_ITER}"
        --checkpoint_iterations  "${GP_ITER}"
    )
fi

python "${SCRIPT_DIR}/gaussianpro_train.py" \
    --dataset       "${DATASET}"       \
    --iter          "${GP_ITER}"       \
    --lambda_lpips  "${GP_LAMBDA_LPIPS}" \
    ${USE_DEPTH_PRIOR:+--use_depth_prior} \
    --device        "${DEVICE}"        \
    "${SAVE_ITER_ARGS[@]}"

echo "[Stage C] Done."

# --------------------------------------------------------------------------- #
# Stage D — render (optional)
# --------------------------------------------------------------------------- #
if [[ -z "${SKIP_RENDER}" ]]; then
    echo ""
    echo "[Stage D] Rendering …"

    SUFFIX=""
    [[ -n "${USE_DEPTH_PRIOR}" ]] && SUFFIX="_depth_prior"
    MODEL_PATH="${SCRIPT_DIR}/data/scenes/${DATASET}/output_${GP_ITER}_gp${SUFFIX}"

    RENDER_EVAL_ARGS=()
    [[ -z "${SKIP_EVAL}" ]] && RENDER_EVAL_ARGS+=(--eval)

    python "${SCRIPT_DIR}/gaussianpro_render.py" \
        --model_path  "${MODEL_PATH}"                          \
        --source_path "${SCRIPT_DIR}/data/scenes/${DATASET}"   \
        --device      "${DEVICE}"                              \
        "${RENDER_EVAL_ARGS[@]}"

    echo "[Stage D] Done. Renders in: ${MODEL_PATH}/train/"
fi

echo ""
echo "Pipeline complete."
echo "Outputs: ${SCRIPT_DIR}/data/scenes/${DATASET}/"
