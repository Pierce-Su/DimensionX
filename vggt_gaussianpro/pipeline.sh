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
#   A) get_frame.py               — extract all frames from CogVideoX MP4
#   B) vggt_inference.py          — VGGT camera estimation + point cloud (COLMAP binary)
#   C) gaussianpro_train.py       — GaussianPro optimization with progressive propagation
#   D) gaussianpro_render.py      — render trained views (optional, set SKIP_RENDER=1 to skip;
#      set SKIP_EVAL=1 to omit --eval / train-style renders only)
#   E) gaussianpro_render_path.py — novel-view sinusoidal-elevation orbit video
#      (set SKIP_RENDER_PATH=1 to skip)
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
# Flash-Attention-2 (SDPA) makes total VRAM scale approximately linearly with N.
# Empirically measured on RTX 6000 Ada (47.4 GB usable):
#   N= 80 → 32 GB | N=100 → 41 GB | N=120 → 47 GB (tight) | N=145 → OOM
# Note: the paper reports 200 frames ≈ 40 GB, measured on H100 with FlashAttention-3;
# Ada/FA2 has a larger per-frame activation footprint (~0.4 GB/frame overhead).
# Safe defaults: 110 for 48 GB VRAM | 60 for 24 GB VRAM | unset = no limit.
MAX_FRAMES="${MAX_FRAMES:-110}"
# Absolute confidence threshold for depth filtering (no-BA path).
# 3.0 keeps more points from partially-OOD orbit content while still filtering
# clear outliers. The automatic percentile fallback (keeping top 35% of pixels)
# activates when fewer than 1000 pixels pass this threshold.
CONF_THRES="${CONF_THRES:-3.0}"

# --- GaussianPro (Stage C/D) ---
GP_ITER="${GP_ITER:-30000}"         # total GaussianPro training iterations
GP_LAMBDA_LPIPS="${GP_LAMBDA_LPIPS:-0.3}"
USE_DEPTH_PRIOR="${USE_DEPTH_PRIOR:-1}"  # set to "" to disable VGGT depth prior injection
# Save only the final iteration's point cloud and checkpoint (default: yes).
# Set to "" to restore the full default schedule {1, 7000, 20000, 30000}.
GP_SAVE_ONLY_FINAL="${GP_SAVE_ONLY_FINAL:-1}"
SKIP_RENDER="${SKIP_RENDER:-}"        # set to "1" to skip Stage D rendering
SKIP_EVAL="${SKIP_EVAL:-}"            # set to "1" to omit --eval (default: pass --eval)

# --- Novel-view orbit render (Stage E) ---
SKIP_RENDER_PATH="${SKIP_RENDER_PATH:-}"   # set to "1" to skip Stage E
N_FRAMES="${N_FRAMES:-120}"               # number of frames in the orbit video
Z_AMPLITUDE_FRAC="${Z_AMPLITUDE_FRAC:-0.35}"  # elevation amplitude (fraction of orbit radius)
RESIZE="${RESIZE:-crop}"                  # crop | pad | original
FPS="${FPS:-30}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
# Reduce VRAM fragmentation; helps when many frame-count configurations are tested.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

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
echo "  render path  : ${SKIP_RENDER_PATH:+skipped}"
echo "  n_frames     : ${N_FRAMES}"
echo "  z_amp_frac   : ${Z_AMPLITUDE_FRAC}"
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

# Derive model path once — shared by Stage D and Stage E.
SUFFIX=""
[[ -n "${USE_DEPTH_PRIOR}" ]] && SUFFIX="_depth_prior"
MODEL_PATH="${SCRIPT_DIR}/data/scenes/${DATASET}/output_${GP_ITER}_gp${SUFFIX}"

# --------------------------------------------------------------------------- #
# Stage D — render trained views (optional)
# --------------------------------------------------------------------------- #
if [[ -z "${SKIP_RENDER}" ]]; then
    echo ""
    echo "[Stage D] Rendering …"

    RENDER_EVAL_ARGS=()
    [[ -z "${SKIP_EVAL}" ]] && RENDER_EVAL_ARGS+=(--eval)

    python "${SCRIPT_DIR}/gaussianpro_render.py" \
        --model_path  "${MODEL_PATH}"                          \
        --source_path "${SCRIPT_DIR}/data/scenes/${DATASET}"   \
        --device      "${DEVICE}"                              \
        "${RENDER_EVAL_ARGS[@]}"

    echo "[Stage D] Done. Renders in: ${MODEL_PATH}/train/"
fi

# --------------------------------------------------------------------------- #
# Stage E — novel-view sinusoidal-elevation orbit render
# --------------------------------------------------------------------------- #
if [[ -z "${SKIP_RENDER_PATH}" ]]; then
    echo ""
    echo "[Stage E] Rendering novel-view orbit …"

    python "${SCRIPT_DIR}/gaussianpro_render_path.py" \
        --model_path         "${MODEL_PATH}"                        \
        --source_path        "${SCRIPT_DIR}/data/scenes/${DATASET}" \
        --n_frames           "${N_FRAMES}"                          \
        --z_amplitude_frac   "${Z_AMPLITUDE_FRAC}"                  \
        --resize             "${RESIZE}"                            \
        --fps                "${FPS}"                               \
        --device             "${DEVICE}"

    echo "[Stage E] Done. Orbit video in: ${MODEL_PATH}/render/"
fi

echo ""
echo "Pipeline complete."
echo "Outputs: ${SCRIPT_DIR}/data/scenes/${DATASET}/"
