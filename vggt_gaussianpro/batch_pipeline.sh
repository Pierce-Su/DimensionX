#!/usr/bin/env bash
# =============================================================================
# DimensionX — VGGT + GaussianPro lifting pipeline (batch / multi-scene)
#
# Expected layout under DATAROOT:
#   DATAROOT/
#     {type}/            e.g. "photorealistic" or "stylized"
#       index_{id}/
#         video.mp4
#
# Output datasets are named  {type}_index_{id}  so every output folder
# encodes both the style category and the scene index, e.g.:
#   data/scenes/photorealistic_index_0003/
#   data/scenes/stylized_index_0017/
#
# Only the iteration-30000 point cloud and checkpoint are saved by default
# (GP_SAVE_ONLY_FINAL=1).  Set GP_SAVE_ONLY_FINAL="" to restore the full
# default save schedule {1, 7000, 20000, 30000}.
#
# Usage:
#   bash batch_pipeline.sh
#   DATAROOT=/mnt/data/videos TYPES="photorealistic stylized" bash batch_pipeline.sh
#   SKIP_RENDER="" bash batch_pipeline.sh   # also run Stage D renders
# =============================================================================

set -euo pipefail

# --------------------------------------------------------------------------- #
# Configurable variables (override via environment)
# --------------------------------------------------------------------------- #
DATAROOT="${DATAROOT:-./workspace/DimensionX/data/dimensionx_batch_sat360}"
TYPES="${TYPES:-photorealistic stylized}"   # space-separated style categories
NUM_FRAMES="${NUM_FRAMES:-}"                # leave empty to extract ALL frames
DEVICE="${DEVICE:-cuda:0}"
USE_BA="${USE_BA:-}"               # set to "--use_ba" to enable bundle adjustment
SAVE_GLB="${SAVE_GLB:-}"           # set to "--save_glb" to write scene.glb
# Maximum frames fed to VGGT (prevents OOM on large frame counts).
MAX_FRAMES="${MAX_FRAMES:-48}"
# Absolute confidence threshold for depth filtering (no-BA path).
CONF_THRES="${CONF_THRES:-}"

# --- GaussianPro (Stage C/D) ---
GP_ITER="${GP_ITER:-30000}"
GP_LAMBDA_LPIPS="${GP_LAMBDA_LPIPS:-0.3}"
USE_DEPTH_PRIOR="${USE_DEPTH_PRIOR:-1}"     # set to "" to disable depth prior
# Save only the final iteration's point cloud and checkpoint (default: yes).
# Set to "" to restore the full default schedule {1, 7000, 20000, 30000}.
GP_SAVE_ONLY_FINAL="${GP_SAVE_ONLY_FINAL:-1}"
SKIP_RENDER="${SKIP_RENDER:-1}"             # set to "" to enable Stage D renders
SKIP_EVAL="${SKIP_EVAL:-}"                  # set to "1" to omit --eval in render

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "============================================================"
echo "  DimensionX VGGT + GaussianPro Batch Pipeline"
echo "  dataroot        : ${DATAROOT}"
echo "  types           : ${TYPES}"
echo "  frames          : ${NUM_FRAMES:-all}"
echo "  max_frames(VGGT): ${MAX_FRAMES:-unlimited}"
echo "  device          : ${DEVICE}"
echo "  BA              : ${USE_BA:-disabled}"
echo "  GP iter         : ${GP_ITER}"
echo "  depth prior     : ${USE_DEPTH_PRIOR:-disabled}"
echo "  save only final : ${GP_SAVE_ONLY_FINAL:-no (full schedule)}"
echo "  render          : ${SKIP_RENDER:+skipped}"
if [[ -z "${SKIP_EVAL}" ]]; then
    echo "  eval render     : yes (--eval)"
else
    echo "  eval render     : no (--eval omitted)"
fi
echo "============================================================"

SUCCESS=0
FAIL=0

for TYPE in ${TYPES}; do
    for VIDEO_PATH in "${DATAROOT}/${TYPE}"/index_*/video.mp4; do
        [[ -f "${VIDEO_PATH}" ]] || continue

        # e.g.  photorealistic_index_0003
        INDEX_DIR="$(basename "$(dirname "${VIDEO_PATH}")")"
        DATASET="${TYPE}_${INDEX_DIR}"

        echo ""
        echo "------------------------------------------------------------"
        echo "  Processing : ${DATASET}"
        echo "  Video      : ${VIDEO_PATH}"
        echo "------------------------------------------------------------"

        IMAGES_DIR="${SCRIPT_DIR}/data/images/${DATASET}"

        # ------------------------------------------------------------------ #
        # Stage A — frame extraction
        # ------------------------------------------------------------------ #
        echo "[A] Extracting frames …"
        if python "${SCRIPT_DIR}/get_frame.py" \
                "${VIDEO_PATH}" \
                "${IMAGES_DIR}" \
                ${NUM_FRAMES}; then
            echo "[A] Done. Frames in: ${IMAGES_DIR}"
        else
            echo "[A] FAILED for ${DATASET}" >&2
            (( FAIL++ )) || true
            continue
        fi

        # ------------------------------------------------------------------ #
        # Stage B — VGGT geometry estimation
        # ------------------------------------------------------------------ #
        echo "[B] Running VGGT …"
        if python "${SCRIPT_DIR}/vggt_inference.py" \
                --dataset   "${DATASET}"  \
                --device    "${DEVICE}"   \
                ${USE_BA}                 \
                ${SAVE_GLB}               \
                ${MAX_FRAMES:+--max_frames "${MAX_FRAMES}"} \
                ${CONF_THRES:+--conf_thres_value "${CONF_THRES}"}; then
            echo "[B] Done. Scene in: ${SCRIPT_DIR}/data/scenes/${DATASET}/"
        else
            echo "[B] FAILED for ${DATASET}" >&2
            (( FAIL++ )) || true
            continue
        fi

        # ------------------------------------------------------------------ #
        # Stage C — GaussianPro optimisation
        # ------------------------------------------------------------------ #
        echo "[C] Running GaussianPro …"

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

        if python "${SCRIPT_DIR}/gaussianpro_train.py" \
                --dataset       "${DATASET}"        \
                --iter          "${GP_ITER}"        \
                --lambda_lpips  "${GP_LAMBDA_LPIPS}" \
                ${USE_DEPTH_PRIOR:+--use_depth_prior} \
                --device        "${DEVICE}"         \
                "${SAVE_ITER_ARGS[@]}"; then
            echo "[C] Done."
        else
            echo "[C] FAILED for ${DATASET}" >&2
            (( FAIL++ )) || true
            continue
        fi

        # ------------------------------------------------------------------ #
        # Stage D — render (optional, skipped by default in batch mode)
        # ------------------------------------------------------------------ #
        if [[ -z "${SKIP_RENDER}" ]]; then
            echo "[D] Rendering …"

            SUFFIX=""
            [[ -n "${USE_DEPTH_PRIOR}" ]] && SUFFIX="_depth_prior"
            MODEL_PATH="${SCRIPT_DIR}/data/scenes/${DATASET}/output_${GP_ITER}_gp${SUFFIX}"

            RENDER_EVAL_ARGS=()
            [[ -z "${SKIP_EVAL}" ]] && RENDER_EVAL_ARGS+=(--eval)

            if python "${SCRIPT_DIR}/gaussianpro_render.py" \
                    --model_path  "${MODEL_PATH}"                          \
                    --source_path "${SCRIPT_DIR}/data/scenes/${DATASET}"   \
                    --device      "${DEVICE}"                              \
                    "${RENDER_EVAL_ARGS[@]}"; then
                echo "[D] Done. Renders in: ${MODEL_PATH}/train/"
            else
                echo "[D] FAILED for ${DATASET}" >&2
                (( FAIL++ )) || true
                continue
            fi
        fi

        (( SUCCESS++ )) || true
        echo "  ✓ ${DATASET} complete."
    done
done

echo ""
echo "============================================================"
echo "  Batch complete.  success=${SUCCESS}  failed=${FAIL}"
echo "============================================================"
