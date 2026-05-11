#!/usr/bin/env bash
# =============================================================================
# DimensionX — VGGT + GaussianPro lifting pipeline (batch / multi-scene)
#
# Expected layout under DATAROOT:
#   DATAROOT/
#     {type}/
#       index_{id}/
#         video.mp4
#
# Example:
#   DATAROOT=./data/video TYPES="orbit dolly" bash batch_pipeline.sh
# =============================================================================

set -euo pipefail

DATAROOT="${DATAROOT:-./workspace/DimensionX/data/dimensionx_batch_sat360}"
TYPES="${TYPES:-orbit dolly}"     # space-separated list of scene types
NUM_FRAMES="${NUM_FRAMES:-}"       # leave empty to extract ALL frames
DEVICE="${DEVICE:-cuda:0}"
USE_BA="${USE_BA:-}"               # set to "--use_ba" to enable bundle adjustment

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "============================================================"
echo "  DimensionX VGGT Batch Pipeline"
echo "  dataroot : ${DATAROOT}"
echo "  types    : ${TYPES}"
echo "  frames   : ${NUM_FRAMES:-all}"
echo "  device   : ${DEVICE}"
echo "============================================================"

SUCCESS=0
FAIL=0

for TYPE in ${TYPES}; do
    for VIDEO_PATH in "${DATAROOT}/${TYPE}"/index_*/video.mp4; do
        [[ -f "${VIDEO_PATH}" ]] || continue

        ID="$(basename "$(dirname "${VIDEO_PATH}")")"
        DATASET="${TYPE}_${ID}"

        echo ""
        echo "------------------------------------------------------------"
        echo "  Processing: ${DATASET}"
        echo "  Video     : ${VIDEO_PATH}"
        echo "------------------------------------------------------------"

        IMAGES_DIR="${SCRIPT_DIR}/data/images/${DATASET}"

        # Stage A — frame extraction
        echo "[A] Extracting frames …"
        python "${SCRIPT_DIR}/get_frame.py" \
            "${VIDEO_PATH}" \
            "${IMAGES_DIR}" \
            ${NUM_FRAMES} \
        && echo "[A] Done." \
        || { echo "[A] FAILED for ${DATASET}"; (( FAIL++ )); continue; }

        # Stage B — VGGT geometry
        echo "[B] Running VGGT …"
        python "${SCRIPT_DIR}/vggt_inference.py" \
            --dataset "${DATASET}" \
            --device  "${DEVICE}"  \
            ${USE_BA} \
        && echo "[B] Done." \
        || { echo "[B] FAILED for ${DATASET}"; (( FAIL++ )); continue; }

        # Stage C — GaussianPro (TODO)
        # python "${SCRIPT_DIR}/gaussianpro_train.py" \
        #     --dataset "${DATASET}" \
        #     --iter 30000 \
        #     --lambda_lpips 0.3 \
        #     --use_depth_prior

        (( SUCCESS++ ))
        echo "  ✓ ${DATASET} complete."
    done
done

echo ""
echo "============================================================"
echo "  Batch complete.  success=${SUCCESS}  failed=${FAIL}"
echo "============================================================"
