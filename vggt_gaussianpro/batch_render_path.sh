#!/usr/bin/env bash
# =============================================================================
# DimensionX — Batch ellipse-orbit render (gaussianpro_render_path.py)
#
# Runs gaussianpro_render_path.py for every trained scene under
# SCENES_DIR, producing a 120-frame orbit video that mirrors the
# InstantSplat render_path() output.
#
# Expected layout:
#   SCENES_DIR/
#     {dataset}/
#       output_{GP_ITER}_gp{_depth_prior}/
#         point_cloud/
#
# Output per scene (written inside the model directory):
#   render/ours_<iter>/renders/               PNG frames
#   render/ours_<iter>/interpolation_renders.mp4
#   render/ours_<iter>/train_renders/         re-rendered train views
#   render/ours_<iter>/train_renders.mp4
#
# Usage:
#   bash batch_render_path.sh
#   SCENES_DIR=/path/to/scenes N_FRAMES=60 bash batch_render_path.sh
#   SKIP_EXISTING="" bash batch_render_path.sh   # re-render everything
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --------------------------------------------------------------------------- #
# Configurable variables (override via environment)
# --------------------------------------------------------------------------- #
SCENES_DIR="${SCENES_DIR:-${SCRIPT_DIR}/data/scenes}"
GP_ITER="${GP_ITER:-30000}"
USE_DEPTH_PRIOR="${USE_DEPTH_PRIOR:-1}"   # set to "" if models lack _depth_prior suffix
N_FRAMES="${N_FRAMES:-120}"
RESIZE="${RESIZE:-crop}"                  # crop | pad | original
FPS="${FPS:-30}"
DEVICE="${DEVICE:-cuda:0}"
WHITE_BG="${WHITE_BG:-}"                  # set to "1" for white background
# Skip scenes whose interpolation_renders.mp4 already exists (default: yes).
SKIP_EXISTING="${SKIP_EXISTING:-1}"

# --------------------------------------------------------------------------- #
# Derived model-directory suffix
# --------------------------------------------------------------------------- #
MODEL_SUFFIX="output_${GP_ITER}_gp"
[[ -n "${USE_DEPTH_PRIOR}" ]] && MODEL_SUFFIX="${MODEL_SUFFIX}_depth_prior"

# --------------------------------------------------------------------------- #
# Banner
# --------------------------------------------------------------------------- #
echo "============================================================"
echo "  DimensionX — Batch GaussianPro Render Path"
echo "  scenes_dir    : ${SCENES_DIR}"
echo "  model suffix  : ${MODEL_SUFFIX}"
echo "  n_frames      : ${N_FRAMES}"
echo "  resize        : ${RESIZE}"
echo "  fps           : ${FPS}"
echo "  device        : ${DEVICE}"
echo "  white_bg      : ${WHITE_BG:-no}"
echo "  skip existing : ${SKIP_EXISTING:-no}"
echo "============================================================"
echo ""

SUCCESS=0
FAIL=0
SKIPPED=0
TOTAL=0

# --------------------------------------------------------------------------- #
# Iterate over every scene directory
# --------------------------------------------------------------------------- #
for SCENE_DIR in "${SCENES_DIR}"/*/; do
    [[ -d "${SCENE_DIR}" ]] || continue

    DATASET="$(basename "${SCENE_DIR}")"
    MODEL_PATH="${SCENE_DIR}${MODEL_SUFFIX}"
    SOURCE_PATH="${SCENE_DIR}"

    # Skip scenes that have no trained model yet.
    if [[ ! -d "${MODEL_PATH}/point_cloud" ]]; then
        echo "  [skip] ${DATASET}  — no checkpoint at ${MODEL_PATH}"
        (( SKIPPED++ )) || true
        continue
    fi

    (( TOTAL++ )) || true

    # Determine which iteration was saved so we can check for existing output.
    # searchForMaxIteration looks for the highest iteration_* folder.
    LATEST_ITER=$(ls "${MODEL_PATH}/point_cloud/" 2>/dev/null \
        | grep -E '^iteration_[0-9]+$' \
        | sed 's/iteration_//' \
        | sort -n \
        | tail -1)

    RENDER_VIDEO="${MODEL_PATH}/render/ours_${LATEST_ITER}/interpolation_renders.mp4"

    if [[ -n "${SKIP_EXISTING}" && -f "${RENDER_VIDEO}" ]]; then
        echo "  [skip] ${DATASET}  — render already exists (${RENDER_VIDEO})"
        (( SKIPPED++ )) || true
        (( TOTAL-- ))   || true
        continue
    fi

    echo "------------------------------------------------------------"
    echo "  Rendering : ${DATASET}"
    echo "  model     : ${MODEL_PATH}"
    echo "------------------------------------------------------------"

    EXTRA_ARGS=()
    [[ -n "${WHITE_BG}" ]] && EXTRA_ARGS+=(--white_background)

    if python "${SCRIPT_DIR}/gaussianpro_render_path.py" \
            --model_path  "${MODEL_PATH}"  \
            --source_path "${SOURCE_PATH}" \
            --n_frames    "${N_FRAMES}"    \
            --resize      "${RESIZE}"      \
            --fps         "${FPS}"         \
            --device      "${DEVICE}"      \
            "${EXTRA_ARGS[@]}"; then
        echo "  ✓ ${DATASET} done."
        (( SUCCESS++ )) || true
    else
        echo "  ✗ ${DATASET} FAILED." >&2
        (( FAIL++ )) || true
    fi

    echo ""
done

# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #
echo "============================================================"
echo "  Batch render complete."
echo "  rendered : ${SUCCESS} / ${TOTAL}"
echo "  failed   : ${FAIL}"
echo "  skipped  : ${SKIPPED}"
echo "============================================================"
