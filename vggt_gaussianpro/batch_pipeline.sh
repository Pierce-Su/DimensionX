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
#   SKIP_RENDER="" bash batch_pipeline.sh      # also run Stage D renders
#   SKIP_RENDER_PATH=1 bash batch_pipeline.sh  # skip Stage E novel-view orbit
# =============================================================================

set -euo pipefail

# --------------------------------------------------------------------------- #
# Configurable variables (override via environment)
# --------------------------------------------------------------------------- #
DATAROOT="${DATAROOT:-${1:-/workspace/DimensionX/data/dimensionx_batch_sat360}}"
TYPES="${TYPES:-photorealistic stylized}"   # space-separated style categories
NUM_FRAMES="${NUM_FRAMES:-}"                # leave empty to extract ALL frames
DEVICE="${DEVICE:-cuda:0}"
USE_BA="${USE_BA:-}"               # set to "--use_ba" to enable bundle adjustment
SAVE_GLB="${SAVE_GLB:-}"           # set to "--save_glb" to write scene.glb
# Maximum frames fed to VGGT (prevents OOM on large frame counts).
# Empirically on RTX 6000 Ada (47.4 GB usable): N=80→32 GB, N=100→41 GB,
# N=120→47 GB (tight), N=145→OOM. Safe defaults: 110 for 48 GB | 60 for 24 GB.
MAX_FRAMES="${MAX_FRAMES:-110}"
# Absolute confidence threshold for depth filtering (no-BA path).
# 3.0 keeps more points from orbit/OOD content; percentile fallback handles full OOD.
CONF_THRES="${CONF_THRES:-3.0}"

# --- GaussianPro (Stage C/D) ---
GP_ITER="${GP_ITER:-30000}"
GP_LAMBDA_LPIPS="${GP_LAMBDA_LPIPS:-0.3}"
USE_DEPTH_PRIOR="${USE_DEPTH_PRIOR:-1}"     # set to "" to disable depth prior
# Save only the final iteration's point cloud and checkpoint (default: yes).
# Set to "" to restore the full default schedule {1, 7000, 20000, 30000}.
GP_SAVE_ONLY_FINAL="${GP_SAVE_ONLY_FINAL:-1}"
SKIP_RENDER="${SKIP_RENDER:-1}"             # set to "" to enable Stage D renders
SKIP_EVAL="${SKIP_EVAL:-}"                  # set to "1" to omit --eval in render

# --- Novel-view orbit render (Stage E) ---
SKIP_RENDER_PATH="${SKIP_RENDER_PATH:-}"    # set to "1" to skip Stage E
N_FRAMES="${N_FRAMES:-120}"                 # number of frames in the orbit video
Z_AMPLITUDE_FRAC="${Z_AMPLITUDE_FRAC:-0.35}"  # elevation amplitude (fraction of orbit radius)
RESIZE="${RESIZE:-crop}"                    # crop | pad | original
FPS="${FPS:-30}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

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
echo "  render path     : ${SKIP_RENDER_PATH:+skipped}"
echo "  n_frames        : ${N_FRAMES}"
echo "  z_amp_frac      : ${Z_AMPLITUDE_FRAC}"
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

        # Skip only if the final GaussianPro checkpoint exists (not a partial run).
        _SKIP_SUFFIX=""
        [[ -n "${USE_DEPTH_PRIOR}" ]] && _SKIP_SUFFIX="_depth_prior"
        _EXPECTED_MODEL="${SCRIPT_DIR}/data/scenes/${DATASET}/output_${GP_ITER}_gp${_SKIP_SUFFIX}"
        _FINAL_PLY="${_EXPECTED_MODEL}/point_cloud/iteration_${GP_ITER}/point_cloud.ply"
        _FINAL_CHKPT="${_EXPECTED_MODEL}/chkpnt${GP_ITER}.pth"
        if [[ -f "${_FINAL_PLY}" ]] || [[ -f "${_FINAL_CHKPT}" ]]; then
            echo "  ↷ ${DATASET} already complete (found final checkpoint), skipping."
            (( SUCCESS++ )) || true
            continue
        fi
        if [[ -d "${_EXPECTED_MODEL}" ]]; then
            echo "  ↻ ${DATASET} has partial Stage C output — resuming training."
        fi

        IMAGES_DIR="${SCRIPT_DIR}/data/images/${DATASET}"
        SCENE_DIR="${SCRIPT_DIR}/data/scenes/${DATASET}"

        # ------------------------------------------------------------------ #
        # Stage A — frame extraction
        # ------------------------------------------------------------------ #
        _stage_a_frames=$(ls "${IMAGES_DIR}" 2>/dev/null | wc -l)
        if [[ "${_stage_a_frames}" -gt 0 ]]; then
            echo "[Stage A] Already done (${_stage_a_frames} frames found), skipping."
        else
            echo "[Stage A] Extracting frames …"
            if python "${SCRIPT_DIR}/get_frame.py" \
                    "${VIDEO_PATH}" \
                    "${IMAGES_DIR}" \
                    ${NUM_FRAMES}; then
                echo "[Stage A] Done. Frames in: ${IMAGES_DIR}"
            else
                echo "[Stage A] FAILED for ${DATASET}" >&2
                (( FAIL++ )) || true
                continue
            fi
        fi

        # ------------------------------------------------------------------ #
        # Stage B — VGGT geometry estimation
        # ------------------------------------------------------------------ #
        # Complete Stage B requires COLMAP binaries in sparse/ (VGGT writes
        # directly to sparse/, not sparse/0/) plus non-empty confidence_maps/.
        _stage_b_conf=$(ls "${SCENE_DIR}/confidence_maps/" 2>/dev/null | wc -l)
        if [[ -f "${SCENE_DIR}/sparse/cameras.bin"  ]] && \
           [[ -f "${SCENE_DIR}/sparse/images.bin"   ]] && \
           [[ -f "${SCENE_DIR}/sparse/points3D.bin" ]] && \
           [[ "${_stage_b_conf}" -gt 0 ]]; then
            echo "[Stage B] Already done (${_stage_b_conf} frames, COLMAP sparse present), skipping."
        else
            if [[ "${_stage_b_conf}" -gt 0 ]] || [[ -d "${SCENE_DIR}/sparse" ]]; then
                echo "[Stage B] Partial / corrupt Stage B output detected — re-running."
            else
                echo "[Stage B] Running VGGT …"
            fi
            if python "${SCRIPT_DIR}/vggt_inference.py" \
                    --dataset   "${DATASET}"  \
                    --device    "${DEVICE}"   \
                    ${USE_BA}                 \
                    ${SAVE_GLB}               \
                    ${MAX_FRAMES:+--max_frames "${MAX_FRAMES}"} \
                    ${CONF_THRES:+--conf_thres_value "${CONF_THRES}"}; then
                echo "[Stage B] Done. Scene in: ${SCENE_DIR}/"
            else
                echo "[Stage B] FAILED for ${DATASET}" >&2
                (( FAIL++ )) || true
                continue
            fi
        fi

        # ------------------------------------------------------------------ #
        # Stage C — GaussianPro optimisation
        # ------------------------------------------------------------------ #
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

        if python "${SCRIPT_DIR}/gaussianpro_train.py" \
                --dataset       "${DATASET}"        \
                --iter          "${GP_ITER}"        \
                --lambda_lpips  "${GP_LAMBDA_LPIPS}" \
                ${USE_DEPTH_PRIOR:+--use_depth_prior} \
                --device        "${DEVICE}"         \
                "${SAVE_ITER_ARGS[@]}"; then
            echo "[Stage C] Done."
        else
            echo "[Stage C] FAILED for ${DATASET}" >&2
            (( FAIL++ )) || true
            continue
        fi

        # Derive model path once — shared by Stage D and Stage E.
        SUFFIX=""
        [[ -n "${USE_DEPTH_PRIOR}" ]] && SUFFIX="_depth_prior"
        MODEL_PATH="${SCRIPT_DIR}/data/scenes/${DATASET}/output_${GP_ITER}_gp${SUFFIX}"

        # ------------------------------------------------------------------ #
        # Stage D — render trained views (optional, skipped by default)
        # ------------------------------------------------------------------ #
        if [[ -z "${SKIP_RENDER}" ]]; then
            echo "[Stage D] Rendering …"

            RENDER_EVAL_ARGS=()
            [[ -z "${SKIP_EVAL}" ]] && RENDER_EVAL_ARGS+=(--eval)

            if python "${SCRIPT_DIR}/gaussianpro_render.py" \
                    --model_path  "${MODEL_PATH}"                          \
                    --source_path "${SCRIPT_DIR}/data/scenes/${DATASET}"   \
                    --device      "${DEVICE}"                              \
                    "${RENDER_EVAL_ARGS[@]}"; then
                echo "[Stage D] Done. Renders in: ${MODEL_PATH}/train/"
            else
                echo "[Stage D] FAILED for ${DATASET}" >&2
                (( FAIL++ )) || true
                continue
            fi
        fi

        # ------------------------------------------------------------------ #
        # Stage E — novel-view sinusoidal-elevation orbit render
        # ------------------------------------------------------------------ #
        if [[ -z "${SKIP_RENDER_PATH}" ]]; then
            echo "[Stage E] Rendering novel-view orbit …"

            if python "${SCRIPT_DIR}/gaussianpro_render_path.py" \
                    --model_path         "${MODEL_PATH}"                        \
                    --source_path        "${SCRIPT_DIR}/data/scenes/${DATASET}" \
                    --n_frames           "${N_FRAMES}"                          \
                    --z_amplitude_frac   "${Z_AMPLITUDE_FRAC}"                  \
                    --resize             "${RESIZE}"                            \
                    --fps                "${FPS}"                               \
                    --device             "${DEVICE}"; then
                echo "[Stage E] Done. Orbit video in: ${MODEL_PATH}/render/"
            else
                echo "[Stage E] FAILED for ${DATASET}" >&2
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
