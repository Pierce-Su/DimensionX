#!/usr/bin/env bash
# Setup SAT 360-degree pipeline weights using Hugging Face only (no Tsinghua links).
#
# The main model for 145-frame inference is the DimensionX 360° checkpoint (from the authors);
# you may already have it under e.g. DimensionX/checkpoints/. The only external components
# are T5 and 3D VAE (CogVideoX). This script downloads:
#   - T5 + 3D VAE from zai-org/CogVideoX1.5-5B-SAT
#   - Optionally the DimensionX 360° checkpoint (skip with second arg: --t5-vae-only)
#
# Usage (run from DimensionX repo root):
#   bash scripts/setup_sat_360_weights.sh [OUTPUT_DIR]
#   bash scripts/setup_sat_360_weights.sh [OUTPUT_DIR] --t5-vae-only   # if you already have checkpoints/
# Default OUTPUT_DIR is ./sat_weights.

set -e
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SKIP_360=false
if [[ "${2:-}" == "--t5-vae-only" ]] || [[ "${1:-}" == "--t5-vae-only" ]]; then
  SKIP_360=true
  [[ "${1:-}" == "--t5-vae-only" ]] && OUTPUT_DIR="${2:-${REPO_ROOT}/sat_weights}" || OUTPUT_DIR="${1:-${REPO_ROOT}/sat_weights}"
else
  OUTPUT_DIR="${1:-${REPO_ROOT}/sat_weights}"
fi
# Resolve to absolute path
if [[ -d "$OUTPUT_DIR" ]]; then
  OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"
else
  OUTPUT_DIR="$(cd "$(dirname "$OUTPUT_DIR")" 2>/dev/null && pwd)/$(basename "$OUTPUT_DIR")"
fi

echo "SAT 360 weights will be downloaded to: $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"
cd "$OUTPUT_DIR"

# --- 1) T5 and 3D VAE from zai-org (only external components) ---
echo ""
echo ">>> Downloading T5 and 3D VAE from zai-org/CogVideoX1.5-5B-SAT..."
huggingface-cli download zai-org/CogVideoX1.5-5B-SAT \
  --local-dir . \
  --local-dir-use-symlinks False \
  --include "t5-v1_1-xxl/*" "vae/3d-vae.pt"

# --- 2) DimensionX 360° checkpoint (optional; skip if you already have it) ---
if [[ "$SKIP_360" == "false" ]]; then
  echo ""
  echo ">>> Downloading DimensionX 360° checkpoint from ShuoChen20/DimensionX_360orbit..."
  mkdir -p checkpoints/1
  huggingface-cli download ShuoChen20/DimensionX_360orbit mp_rank_00_model_states.pt --local-dir ./checkpoints/1
  huggingface-cli download ShuoChen20/DimensionX_360orbit latest --local-dir ./checkpoints
fi

echo ""
echo "Done. Use with run_batch_pipeline.py (--video_backend sat_360):"
echo "  --sat_t5_dir         $OUTPUT_DIR/t5-v1_1-xxl"
echo "  --sat_vae_ckpt       $OUTPUT_DIR/vae/3d-vae.pt"
if [[ "$SKIP_360" == "false" ]]; then
  echo "  --sat_checkpoint_dir $OUTPUT_DIR/checkpoints"
else
  echo "  --sat_checkpoint_dir <path/to/your/checkpoints>   # e.g. DimensionX/checkpoints"
fi
