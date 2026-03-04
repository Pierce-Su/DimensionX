#!/usr/bin/env python3
"""
Download SAT 360° pipeline weights from Hugging Face only (no Tsinghua links).

The main model for 145-frame inference is the DimensionX 360° checkpoint (mp_rank_00_model_states.pt),
which the DimensionX authors provide—you may already have it under e.g. DimensionX/checkpoints/.
The only weights that must come from external sources are T5 and 3D VAE (CogVideoX components).

This script downloads:
  - T5 + 3D VAE from zai-org/CogVideoX1.5-5B-SAT (the only external components)
  - Optionally the DimensionX 360° checkpoint from ShuoChen20/DimensionX_360orbit (skip with --t5-vae-only)

Usage (from DimensionX repo root):
  python scripts/setup_sat_360_weights.py [OUTPUT_DIR]
  python scripts/setup_sat_360_weights.py [OUTPUT_DIR] --t5-vae-only   # if you already have checkpoints/
Default OUTPUT_DIR: ./sat_weights
"""

import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(description="Download SAT 360 weights from Hugging Face.")
    parser.add_argument(
        "output_dir",
        nargs="?",
        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sat_weights"),
        help="Directory to download weights into (default: repo_root/sat_weights)",
    )
    parser.add_argument(
        "--t5-vae-only",
        action="store_true",
        help="Only download T5 and VAE (skip DimensionX 360° checkpoint). Use if you already have checkpoints/ from the authors.",
    )
    args = parser.parse_args()
    out = os.path.abspath(args.output_dir)
    os.makedirs(out, exist_ok=True)

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Install huggingface_hub: pip install -U huggingface_hub", file=sys.stderr)
        sys.exit(1)

    print("Downloading T5 and 3D VAE from zai-org/CogVideoX1.5-5B-SAT (only external components)...")
    snapshot_download(
        repo_id="zai-org/CogVideoX1.5-5B-SAT",
        local_dir=out,
        local_dir_use_symlinks=False,
        allow_patterns=["t5-v1_1-xxl/*", "vae/3d-vae.pt"],
    )

    if not args.t5_vae_only:
        print("Downloading DimensionX 360° checkpoint from ShuoChen20/DimensionX_360orbit...")
        os.makedirs(os.path.join(out, "checkpoints", "1"), exist_ok=True)
        snapshot_download(
            repo_id="ShuoChen20/DimensionX_360orbit",
            local_dir=os.path.join(out, "checkpoints"),
            local_dir_use_symlinks=False,
            allow_patterns=["mp_rank_00_model_states.pt", "latest"],
        )
        import shutil
        pt_src = os.path.join(out, "checkpoints", "mp_rank_00_model_states.pt")
        if os.path.isfile(pt_src):
            shutil.move(pt_src, os.path.join(out, "checkpoints", "1", "mp_rank_00_model_states.pt"))
        print("\nDone. Use with run_batch_pipeline.py (--video_backend sat_360):")
        print(f"  --sat_t5_dir         {os.path.join(out, 't5-v1_1-xxl')}")
        print(f"  --sat_vae_ckpt       {os.path.join(out, 'vae', '3d-vae.pt')}")
        print(f"  --sat_checkpoint_dir {os.path.join(out, 'checkpoints')}")
    else:
        print("\nDone (T5 + VAE only). You already have the main model (DimensionX 360° checkpoint).")
        print("Use with run_batch_pipeline.py (--video_backend sat_360):")
        print(f"  --sat_t5_dir         {os.path.join(out, 't5-v1_1-xxl')}")
        print(f"  --sat_vae_ckpt       {os.path.join(out, 'vae', '3d-vae.pt')}")
        print(f"  --sat_checkpoint_dir <path/to/your/checkpoints>   # e.g. DimensionX/checkpoints")


if __name__ == "__main__":
    main()
