"""
Stage D — GaussianPro Rendering
=================================
Thin wrapper around GaussianPro's render.py.

Renders train and/or test views from a trained GaussianPro scene and writes
them under the model directory.

Usage
-----
    python gaussianpro_render.py \\
        --model_path data/scenes/<dataset>/output_30000_gp_depth_prior/ \\
        --source_path data/scenes/<dataset>/

    # skip training renders, only test views
    python gaussianpro_render.py \\
        --model_path data/scenes/<dataset>/output_30000_gp/ \\
        --source_path data/scenes/<dataset>/ \\
        --skip_train

    # render a specific checkpoint iteration (defaults to latest)
    python gaussianpro_render.py \\
        --model_path data/scenes/<dataset>/output_30000_gp/ \\
        --source_path data/scenes/<dataset>/ \\
        --iteration 7000

Output
------
    <model_path>/train/ours_<iter>/renders/   rendered RGB frames
    <model_path>/train/ours_<iter>/gt/        ground-truth frames
    <model_path>/test/ours_<iter>/renders/    (if test set is non-empty)
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_SCRIPT_DIR  = Path(__file__).resolve().parent
_GP_RENDER_PY = _SCRIPT_DIR / "third_party" / "GaussianPro" / "render.py"


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage D: GaussianPro render wrapper for the DimensionX pipeline."
    )
    parser.add_argument(
        "--model_path", type=str, required=True,
        help="Path to a trained GaussianPro model directory "
             "(e.g. data/scenes/<dataset>/output_30000_gp_depth_prior/).",
    )
    parser.add_argument(
        "--source_path", type=str, required=True,
        help="Path to the COLMAP scene directory "
             "(e.g. data/scenes/<dataset>/).",
    )
    parser.add_argument(
        "--iteration", type=int, default=-1,
        help="Checkpoint iteration to load.  -1 (default) loads the latest "
             "saved checkpoint.",
    )
    parser.add_argument(
        "--skip_train", action="store_true", default=False,
        help="Skip rendering training views.",
    )
    parser.add_argument(
        "--skip_test", action="store_true", default=False,
        help="Skip rendering test views.",
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0",
        help="Torch device string (default: cuda:0).  Sets CUDA_VISIBLE_DEVICES.",
    )
    parser.add_argument(
        "--eval", action="store_true", default=False,
        help="Hold out every 8th frame as a test set when loading the scene, "
             "producing renders in test/ours_<iter>/.  Only meaningful if the "
             "model was also trained with --eval (same split); otherwise the "
             "Gaussians were trained on all frames.",
    )
    parser.add_argument(
        "--quiet", action="store_true", default=False,
        help="Suppress GaussianPro render.py output.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    model_path  = Path(args.model_path).resolve()
    source_path = Path(args.source_path).resolve()

    if not model_path.is_dir():
        sys.exit(
            f"[gaussianpro_render] ERROR: model_path not found: {model_path}\n"
            "Run gaussianpro_train.py first."
        )

    if not source_path.is_dir():
        sys.exit(
            f"[gaussianpro_render] ERROR: source_path not found: {source_path}"
        )

    print("=" * 60)
    print("  DimensionX — GaussianPro Rendering (Stage D)")
    print(f"  model_path  : {model_path}")
    print(f"  source_path : {source_path}")
    print(f"  iteration   : {'latest' if args.iteration == -1 else args.iteration}")
    print(f"  skip_train  : {args.skip_train}")
    print(f"  skip_test   : {args.skip_test}")
    print("=" * 60)

    cmd = [
        sys.executable,
        str(_GP_RENDER_PY),
        "--model_path",  str(model_path),
        "--source_path", str(source_path),
        "--iteration",   str(args.iteration),
    ]

    if args.skip_train:
        cmd.append("--skip_train")
    if args.skip_test:
        cmd.append("--skip_test")
    if args.eval:
        cmd.append("--eval")
    if args.quiet:
        cmd.append("--quiet")

    # Set CUDA_VISIBLE_DEVICES from --device (e.g. "cuda:1" → "1").
    env = os.environ.copy()
    device_idx = args.device.split(":")[-1] if ":" in args.device else "0"
    env.setdefault("CUDA_VISIBLE_DEVICES", device_idx)

    # Run render.py from its own directory so relative imports resolve.
    gp_root = _GP_RENDER_PY.parent
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        str(gp_root) + os.pathsep + existing_pythonpath
        if existing_pythonpath
        else str(gp_root)
    )

    print("\n[gaussianpro_render] Launching GaussianPro render.py …")
    print("  CMD:", " ".join(cmd))
    print()

    result = subprocess.run(cmd, cwd=str(gp_root), env=env)

    if result.returncode != 0:
        sys.exit(
            f"[gaussianpro_render] GaussianPro render.py exited with code {result.returncode}."
        )

    print(f"\n[gaussianpro_render] Done.  Renders written to: {model_path}/train/  and/or  {model_path}/test/")


if __name__ == "__main__":
    main()
