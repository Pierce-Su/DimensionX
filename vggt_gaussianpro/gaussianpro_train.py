"""
Stage C — GaussianPro Optimization
====================================
Thin wrapper around GaussianPro's train.py that:

  1. Resolves source/model paths from a dataset name.
  2. Creates a ``metricdepth/`` symlink → ``depth_maps/`` inside the scene
     directory so GaussianPro's depth-prior loader can find VGGT depth maps
     using its expected naming convention.
  3. Invokes GaussianPro's ``train.py`` via subprocess with the right flags.

Usage
-----
    python gaussianpro_train.py --dataset <name> [options]

    # minimal
    python gaussianpro_train.py --dataset index_0003

    # with VGGT depth prior
    python gaussianpro_train.py \\
        --dataset index_0003 \\
        --iter 30000 \\
        --use_depth_prior

Data contract
-------------
Reads:
    data/scenes/<dataset>/images/          PNG frames
    data/scenes/<dataset>/sparse/          cameras.bin  images.bin  points3D.bin
    data/scenes/<dataset>/depth_maps/      {i}.npy  float32 metric depth   (when --use_depth_prior)
    data/scenes/<dataset>/normals/         {i}.npy  float32 [0,1]-mapped normals (when --use_depth_prior)

Writes:
    data/scenes/<dataset>/output_{iter}_gp[_depth_prior]/
        point_cloud/iteration_*/point_cloud.ply
        cameras.json
        cfg_args
        tb_logs/
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_SCRIPT_DIR   = Path(__file__).resolve().parent
_GP_TRAIN_PY  = _SCRIPT_DIR / "third_party" / "GaussianPro" / "train.py"


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage C: GaussianPro optimization wrapper for the DimensionX pipeline."
    )
    parser.add_argument(
        "--dataset", type=str, required=True,
        help="Dataset name.  Reads from data/scenes/<dataset>/, "
             "writes to data/scenes/<dataset>/output_{iter}_gp[_depth_prior]/.",
    )
    parser.add_argument(
        "--scenes_dir", type=str, default=None,
        help="Override scenes root directory (default: data/scenes/).",
    )
    parser.add_argument(
        "--iter", type=int, default=30_000,
        help="Total training iterations (default: 30000).",
    )
    parser.add_argument(
        "--lambda_lpips", type=float, default=0.3,
        help="Perceptual loss weight.  Accepted for CLI parity with the existing "
             "pipeline; GaussianPro uses lambda_dssim (0.2) natively so this "
             "value is not forwarded to the subprocess.",
    )
    # --- depth prior ---
    parser.add_argument(
        "--use_depth_prior", action="store_true", default=False,
        help="Inject VGGT depth maps and normals into GaussianPro propagation "
             "(activates --load_depth --load_normal --depth_loss --normal_loss).",
    )
    parser.add_argument(
        "--confidence_threshold", type=float, default=0.3,
        help="Minimum normalised depth confidence for a pixel to participate in "
             "normal/depth supervision (informational; not forwarded to GaussianPro "
             "as a native flag in the current upstream release).",
    )
    # --- propagation schedule ---
    parser.add_argument(
        "--propagation_interval", type=int, default=20,
        help="Iterations between successive propagation steps (default: 20, "
             "matches GaussianPro upstream default).",
    )
    parser.add_argument(
        "--propagation_start", type=int, default=1000,
        help="Warm-up iterations before the first propagation step "
             "(maps to --propagated_iteration_begin, default: 1000).",
    )
    parser.add_argument(
        "--propagation_end", type=int, default=12_000,
        help="Iteration at which propagation stops "
             "(maps to --propagated_iteration_after, default: 12000).",
    )
    parser.add_argument(
        "--patch_size", type=int, default=20,
        help="Patch size for the ACMH-style patch-matching step (default: 20).",
    )
    # --- eval split ---
    parser.add_argument(
        "--eval", action="store_true", default=False,
        help="Hold out every 8th frame as a test set (LLFF-style).  "
             "When set, test/ours_<iter>/ will contain rendered held-out views "
             "for PSNR/SSIM evaluation.  When unset (default) all frames are used "
             "for training and test/ renders will be empty.",
    )
    # --- misc ---
    parser.add_argument(
        "--export_ply", action="store_true", default=False,
        help="(informational) GaussianPro always saves point_cloud.ply at "
             "save_iterations; this flag is accepted for CLI parity.",
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0",
        help="Torch device string (default: cuda:0).  Sets CUDA_VISIBLE_DEVICES "
             "to the device index.",
    )
    parser.add_argument(
        "--port", type=int, default=6099,
        help="Network GUI port for GaussianPro's viewer (default: 6099; "
             "use a free port to avoid conflicts).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Scene preparation helpers
# ---------------------------------------------------------------------------

def _validate_sparse(scene_dir: Path) -> None:
    """
    Sanity-check the COLMAP sparse output before launching GaussianPro.
    Raises SystemExit with a clear message if the data looks corrupt so the
    user doesn't have to read a deep traceback.
    """
    images_bin = scene_dir / "sparse" / "images.bin"
    if not images_bin.exists():
        sys.exit(
            f"[gaussianpro_train] ERROR: {images_bin} not found.\n"
            "Run vggt_inference.py first to generate the COLMAP scene."
        )

    # images.bin header is a uint64 count; 8 bytes ⇒ 0 images.
    if images_bin.stat().st_size <= 8:
        sys.exit(
            f"[gaussianpro_train] ERROR: {images_bin} exists but contains 0 images "
            "(file is only {images_bin.stat().st_size} bytes).\n\n"
            "This is caused by a pycolmap 4.x compatibility bug that has now been "
            "patched in vggt/dependency/np_to_pycolmap.py.  Please re-run "
            "vggt_inference.py for this dataset to regenerate the COLMAP files:\n\n"
            f"    python vggt_inference.py --dataset {scene_dir.name}\n"
        )


def ensure_sparse_subdir(scene_dir: Path) -> None:
    """
    GaussianPro's readColmapSceneInfo always looks for COLMAP files under
    ``<scene>/sparse/0/``, but VGGT writes them directly to ``<scene>/sparse/``.

    Create ``sparse/0/`` as a directory containing symlinks to the ``.bin``
    (and ``.ply``) files in ``sparse/`` so GaussianPro can find them.
    """
    sparse_dir  = scene_dir / "sparse"
    subdir      = sparse_dir / "0"

    if subdir.exists() and not subdir.is_symlink():
        # Already a real directory — check that the required files are present.
        if (subdir / "cameras.bin").exists():
            print("[gaussianpro_train] sparse/0/ already populated, skipping.")
            return

    subdir.mkdir(parents=True, exist_ok=True)

    for src in sparse_dir.iterdir():
        if src.name == "0":
            continue
        dst = subdir / src.name
        if dst.exists() or dst.is_symlink():
            continue
        dst.symlink_to(src.resolve())

    print(f"[gaussianpro_train] Created sparse/0/ with symlinks → {sparse_dir}")


def ensure_metricdepth_link(scene_dir: Path) -> None:
    """
    GaussianPro reads metric depth from ``<scene>/metricdepth/{stem}.npy``.
    VGGT writes depth maps to ``<scene>/depth_maps/{stem}.npy``.
    Create a symlink so GaussianPro can find the files under its expected name.
    """
    depth_maps_dir  = scene_dir / "depth_maps"
    metricdepth_dir = scene_dir / "metricdepth"

    if not depth_maps_dir.is_dir():
        raise FileNotFoundError(
            f"depth_maps/ not found at {depth_maps_dir}. "
            "Run vggt_inference.py first, or omit --use_depth_prior."
        )

    if metricdepth_dir.is_symlink():
        # Already a symlink — verify it points to depth_maps/.
        if metricdepth_dir.resolve() != depth_maps_dir.resolve():
            metricdepth_dir.unlink()
            metricdepth_dir.symlink_to(depth_maps_dir.resolve())
            print(f"[gaussianpro_train] Updated symlink: metricdepth/ → {depth_maps_dir}")
        else:
            print("[gaussianpro_train] metricdepth/ symlink already correct.")
    elif metricdepth_dir.is_dir():
        print(
            f"[gaussianpro_train] WARNING: {metricdepth_dir} is a real directory "
            "(not a symlink).  It will be used as-is."
        )
    else:
        metricdepth_dir.symlink_to(depth_maps_dir.resolve())
        print(f"[gaussianpro_train] Created symlink: metricdepth/ → {depth_maps_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------ paths
    scenes_root = Path(args.scenes_dir) if args.scenes_dir else _SCRIPT_DIR / "data" / "scenes"
    scene_dir   = scenes_root / args.dataset

    if not scene_dir.is_dir():
        sys.exit(
            f"[gaussianpro_train] ERROR: scene directory not found: {scene_dir}\n"
            "Run vggt_inference.py first to generate the COLMAP scene."
        )

    suffix     = "_depth_prior" if args.use_depth_prior else ""
    model_path = scene_dir / f"output_{args.iter}_gp{suffix}"

    print("=" * 60)
    print("  DimensionX — GaussianPro Training (Stage C)")
    print(f"  dataset     : {args.dataset}")
    print(f"  source_path : {scene_dir}")
    print(f"  model_path  : {model_path}")
    print(f"  iterations  : {args.iter}")
    print(f"  depth prior : {'enabled' if args.use_depth_prior else 'disabled'}")
    print("=" * 60)

    # ---- validate COLMAP sparse output before doing anything expensive
    _validate_sparse(scene_dir)

    # ---- ensure sparse/0/ exists (GaussianPro expects files there, VGGT writes to sparse/)
    ensure_sparse_subdir(scene_dir)

    # ---------------------------------------- prepare metricdepth/ symlink
    if args.use_depth_prior:
        ensure_metricdepth_link(scene_dir)

    # ----------------------------------------------------- build subprocess command
    save_iters = sorted({1, 7_000, min(20_000, args.iter), args.iter})
    test_iters = sorted({1, 2_000, 7_000, min(20_000, args.iter), args.iter})

    cmd = [
        sys.executable,
        str(_GP_TRAIN_PY),
        "--source_path",   str(scene_dir),
        "--model_path",    str(model_path),
        "--iterations",    str(args.iter),
        "--propagation_interval",        str(args.propagation_interval),
        "--propagated_iteration_begin",  str(args.propagation_start),
        "--propagated_iteration_after",  str(args.propagation_end),
        "--patch_size",    str(args.patch_size),
        "--save_iterations", *[str(it) for it in save_iters],
        "--test_iterations", *[str(it) for it in test_iters],
        "--port",          str(args.port),
    ]

    if args.use_depth_prior:
        cmd += [
            "--load_depth",
            "--load_normal",
            "--depth_loss",
            "--normal_loss",
        ]

    if args.eval:
        cmd.append("--eval")

    # Set CUDA_VISIBLE_DEVICES from --device (e.g. "cuda:0" → "0")
    env = os.environ.copy()
    device_idx = args.device.split(":")[-1] if ":" in args.device else "0"
    env.setdefault("CUDA_VISIBLE_DEVICES", device_idx)

    # GaussianPro's train.py must run with its own directory on sys.path so
    # relative imports (scene, gaussian_renderer, …) resolve correctly.
    gp_root = _GP_TRAIN_PY.parent
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        str(gp_root) + os.pathsep + existing_pythonpath
        if existing_pythonpath
        else str(gp_root)
    )

    # GaussianPro's propagation step (≥ propagation_start iterations) saves debug
    # images to a hardcoded relative path "cost/".  Create it up front so the
    # training loop doesn't crash with FileNotFoundError.
    (gp_root / "cost").mkdir(exist_ok=True)

    print("\n[gaussianpro_train] Launching GaussianPro train.py …")
    print("  CMD:", " ".join(cmd))
    print()

    result = subprocess.run(cmd, cwd=str(gp_root), env=env)

    if result.returncode != 0:
        sys.exit(
            f"[gaussianpro_train] GaussianPro train.py exited with code {result.returncode}."
        )

    print(f"\n[gaussianpro_train] Done.  Outputs at: {model_path}")


if __name__ == "__main__":
    main()
