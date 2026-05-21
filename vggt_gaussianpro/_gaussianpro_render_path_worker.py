"""
_gaussianpro_render_path_worker.py — GaussianPro render worker for ellipse-orbit renders
=========================================================================================
Invoked as a subprocess by ``gaussianpro_render_path.py`` with:
  - ``cwd`` set to the GaussianPro root (so all relative imports resolve)
  - ``PYTHONPATH`` prepended with the GaussianPro root

DO NOT run this script directly.  Use ``gaussianpro_render_path.py`` instead.

The worker receives pre-computed camera parameters (orbit + train views) via a
JSON file written by the orchestrator, builds GaussianPro Camera objects, loads
the Gaussian model from the supplied PLY path, renders every camera, saves PNG
frames, and stitches MP4 videos with OpenCV.
"""

import os
import sys

# ---------------------------------------------------------------------------
# Path fix — must happen before ANY other import.
#
# Root cause: vggt_gaussianpro/utils/__init__.py is a *regular* package.
# GaussianPro's utils/ has NO __init__.py, making it a *namespace* package.
# Python always prefers a regular package over a namespace package regardless
# of sys.path ordering — so vggt_gaussianpro/utils shadows GP's utils even
# when _GP_ROOT is sys.path[0], causing ModuleNotFoundError: utils.system_utils.
#
# Fix: remove vggt_gaussianpro/ (this script's parent directory) from sys.path
# entirely before any import, so Python only sees GaussianPro's namespace
# utils/ and not the pipeline's regular utils/ package.
# ---------------------------------------------------------------------------
_WORKER_DIR = os.path.dirname(os.path.abspath(__file__))
_GP_ROOT    = os.path.join(_WORKER_DIR, "third_party", "GaussianPro")

# Strip any reference to the worker's own directory from sys.path so the
# pipeline's utils/ regular package does not shadow GaussianPro's utils/.
sys.path = [p for p in sys.path if os.path.abspath(p) != _WORKER_DIR]

# Ensure GP root is present (it should already be via PYTHONPATH, but be explicit).
if _GP_ROOT not in sys.path:
    sys.path.insert(0, _GP_ROOT)

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision
from tqdm import tqdm

# ---------------------------------------------------------------------------
# GaussianPro imports — these only work when cwd / PYTHONPATH is GP root.
# ---------------------------------------------------------------------------
from gaussian_renderer import GaussianModel, render          # noqa: E402
from scene.cameras import Camera                              # noqa: E402
from utils.graphics_utils import (                            # noqa: E402
    focal2fov,
    fov2focal,
    getProjectionMatrix,
)


# ===========================================================================
# Camera construction
# ===========================================================================

def _build_camera(cam_dict: dict, data_device: str = "cuda") -> Camera:
    """Construct a GaussianPro Camera from a serialised camera dict."""
    R_cam = np.array(cam_dict["R_cam"], dtype=np.float64)
    T_cam = np.array(cam_dict["T_cam"], dtype=np.float64)
    FoVx  = float(cam_dict["FoVx"])
    FoVy  = float(cam_dict["FoVy"])
    fx    = float(cam_dict["fx"])
    fy    = float(cam_dict["fy"])
    width  = int(cam_dict["width"])
    height = int(cam_dict["height"])
    uid    = int(cam_dict["uid"])
    image_name = str(cam_dict["image_name"])

    # K = [fx, fy, cx, cy] as expected by Camera.__init__
    K = np.array([fx, fy, width / 2.0, height / 2.0], dtype=np.float32)

    dummy_image = torch.zeros((3, height, width), dtype=torch.float32)

    return Camera(
        colmap_id=uid,
        R=R_cam,
        T=T_cam,
        FoVx=FoVx,
        FoVy=FoVy,
        image=dummy_image,
        gt_alpha_mask=None,
        image_name=image_name,
        uid=uid,
        K=K,
        sky_mask=None,
        normal=None,
        depth=None,
        data_device=data_device,
    )


# ===========================================================================
# Resize helper (mirrors InstantSplat render_path() resize logic)
# ===========================================================================

def _apply_resize(view: Camera, method: str) -> Camera:
    """Mutate a camera's resolution in-place before rasterisation."""
    if method == "original":
        return view
    if method == "crop":
        new_w = new_h = 512
    elif method == "pad":
        new_w = new_h = max(view.image_width, view.image_height)
    else:
        raise ValueError(f"Unknown resize method: {method!r}. Use crop | pad | original")

    fx = fov2focal(view.FoVx, view.image_width)
    fy = fov2focal(view.FoVy, view.image_height)
    view.original_image = torch.zeros(
        (3, new_h, new_w), device=view.original_image.device
    )
    view.image_width  = new_w
    view.image_height = new_h
    view.FoVx = focal2fov(fx, new_w)
    view.FoVy = focal2fov(fy, new_h)
    view.projection_matrix = (
        getProjectionMatrix(
            znear=view.znear, zfar=view.zfar, fovX=view.FoVx, fovY=view.FoVy
        )
        .transpose(0, 1)
        .cuda()
        .float()
    )
    view.full_proj_transform = (
        view.world_view_transform.unsqueeze(0)
        .bmm(view.projection_matrix.unsqueeze(0))
        .squeeze(0)
    )
    return view


# ===========================================================================
# Video helper
# ===========================================================================

def _images_to_video(image_dir: str, output_path: str, fps: int = 30) -> None:
    files = sorted(f for f in os.listdir(image_dir) if f.lower().endswith(".png"))
    if not files:
        print(f"[worker] No PNG frames in {image_dir}; skipping video.")
        return
    first = cv2.imread(os.path.join(image_dir, files[0]))
    h, w  = first.shape[:2]
    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )
    for fname in files:
        writer.write(cv2.imread(os.path.join(image_dir, fname)))
    writer.release()
    print(f"[worker] Video saved → {output_path}")


# ===========================================================================
# Minimal pipeline params (avoids argparse dependency on GaussianPro's arguments/)
# ===========================================================================

class _PipelineParams:
    convert_SHs_python = False
    compute_cov3D_python = False
    debug = False


# ===========================================================================
# Core render logic
# ===========================================================================

@torch.no_grad()
def _render_camera_list(
    cams,
    gaussians: GaussianModel,
    pipeline,
    background: torch.Tensor,
    out_dir: str,
    resize: str,
    desc: str = "Rendering",
    quiet: bool = False,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for idx, view in enumerate(tqdm(cams, desc=desc, disable=quiet)):
        view = _apply_resize(view, resize)
        pkg  = render(view, gaussians, pipeline, background)
        torchvision.utils.save_image(
            pkg["render"], os.path.join(out_dir, f"{idx:05d}.png")
        )


def run(
    model_path: str,
    cameras_file: str,
    out_root: str,
    ply_path: str,
    resize: str = "crop",
    fps: int = 30,
    white_background: bool = False,
    quiet: bool = False,
) -> None:
    # Load camera payload
    with open(cameras_file) as f:
        payload = json.load(f)

    train_cam_dicts = payload["train_cams"]
    orbit_cam_dicts = payload["orbit_cams"]

    print(f"[worker] Building {len(train_cam_dicts)} train cameras …")
    train_cams = [_build_camera(d) for d in train_cam_dicts]
    print(f"[worker] Building {len(orbit_cam_dicts)} orbit cameras …")
    orbit_cams = [_build_camera(d) for d in orbit_cam_dicts]

    # Load Gaussian model directly from PLY (no Scene needed)
    print(f"[worker] Loading gaussians from {ply_path} …")
    gaussians = GaussianModel(sh_degree=3)
    gaussians.load_ply(ply_path)

    bg_color   = [1, 1, 1] if white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    pipeline   = _PipelineParams()

    train_out = os.path.join(out_root, "train_renders")
    orbit_out = os.path.join(out_root, "renders")

    # Train-view re-renders
    _render_camera_list(
        train_cams, gaussians, pipeline, background,
        out_dir=train_out, resize=resize,
        desc="Train-view renders", quiet=quiet,
    )
    _images_to_video(train_out, os.path.join(out_root, "train_renders.mp4"), fps=fps)

    # Orbit renders
    _render_camera_list(
        orbit_cams, gaussians, pipeline, background,
        out_dir=orbit_out, resize=resize,
        desc="Orbit renders", quiet=quiet,
    )
    _images_to_video(
        orbit_out,
        os.path.join(out_root, "interpolation_renders.mp4"),
        fps=fps,
    )


# ===========================================================================
# CLI
# ===========================================================================

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GaussianPro render worker for ellipse-orbit renders."
    )
    parser.add_argument("--model_path",   type=str, required=True)
    parser.add_argument("--cameras_file", type=str, required=True,
                        help="Path to temp JSON with train_cams + orbit_cams.")
    parser.add_argument("--out_root",     type=str, required=True,
                        help="Output directory root (render/ours_<iter>/).")
    parser.add_argument("--ply_path",     type=str, required=True,
                        help="Path to point_cloud.ply for the target iteration.")
    parser.add_argument("--resize",       type=str, default="crop",
                        choices=["crop", "pad", "original"])
    parser.add_argument("--fps",          type=int, default=30)
    parser.add_argument("--white_background", action="store_true", default=False)
    parser.add_argument("--quiet",        action="store_true", default=False)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(
        model_path=args.model_path,
        cameras_file=args.cameras_file,
        out_root=args.out_root,
        ply_path=args.ply_path,
        resize=args.resize,
        fps=args.fps,
        white_background=args.white_background,
        quiet=args.quiet,
    )
