"""
gaussianpro_render_path.py — Ellipse-orbit render for GaussianPro checkpoints
==============================================================================
Mirrors the 120-frame orbit produced by InstantSplat's ``render_path()`` in
``3dgs.py``, but operates on a trained GaussianPro scene.

Architecture
------------
This script is a **pure orchestrator** — it contains no GaussianPro imports.
All GaussianPro-dependent work (loading the model, building Camera objects,
rasterisation) is delegated to ``_gaussianpro_render_path_worker.py``, which
is invoked as a subprocess with ``cwd`` and ``PYTHONPATH`` set to the
GaussianPro root.  This mirrors the pattern used by every other stage wrapper
in this pipeline (``gaussianpro_render.py``, ``gaussianpro_train.py``), and
avoids the ``ModuleNotFoundError: No module named 'utils.system_utils'``
conflict that arises when the VGGT installation's own ``utils`` package is
already cached in ``sys.modules``.

The outer script is responsible for:
  1. Reading ``<model_path>/cameras.json`` to recover training-camera poses.
  2. Generating the ellipse-orbit camera poses (pure NumPy / SciPy).
  3. Serialising orbit + train camera params to a temporary JSON file.
  4. Invoking the worker subprocess with that JSON.

The worker is responsible for:
  1. Building GaussianPro ``Camera`` objects from the JSON params.
  2. Loading the Gaussian model (``point_cloud.ply``).
  3. Rendering every camera and saving PNG frames.
  4. Stitching frames into MP4 videos.

Usage
-----
    python gaussianpro_render_path.py \\
        --model_path  data/scenes/<dataset>/output_30000_gp_depth_prior/ \\
        --source_path data/scenes/<dataset>/

    # Fewer frames, native resolution
    python gaussianpro_render_path.py \\
        --model_path  data/scenes/<dataset>/output_30000_gp/ \\
        --source_path data/scenes/<dataset>/ \\
        --n_frames 60 --resize original

Output
------
    <model_path>/render/ours_<iter>/renders/            PNG frames (00000.png …)
    <model_path>/render/ours_<iter>/interpolation_renders.mp4
    <model_path>/render/ours_<iter>/train_renders/      training-view re-renders
    <model_path>/render/ours_<iter>/train_renders.mp4
"""

import argparse
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import List

import numpy as np
from scipy.special import softmax

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE     = Path(__file__).resolve().parent
_GP_ROOT  = _HERE / "third_party" / "GaussianPro"
_WORKER   = _HERE / "_gaussianpro_render_path_worker.py"


# ===========================================================================
# Minimal math helpers (no GaussianPro imports)
# ===========================================================================

def _focal2fov(focal: float, pixels: int) -> float:
    return 2.0 * math.atan(pixels / (2.0 * focal))


def _searchForMaxIteration(point_cloud_dir: str) -> int:
    iters = [
        int(d.split("_")[-1])
        for d in os.listdir(point_cloud_dir)
        if d.startswith("iteration_")
    ]
    if not iters:
        raise RuntimeError(f"No iteration_* folders found in {point_cloud_dir}")
    return max(iters)


# ===========================================================================
# Ellipse-path generation (ported from instantsplat/gaussian-splatting/utils/camera_utils.py)
# ===========================================================================

def _normalize(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x)


def _viewmatrix(lookdir: np.ndarray, up: np.ndarray, position: np.ndarray) -> np.ndarray:
    vec2 = _normalize(lookdir)
    vec0 = _normalize(np.cross(up, vec2))
    vec1 = _normalize(np.cross(vec2, vec0))
    return np.stack([vec0, vec1, vec2, position], axis=1)  # (3, 4)


def _focus_point_fn(poses: np.ndarray) -> np.ndarray:
    directions = poses[:, :3, 2:3]
    origins     = poses[:, :3, 3:4]
    m    = np.eye(3) - directions * np.transpose(directions, [0, 2, 1])
    mt_m = np.transpose(m, [0, 2, 1]) @ m
    return np.linalg.inv(mt_m.mean(0)) @ (mt_m @ origins).mean(0)[:, 0]


def _pad_poses(p: np.ndarray) -> np.ndarray:
    bottom = np.broadcast_to([0, 0, 0, 1.0], p[..., :1, :4].shape)
    return np.concatenate([p[..., :3, :4], bottom], axis=-2)


def _unpad_poses(p: np.ndarray) -> np.ndarray:
    return p[..., :3, :4]


def _transform_poses_pca(poses: np.ndarray):
    t      = poses[:, :3, 3]
    t_mean = t.mean(axis=0)
    t      = t - t_mean
    eigval, eigvec = np.linalg.eig(t.T @ t)
    inds   = np.argsort(eigval)[::-1]
    eigvec = eigvec[:, inds]
    rot    = eigvec.T
    if np.linalg.det(rot) < 0:
        rot = np.diag(np.array([1, 1, -1])) @ rot
    transform = np.concatenate([rot, rot @ -t_mean[:, None]], -1)
    poses_recentered = _unpad_poses(transform @ _pad_poses(poses))
    transform = np.concatenate([transform, np.eye(4)[3:]], axis=0)
    if poses_recentered.mean(axis=0)[2, 1] < 0:
        poses_recentered = np.diag(np.array([1, -1, -1])) @ poses_recentered
        transform        = np.diag(np.array([1, -1, -1, 1])) @ transform
    scale_factor = 1.0 / np.max(np.abs(poses_recentered[:, :3, 3]))
    poses_recentered[:, :3, 3] *= scale_factor
    return poses_recentered, transform, scale_factor


def _invert_transform_poses_pca(poses_recentered, transform, scale_factor):
    poses_recentered[:, :3, 3] /= scale_factor
    return _unpad_poses(np.linalg.inv(transform) @ _pad_poses(poses_recentered))


def _generate_ellipse_path(poses: np.ndarray, n_frames: int = 120,
                            z_amplitude_frac: float = 0.35) -> np.ndarray:
    """Return ``(n_frames, 3, 4)`` C2W matrices on a sinusoidal-elevation orbit.

    For object-centric orbit training data (e.g. CogVideoX), the training cameras
    already span the full horizontal orbit, so a flat XY ellipse would reproduce
    the same trajectory rather than producing novel views.  Adding a sinusoidal Z
    component makes the camera rise above and dip below the training orbit plane,
    visiting elevations that the training frames never covered.

    Args:
        poses: (N, 3, 4) C2W matrices in NeRF convention (after Y/Z flip).
        n_frames: Number of frames in the output orbit.
        z_amplitude_frac: Elevation amplitude as a fraction of the mean XY orbit
            radius.  0.35 means the camera rises/dips ±35% of the orbit radius.
            Set to 0.0 to recover the original flat-ellipse behaviour.
    """
    center = _focus_point_fn(poses)
    # Use the full 3-D focus point so the bounding box is correctly centred on
    # the scene regardless of any residual Z offset after PCA alignment.
    offset = center.copy()
    sc     = np.percentile(np.abs(poses[:, :3, 3] - offset), 100, axis=0)
    low, high = -sc + offset, sc + offset

    # Elevation amplitude: a fraction of the mean horizontal radius so it scales
    # naturally with scene size.  A minimum guard keeps it non-zero even for
    # scenes where sc[0]/sc[1] are tiny (degenerate point-cloud cases).
    mean_xy_radius = 0.5 * (sc[0] + sc[1])
    z_amplitude    = z_amplitude_frac * max(mean_xy_radius, 1e-3)

    def get_positions(theta):
        return np.stack([
            low[0] + (high - low)[0] * (np.cos(theta) * 0.5 + 0.5),
            low[1] + (high - low)[1] * (np.sin(theta) * 0.5 + 0.5),
            offset[2] + z_amplitude * np.sin(theta),
        ], -1)

    theta     = np.linspace(0, 2.0 * np.pi, n_frames + 1, endpoint=True)
    positions = get_positions(theta)

    lengths   = np.linalg.norm(positions[1:] - positions[:-1], axis=-1)
    w         = softmax(np.log(np.maximum(lengths, 1e-8)))
    cw        = np.concatenate([[0.0], np.minimum(1.0, np.cumsum(w[:-1])), [1.0]])
    u         = np.linspace(0, 1.0 - np.finfo(np.float32).eps, n_frames + 1)
    theta_rs  = np.interp(u, cw, theta)
    positions = get_positions(theta_rs)[:-1]  # drop duplicated endpoint

    avg_up = poses[:, :3, 1].mean(0)
    avg_up /= np.linalg.norm(avg_up)
    ind_up = np.argmax(np.abs(avg_up))
    up     = np.eye(3)[ind_up] * np.sign(avg_up[ind_up])

    return np.stack([_viewmatrix(p - center, up, p) for p in positions])


# ===========================================================================
# Camera JSON helpers
# ===========================================================================

def _load_train_cameras_from_json(cameras_json_path: str) -> List[dict]:
    """
    Read <model_path>/cameras.json and return a list of camera dicts, each with:
      c2w_rotation  : (3, 3) ndarray  — C2W rotation
      c2w_position  : (3,)  ndarray   — camera centre in world
      fx, fy        : float
      width, height : int
    """
    with open(cameras_json_path) as f:
        entries = json.load(f)
    cams = []
    for e in entries:
        cams.append({
            "c2w_rotation": np.array(e["rotation"], dtype=np.float64),
            "c2w_position": np.array(e["position"], dtype=np.float64),
            "fx":    float(e["fx"]),
            "fy":    float(e["fy"]),
            "width":  int(e["width"]),
            "height": int(e["height"]),
        })
    return cams


def _cam_dict_to_serialisable(uid: int, image_name: str, c2w_rotation, c2w_position,
                               fx: float, fy: float, width: int, height: int) -> dict:
    """Convert one camera to a JSON-serialisable dict for the worker."""
    c2w4 = np.eye(4)
    c2w4[:3, :3] = c2w_rotation
    c2w4[:3, 3]  = c2w_position
    w2c = np.linalg.inv(c2w4)
    # Camera.R = W2C_rotation.T  (matches getWorld2View2 convention)
    R_cam = w2c[:3, :3].T
    T_cam = w2c[:3, 3]
    return {
        "uid":        uid,
        "image_name": image_name,
        "R_cam":      R_cam.tolist(),
        "T_cam":      T_cam.tolist(),
        "FoVx":       _focal2fov(fx, width),
        "FoVy":       _focal2fov(fy, height),
        "fx":         fx,
        "fy":         fy,
        "width":      width,
        "height":     height,
    }


def _build_camera_payload(train_cams: List[dict], n_frames: int,
                           z_amplitude_frac: float = 0.35) -> dict:
    """
    Given the list of train camera dicts from cameras.json, compute the ellipse
    orbit and return a dict with keys "train_cams" and "orbit_cams", each a
    list of serialisable camera dicts.
    """
    # Build C2W poses array for the ellipse generator.
    c2w_list = []
    for cam in train_cams:
        c2w = np.eye(4)
        c2w[:3, :3] = cam["c2w_rotation"]
        c2w[:3, 3]  = cam["c2w_position"]
        c2w_list.append(c2w[:3, :4])
    poses = np.array(c2w_list)  # (N, 3, 4)

    # Flip y/z to NeRF convention for the ellipse algorithm, then flip back.
    poses[:, :, 1:3] *= -1
    poses, transform, scale_factor = _transform_poses_pca(poses)
    render_poses_pca = _generate_ellipse_path(poses, n_frames,
                                              z_amplitude_frac=z_amplitude_frac)
    render_poses = _invert_transform_poses_pca(render_poses_pca, transform, scale_factor)
    render_poses[:, :, 1:3] *= -1

    # Borrow intrinsics from the first training camera.
    ref = train_cams[0]
    fx, fy, w, h = ref["fx"], ref["fy"], ref["width"], ref["height"]

    train_cam_dicts = [
        _cam_dict_to_serialisable(
            i, f"{i:05d}.png",
            cam["c2w_rotation"], cam["c2w_position"],
            cam["fx"], cam["fy"], cam["width"], cam["height"],
        )
        for i, cam in enumerate(train_cams)
    ]
    orbit_cam_dicts = [
        _cam_dict_to_serialisable(
            i, f"{i:05d}.png",
            c2w[:3, :3], c2w[:3, 3],
            fx, fy, w, h,
        )
        for i, c2w in enumerate(render_poses)
    ]

    return {"train_cams": train_cam_dicts, "orbit_cams": orbit_cam_dicts}


# ===========================================================================
# CLI
# ===========================================================================

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render a 120-frame ellipse-orbit video from a GaussianPro checkpoint, "
            "matching the InstantSplat render_path() pipeline."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model_path", type=str, required=True,
        help="Trained GaussianPro model directory "
             "(e.g. data/scenes/<dataset>/output_30000_gp_depth_prior/).",
    )
    parser.add_argument(
        "--source_path", type=str, required=True,
        help="COLMAP scene directory (e.g. data/scenes/<dataset>/).",
    )
    parser.add_argument(
        "--iteration", type=int, default=-1,
        help="Checkpoint iteration to load.  -1 = latest saved checkpoint.",
    )
    parser.add_argument(
        "--n_frames", type=int, default=120,
        help="Number of frames in the ellipse orbit.",
    )
    parser.add_argument(
        "--resize", type=str, default="crop",
        choices=["crop", "pad", "original"],
        help=(
            "Frame resize strategy: "
            "'crop' → 512×512, "
            "'pad' → square at max(w,h), "
            "'original' → native camera resolution."
        ),
    )
    parser.add_argument("--fps",  type=int, default=30)
    parser.add_argument(
        "--z_amplitude_frac", type=float, default=0.35,
        help=(
            "Elevation amplitude of the novel-view orbit as a fraction of the "
            "mean horizontal orbit radius.  0.35 (default) makes the camera "
            "rise/dip ±35%% of the orbit radius above/below the training plane, "
            "producing views at elevations not covered by the training orbit. "
            "Set to 0.0 to recover the original flat-ellipse behaviour."
        ),
    )
    parser.add_argument(
        "--white_background", action="store_true", default=False,
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--quiet", action="store_true", default=False)
    return parser.parse_args()


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    args = _parse_args()

    model_path  = Path(args.model_path).resolve()
    source_path = Path(args.source_path).resolve()

    if not model_path.is_dir():
        sys.exit(
            f"[render_path] ERROR: model_path not found: {model_path}\n"
            "Run gaussianpro_train.py first."
        )
    if not source_path.is_dir():
        sys.exit(f"[render_path] ERROR: source_path not found: {source_path}")

    cameras_json = model_path / "cameras.json"
    if not cameras_json.is_file():
        sys.exit(
            f"[render_path] ERROR: cameras.json not found at {cameras_json}\n"
            "This file is written by GaussianPro during training."
        )

    # Resolve iteration.
    pc_dir = model_path / "point_cloud"
    if args.iteration == -1:
        iteration = _searchForMaxIteration(str(pc_dir))
    else:
        iteration = args.iteration

    ply_path = pc_dir / f"iteration_{iteration}" / "point_cloud.ply"
    if not ply_path.is_file():
        sys.exit(f"[render_path] ERROR: point cloud not found: {ply_path}")

    device_idx = args.device.split(":")[-1] if ":" in args.device else "0"
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", device_idx)

    print("=" * 60)
    print("  DimensionX — GaussianPro Render Path (ellipse orbit)")
    print(f"  model_path       : {model_path}")
    print(f"  source_path      : {source_path}")
    print(f"  iteration        : {iteration}")
    print(f"  n_frames         : {args.n_frames}")
    print(f"  z_amplitude_frac : {args.z_amplitude_frac}")
    print(f"  resize           : {args.resize}")
    print(f"  fps              : {args.fps}")
    print(f"  white_background : {args.white_background}")
    print("=" * 60)

    # --- Generate camera payload -------------------------------------------
    print("\n[render_path] Building ellipse orbit from cameras.json …")
    train_cams = _load_train_cameras_from_json(str(cameras_json))
    payload    = _build_camera_payload(train_cams, n_frames=args.n_frames,
                                       z_amplitude_frac=args.z_amplitude_frac)
    print(f"  train cameras : {len(payload['train_cams'])}")
    print(f"  orbit cameras : {len(payload['orbit_cams'])}")

    # --- Write payload to a temp file --------------------------------------
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, prefix="gp_render_cams_"
    ) as tf:
        json.dump(payload, tf)
        cameras_tmp = tf.name

    # --- Invoke worker subprocess -------------------------------------------
    out_root = model_path / "render" / f"ours_{iteration}"

    cmd = [
        sys.executable,
        str(_WORKER),
        "--model_path",    str(model_path),
        "--cameras_file",  cameras_tmp,
        "--out_root",      str(out_root),
        "--ply_path",      str(ply_path),
        "--resize",        args.resize,
        "--fps",           str(args.fps),
    ]
    if args.white_background:
        cmd.append("--white_background")
    if args.quiet:
        cmd.append("--quiet")

    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        str(_GP_ROOT) + os.pathsep + existing_pythonpath
        if existing_pythonpath
        else str(_GP_ROOT)
    )

    print(f"\n[render_path] Launching worker …")
    print(f"  CMD: {' '.join(cmd)}\n")

    import subprocess
    result = subprocess.run(cmd, cwd=str(_GP_ROOT), env=env)

    # Cleanup temp file
    try:
        os.unlink(cameras_tmp)
    except OSError:
        pass

    if result.returncode != 0:
        sys.exit(
            f"[render_path] Worker exited with code {result.returncode}."
        )

    mp4_path = out_root / "interpolation_renders.mp4"
    print(f"\n[render_path] Done.")
    print(f"  Orbit frames : {out_root / 'renders'}")
    print(f"  Orbit video  : {mp4_path}")
    print(f"  Train frames : {out_root / 'train_renders'}")


if __name__ == "__main__":
    main()
