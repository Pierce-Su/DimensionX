"""
Stage B — VGGT Geometry Estimation
====================================
Reads extracted frames from  data/images/{dataset}/
Runs VGGT to estimate camera poses, depth maps, and a dense point cloud,
then writes:

  data/scenes/{dataset}/
      images/              copies of the input frames
      sparse/
          cameras.bin      COLMAP binary
          images.bin
          points3D.bin
          points.ply       coloured point cloud (visualisation aid)
      depth_maps/          {i}.npy  float32 (H_vggt, W_vggt)
      confidence_maps/     {i}.npy  float32 (H_vggt, W_vggt)
      normals/             {i}.npy  float32 (H_vggt, W_vggt, 3)

Usage
-----
    python vggt_inference.py --dataset <name> [--use_ba] [--device cuda:0]

The dataset name maps to:
    images in:  data/images/<dataset>/          (0.png, 1.png, …)
    output to:  data/scenes/<dataset>/
"""

import sys
import os
import copy
import glob
import random
import shutil
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Make the bundled VGGT importable
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_VGGT_ROOT  = _SCRIPT_DIR / "third_party" / "vggt"
sys.path.insert(0, str(_VGGT_ROOT))

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images_square
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.helper import create_pixel_coordinate_grid, randomly_limit_trues
from vggt.dependency.np_to_pycolmap import (
    batch_np_matrix_to_pycolmap,
    batch_np_matrix_to_pycolmap_wo_track,
)

from utils.depth_to_normal import batch_point_map_to_normals

# ---------------------------------------------------------------------------
# Constants — match demo_colmap.py defaults
# ---------------------------------------------------------------------------
VGGT_FIXED_RESOLUTION = 518   # resolution VGGT runs at internally
IMG_LOAD_RESOLUTION   = 1024  # resolution images are loaded at before feeding to VGGT
CONF_THRES_VALUE      = 5.0   # depth confidence threshold (no-BA path) — default matches VGGT demo
CONF_THRES_FALLBACK_PERCENTILE = 80  # percentile fallback when absolute threshold yields < MIN_POINTS
MIN_POINTS_BEFORE_FALLBACK     = 1_000
MAX_POINTS_FOR_COLMAP = 100_000


# ---------------------------------------------------------------------------
# PLY export helper (dependency-free)
# ---------------------------------------------------------------------------
def save_points_ply(points: np.ndarray, colors: np.ndarray, output_path: Path) -> None:
    """
    Save a colored point cloud to ASCII PLY.

    Args:
        points: (M, 3) float array
        colors: (M, 3) uint8/int/float array
        output_path: output .ply path
    """
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    colors = np.asarray(colors).reshape(-1, 3)

    if points.shape[0] != colors.shape[0]:
        raise ValueError(
            f"points/color count mismatch: {points.shape[0]} vs {colors.shape[0]}"
        )

    # Normalize colors into [0, 255] uint8.
    if colors.dtype.kind == "f":
        cmax = float(np.nanmax(colors)) if colors.size else 0.0
        if cmax <= 1.0:
            colors = colors * 255.0
    colors = np.clip(colors, 0, 255).astype(np.uint8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="ascii") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {points.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")

        for (x, y, z), (r, g, b) in zip(points, colors):
            f.write(f"{x:.7f} {y:.7f} {z:.7f} {int(r)} {int(g)} {int(b)}\n")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="VGGT geometry estimation for the DimensionX pipeline."
    )
    parser.add_argument(
        "--dataset", type=str, required=True,
        help="Dataset name. Reads from data/images/<dataset>/, writes to data/scenes/<dataset>/.",
    )
    parser.add_argument(
        "--images_dir", type=str, default=None,
        help="Override images directory (default: data/images/<dataset>/).",
    )
    parser.add_argument(
        "--scenes_dir", type=str, default=None,
        help="Override scenes root directory (default: data/scenes/).",
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device (default: cuda if available, else cpu).",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to a local VGGT-1B checkpoint (.pt). "
             "If not provided, weights are downloaded from HuggingFace.",
    )
    # --- BA options ---
    parser.add_argument(
        "--use_ba", action="store_true", default=False,
        help="Enable bundle adjustment via pycolmap (slower but more accurate).",
    )
    parser.add_argument("--max_reproj_error", type=float, default=8.0)
    parser.add_argument("--shared_camera",    action="store_true", default=False)
    parser.add_argument("--camera_type",      type=str, default="SIMPLE_PINHOLE")
    parser.add_argument("--vis_thresh",       type=float, default=0.2)
    parser.add_argument("--query_frame_num",  type=int,   default=8)
    parser.add_argument("--max_query_pts",    type=int,   default=4096)
    parser.add_argument(
        "--fine_tracking", action="store_true", default=True,
        help="Use fine-grained tracking during BA (slower but more accurate).",
    )
    # --- depth confidence filter ---
    parser.add_argument(
        "--conf_thres_value", type=float, default=CONF_THRES_VALUE,
        help="Absolute confidence threshold for depth-map filtering (no-BA path). "
             "Only pixels with confidence >= this value are kept. "
             "With VGGT's expp1 activation the minimum possible value is 1.0; "
             "typical in-distribution scenes yield values of 5–50+. "
             "For out-of-distribution footage (aerial, satellite, etc.) all confidences "
             "collapse to ~1.0 and the fallback percentile logic takes over automatically. "
             "Default: 5.0 (matches the original VGGT demo_colmap.py).",
    )
    # --- memory / frame budget ---
    parser.add_argument(
        "--max_frames", type=int, default=None,
        help="Maximum number of frames fed to VGGT. "
             "When the extracted frame count exceeds this value the frames are "
             "uniformly subsampled before the VGGT forward pass. "
             "Recommended: ≤48 for 48 GB VRAM, ≤32 for 24 GB VRAM. "
             "Default: no limit (use all frames).",
    )
    # --- misc ---
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument("--save_glb", action="store_true", default=False,
                        help="Export a scene.glb visualisation file.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: str | None, device: str) -> VGGT:
    model = VGGT()
    if checkpoint_path and Path(checkpoint_path).is_file():
        print(f"Loading VGGT weights from local checkpoint: {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state)
    else:
        _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
        print(f"Downloading VGGT weights from: {_URL}")
        state = torch.hub.load_state_dict_from_url(_URL, map_location="cpu")
        model.load_state_dict(state)
    model.eval()
    return model.to(device)


def run_vggt(
    model: VGGT,
    images: torch.Tensor,
    dtype: torch.dtype,
    resolution: int = VGGT_FIXED_RESOLUTION,
):
    """
    Run VGGT aggregator + camera_head + depth_head at a fixed square resolution.

    Args:
        images:     (N, 3, H, W) float32 tensor on the correct device.
        dtype:      bfloat16 or float16.
        resolution: internal square resolution for VGGT (default 518).

    Returns:
        extrinsic:  np.ndarray (N, 3, 4) — OpenCV convention, world-to-camera.
        intrinsic:  np.ndarray (N, 3, 3)
        depth_map:  np.ndarray (N, H_vggt, W_vggt) float32
        depth_conf: np.ndarray (N, H_vggt, W_vggt) float32
    """
    assert images.ndim == 4 and images.shape[1] == 3

    images_vggt = F.interpolate(
        images, size=(resolution, resolution), mode="bilinear", align_corners=False
    )

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            images_b = images_vggt[None]  # (1, N, 3, H, W)
            aggregated_tokens_list, ps_idx = model.aggregator(images_b)

            pose_enc = model.camera_head(aggregated_tokens_list)[-1]
            extrinsic, intrinsic = pose_encoding_to_extri_intri(
                pose_enc, images_b.shape[-2:]
            )

            depth_map, depth_conf = model.depth_head(
                aggregated_tokens_list, images_b, ps_idx
            )

    extrinsic  = extrinsic.squeeze(0).cpu().numpy()   # (N, 3, 4)
    intrinsic  = intrinsic.squeeze(0).cpu().numpy()   # (N, 3, 3)
    depth_map  = depth_map.squeeze(0).cpu().numpy()   # (N, H, W)
    depth_conf = depth_conf.squeeze(0).cpu().numpy()  # (N, H, W)

    return extrinsic, intrinsic, depth_map, depth_conf


# ---------------------------------------------------------------------------
# COLMAP renaming / rescaling  (copied verbatim from demo_colmap.py)
# ---------------------------------------------------------------------------

def rename_colmap_recons_and_rescale_camera(
    reconstruction,
    image_paths,
    original_coords,
    img_size,
    shift_point2d_to_original_res=False,
    shared_camera=False,
):
    rescale_camera = True
    for pyimageid in reconstruction.images:
        pyimage  = reconstruction.images[pyimageid]
        pycamera = reconstruction.cameras[pyimage.camera_id]
        pyimage.name = image_paths[pyimageid - 1]

        if rescale_camera:
            pred_params    = copy.deepcopy(pycamera.params)
            real_image_size = original_coords[pyimageid - 1, -2:]
            resize_ratio   = max(real_image_size) / img_size
            pred_params    = pred_params * resize_ratio
            real_pp        = real_image_size / 2
            pred_params[-2:] = real_pp
            pycamera.params = pred_params
            pycamera.width  = int(real_image_size[0])
            pycamera.height = int(real_image_size[1])

        if shift_point2d_to_original_res:
            top_left = original_coords[pyimageid - 1, :2]
            for point2D in pyimage.points2D:
                point2D.xy = (point2D.xy - top_left) * resize_ratio

        if shared_camera:
            rescale_camera = False

    return reconstruction


# ---------------------------------------------------------------------------
# Image discovery
# ---------------------------------------------------------------------------

def collect_image_paths(images_dir: Path) -> list[Path]:
    """
    Return image paths sorted by integer stem (0.png < 1.png < … < 9.png < 10.png).
    Supports .png, .jpg, .jpeg, .JPG, .JPEG.
    """
    exts = {".png", ".jpg", ".jpeg"}
    paths = [p for p in images_dir.iterdir() if p.suffix.lower() in exts]
    if not paths:
        raise FileNotFoundError(f"No images found in {images_dir}")

    def _sort_key(p: Path) -> int:
        try:
            return int(p.stem)
        except ValueError:
            return hash(p.stem)

    return sorted(paths, key=_sort_key)


# ---------------------------------------------------------------------------
# Auxiliary artifact saving
# ---------------------------------------------------------------------------

def save_depth_confidence_normals(
    scene_dir: Path,
    depth_map: np.ndarray,
    depth_conf: np.ndarray,
    points_3d: np.ndarray,
    image_names: list,
):
    """
    Persist per-frame depth maps, confidence maps, and surface normals.

    Files are named after the source image stem (e.g. "144.png" → "144.npy") so
    that GaussianPro's dataset_readers can locate them by replacing the "images"
    directory with "normals" / "metricdepth" in the image path.

    Shapes:
        depth_map:  (N, H, W)
        depth_conf: (N, H, W)
        points_3d:  (N, H, W, 3)
    """
    depth_dir = scene_dir / "depth_maps"
    conf_dir  = scene_dir / "confidence_maps"
    norm_dir  = scene_dir / "normals"
    for d in (depth_dir, conf_dir, norm_dir):
        d.mkdir(parents=True, exist_ok=True)

    normals = batch_point_map_to_normals(points_3d)  # (N, H, W, 3), unit vectors in [-1, 1]

    N = depth_map.shape[0]
    for i in range(N):
        stem = Path(image_names[i]).stem
        np.save(str(depth_dir / f"{stem}.npy"), depth_map[i].astype(np.float32))
        np.save(str(conf_dir  / f"{stem}.npy"), depth_conf[i].astype(np.float32))
        # Store normals as (3, H, W) in [0, 1] range.
        # GaussianPro's loadCam applies .transpose((1,2,0)) expecting channels-first (3,H,W)
        # input, then applies (n - 0.5) * 2 to recover [-1, 1] unit vectors.
        normal_01 = ((normals[i] + 1.0) / 2.0).astype(np.float32)       # (H, W, 3)
        np.save(str(norm_dir  / f"{stem}.npy"), np.transpose(normal_01, (2, 0, 1)))  # (3, H, W)

    print(f"Saved depth maps, confidence maps, normals for {N} frames.")


# ---------------------------------------------------------------------------
# GLB export  (optional visualisation)
# ---------------------------------------------------------------------------

def _try_save_glb(scene_dir: Path, points_3d: np.ndarray, points_rgb: np.ndarray):
    try:
        import trimesh
        ply_path = scene_dir / "sparse" / "points.ply"
        if ply_path.exists():
            pc = trimesh.load(str(ply_path))
            glb_path = scene_dir / "scene.glb"
            pc.export(str(glb_path))
            print(f"Saved GLB: {glb_path}")
    except Exception as exc:
        print(f"[WARN] Could not save GLB: {exc}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = args.device
    dtype  = torch.bfloat16 if (
        torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    ) else torch.float16

    print(f"[vggt_inference] dataset={args.dataset}  device={device}  dtype={dtype}")
    print(f"[vggt_inference] bundle adjustment={'enabled' if args.use_ba else 'disabled'}")

    # ------------------------------------------------------------------ paths
    base_dir    = _SCRIPT_DIR
    images_dir  = Path(args.images_dir) if args.images_dir else base_dir / "data" / "images" / args.dataset
    scenes_root = Path(args.scenes_dir) if args.scenes_dir else base_dir / "data" / "scenes"
    scene_dir   = scenes_root / args.dataset

    out_images_dir = scene_dir / "images"
    sparse_dir     = scene_dir / "sparse"
    out_images_dir.mkdir(parents=True, exist_ok=True)
    sparse_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------- collect images
    image_path_list = collect_image_paths(images_dir)
    base_names      = [p.name for p in image_path_list]
    print(f"Found {len(image_path_list)} images in {images_dir}")

    # ------------------------------------------------------ frame subsampling
    if args.max_frames is not None and len(image_path_list) > args.max_frames:
        total = len(image_path_list)
        indices = [round(i * (total - 1) / (args.max_frames - 1))
                   for i in range(args.max_frames)]
        image_path_list = [image_path_list[i] for i in indices]
        base_names      = [p.name for p in image_path_list]
        print(
            f"[vggt_inference] --max_frames={args.max_frames}: "
            f"uniformly subsampled {total} → {len(image_path_list)} frames"
        )

    # Copy frames into scene images/ so the COLMAP reconstruction is self-contained.
    for src in image_path_list:
        dst = out_images_dir / src.name
        if not dst.exists():
            shutil.copy2(src, dst)

    # ---------------------------------------------------------- load + preprocess
    images, original_coords = load_and_preprocess_images_square(
        [str(p) for p in image_path_list], IMG_LOAD_RESOLUTION
    )
    images          = images.to(device)
    original_coords = original_coords.to(device)

    # ---------------------------------------------------------- load model
    model = load_model(args.checkpoint, device)

    # ---------------------------------------------------------- VGGT forward pass
    print("Running VGGT …")
    extrinsic, intrinsic, depth_map, depth_conf = run_vggt(
        model, images, dtype, VGGT_FIXED_RESOLUTION
    )

    # 3-D point map from depth + cameras (more accurate than point_head branch)
    points_3d = unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic)
    # points_3d: (N, H_vggt, W_vggt, 3)

    # -------------------------------------------- build COLMAP reconstruction
    if args.use_ba:
        # BA path — uses VGGSfM tracker for 2-D tracks
        from vggt.dependency.track_predict import predict_tracks
        import pycolmap

        scale = IMG_LOAD_RESOLUTION / VGGT_FIXED_RESOLUTION
        image_size = np.array(images.shape[-2:])

        with torch.cuda.amp.autocast(dtype=dtype):
            pred_tracks, pred_vis_scores, pred_confs, points_3d_ba, points_rgb = predict_tracks(
                images,
                conf=depth_conf,
                points_3d=points_3d,
                masks=None,
                max_query_pts=args.max_query_pts,
                query_frame_num=args.query_frame_num,
                keypoint_extractor="aliked+sp",
                fine_tracking=args.fine_tracking,
            )
        torch.cuda.empty_cache()

        intrinsic_scaled = intrinsic.copy()
        intrinsic_scaled[:, :2, :] *= scale
        track_mask = pred_vis_scores > args.vis_thresh

        reconstruction, _ = batch_np_matrix_to_pycolmap(
            points_3d_ba,
            extrinsic,
            intrinsic_scaled,
            pred_tracks,
            image_size,
            masks=track_mask,
            max_reproj_error=args.max_reproj_error,
            shared_camera=args.shared_camera,
            camera_type=args.camera_type,
            points_rgb=points_rgb,
        )
        if reconstruction is None:
            raise RuntimeError("COLMAP reconstruction failed with BA; try without --use_ba.")

        ba_options = pycolmap.BundleAdjustmentOptions()
        pycolmap.bundle_adjustment(reconstruction, ba_options)
        reconstruction_resolution = IMG_LOAD_RESOLUTION

    else:
        # Feedforward path (fast, no BA)
        import pycolmap  # needed for reconstruction.write()

        image_size = np.array([VGGT_FIXED_RESOLUTION, VGGT_FIXED_RESOLUTION])
        N, H, W, _ = points_3d.shape

        points_rgb_full = F.interpolate(
            images, size=(VGGT_FIXED_RESOLUTION, VGGT_FIXED_RESOLUTION),
            mode="bilinear", align_corners=False,
        )
        points_rgb_full = (points_rgb_full.cpu().numpy() * 255).astype(np.uint8)
        points_rgb_full = points_rgb_full.transpose(0, 2, 3, 1)  # (N, H, W, 3)

        # (N, H, W, 3) — pixel coordinates + frame index per valid point
        points_xyf = create_pixel_coordinate_grid(N, H, W)

        conf_mask = depth_conf >= args.conf_thres_value
        n_above = int(conf_mask.sum())

        if n_above < MIN_POINTS_BEFORE_FALLBACK:
            # Out-of-distribution scenes (aerial, satellite, …) make VGGT produce
            # very low confidence everywhere (all values ≈ 1.0 with expp1 activation).
            # Fall back to keeping the top CONF_THRES_FALLBACK_PERCENTILE% of pixels
            # by confidence so the point cloud is never empty.
            fallback_thres = float(np.percentile(depth_conf, CONF_THRES_FALLBACK_PERCENTILE))
            print(
                f"[vggt_inference] WARNING: only {n_above} pixels pass the absolute "
                f"confidence threshold ({args.conf_thres_value}). "
                f"Conf-value range: [{depth_conf.min():.4f}, {depth_conf.max():.4f}]. "
                f"Falling back to top-{100 - CONF_THRES_FALLBACK_PERCENTILE}% percentile "
                f"threshold = {fallback_thres:.6f}."
            )
            conf_mask = depth_conf >= fallback_thres

        conf_mask = randomly_limit_trues(conf_mask, MAX_POINTS_FOR_COLMAP)

        pts_filtered   = points_3d[conf_mask]
        xyf_filtered   = points_xyf[conf_mask]
        rgb_filtered   = points_rgb_full[conf_mask]

        print(f"Building COLMAP reconstruction from {pts_filtered.shape[0]} 3-D points …")
        reconstruction = batch_np_matrix_to_pycolmap_wo_track(
            pts_filtered,
            xyf_filtered,
            rgb_filtered,
            extrinsic,
            intrinsic,
            image_size,
            shared_camera=False,
            camera_type="PINHOLE",
        )
        reconstruction_resolution = VGGT_FIXED_RESOLUTION

    # Rename images and rescale camera params to original resolution.
    reconstruction = rename_colmap_recons_and_rescale_camera(
        reconstruction,
        base_names,
        original_coords.cpu().numpy(),
        img_size=reconstruction_resolution,
        shift_point2d_to_original_res=True,
        shared_camera=args.shared_camera if args.use_ba else False,
    )

    # ---------------------------------------------------- write COLMAP binary
    reconstruction.write(str(sparse_dir))

    # Also save a PLY point cloud for quick visualisation.
    try:
        pts_vis = pts_filtered if not args.use_ba else points_3d_ba
        rgb_vis = rgb_filtered if not args.use_ba else points_rgb
        ply_path = sparse_dir / "points.ply"
        save_points_ply(pts_vis, rgb_vis, ply_path)
        print(f"Saved point cloud PLY: {ply_path}")
    except Exception as exc:
        print(f"[WARN] Could not save points.ply: {exc}")

    print(f"COLMAP reconstruction written to: {sparse_dir}")

    # --------------------------------- save depth, confidence, normal maps
    save_depth_confidence_normals(scene_dir, depth_map, depth_conf, points_3d, base_names)

    if args.save_glb:
        pts_vis = pts_filtered if not args.use_ba else points_3d_ba
        rgb_vis = rgb_filtered if not args.use_ba else points_rgb
        _try_save_glb(scene_dir, pts_vis, rgb_vis)

    # ----------------------------------------------------------------- done
    print(f"\n[vggt_inference] Done. Scene outputs at: {scene_dir}")
    print(f"  images/         {len(list(out_images_dir.iterdir()))} files")
    print(f"  sparse/         cameras.bin  images.bin  points3D.bin  points.ply")
    print(f"  depth_maps/     {depth_map.shape[0]} x {depth_map.shape[1]}x{depth_map.shape[2]} float32")
    print(f"  confidence_maps/")
    print(f"  normals/")


if __name__ == "__main__":
    main()
