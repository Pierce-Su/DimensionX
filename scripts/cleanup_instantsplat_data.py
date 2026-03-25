#!/usr/bin/env python3
"""
Clean up InstantSplat outputs under instantsplat/data while keeping what you need to
reload / render 3D Gaussian Splatting later (original gaussian-splatting stack).

Typical layout (after DimensionX batch Stage 2):
  instantsplat/data/images/<case_tag>/          # frames extracted from video (duplicate of scenes/.../images)
  instantsplat/data/scenes/<case_tag>/
    images/ masks/ sparse/0/                    # COLMAP-style source (cameras.txt, images.txt, points3D.ply)
    depth_maps/ pointmaps/ confidence_map/ scene.glb   # Dust3R extras / viz
    output_<iter>_lpips_<...>/                  # 3DGS training output
      cfg_args, cameras.json, input.ply
      point_cloud/iteration_*/point_cloud.ply
      chkpnt*.pth, events.out.tfevents*, render/ ...

Conservative (default): remove obvious scratch (viz maps, TB, checkpoints, render previews).
Aggressive: also drop duplicate extracted frames, intermediate PLY checkpoints, input.ply copy.

Dry-run by default; pass --apply to delete.

Examples:
  python scripts/cleanup_instantsplat_data.py --root instantsplat/data
  python scripts/cleanup_instantsplat_data.py --root instantsplat/data --aggressive --apply
  python scripts/cleanup_instantsplat_data.py --root instantsplat/data --keep-confidence --apply
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Set


@dataclass(frozen=True)
class DeletionCandidate:
    path: Path
    reason: str


OUTPUT_DIR_RE = re.compile(r"^output_\d+_lpips_")


def _bytes(path: Path) -> int:
    try:
        return path.stat().st_size
    except FileNotFoundError:
        return 0


def _dir_size_bytes(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0
    if path.is_file():
        return _bytes(path)
    for p in path.rglob("*"):
        if p.is_file():
            total += _bytes(p)
    return total


def _is_tensorboard_event_file(p: Path) -> bool:
    return p.name.startswith("events.out.tfevents")


def _iter_scene_dirs(data_root: Path) -> Iterator[Path]:
    scenes = data_root / "scenes"
    if not scenes.is_dir():
        return
    for p in sorted(scenes.iterdir()):
        if p.is_dir():
            yield p


def _is_output_training_dir(p: Path) -> bool:
    return p.is_dir() and OUTPUT_DIR_RE.match(p.name) is not None


def _point_cloud_iteration_dirs(out_dir: Path) -> List[tuple[int, Path]]:
    pc_root = out_dir / "point_cloud"
    if not pc_root.is_dir():
        return []
    pairs: List[tuple[int, Path]] = []
    for child in pc_root.iterdir():
        if not child.is_dir():
            continue
        if not child.name.startswith("iteration_"):
            continue
        try:
            it = int(child.name.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        pairs.append((it, child))
    pairs.sort(key=lambda x: x[0])
    return pairs


def _dedupe_under_parent(candidates: Iterable[DeletionCandidate]) -> List[DeletionCandidate]:
    items = sorted({c.path.resolve(): c for c in candidates}.values(), key=lambda c: len(str(c.path)))
    kept: List[DeletionCandidate] = []
    kept_paths: List[Path] = []
    for c in items:
        cp = c.path.resolve()
        if any(str(cp).startswith(str(kp) + os.sep) for kp in kept_paths):
            continue
        kept.append(c)
        kept_paths.append(cp)
    return kept


def _format_bytes(n: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    f = float(n)
    for u in units:
        if f < 1024.0 or u == units[-1]:
            if u == "B":
                return f"{int(f)} {u}"
            return f"{f:.2f} {u}"
        f /= 1024.0
    return f"{int(n)} B"


def _delete_path(p: Path) -> None:
    if not p.exists():
        return
    if p.is_dir():
        shutil.rmtree(p)
    else:
        p.unlink()


def _plan_scene_scratch(
    scene_dir: Path,
    *,
    keep_confidence: bool,
    keep_glb: bool,
    remove_masks: bool,
) -> List[DeletionCandidate]:
    c: List[DeletionCandidate] = []
    for name, reason in (
        ("depth_maps", "Dust3R depth visualizations (reproducible from depth tensors)"),
        ("pointmaps", "Dust3R pointmap visualizations"),
    ):
        p = scene_dir / name
        if p.exists():
            c.append(DeletionCandidate(p, reason))

    if not keep_confidence:
        p = scene_dir / "confidence_map"
        if p.exists():
            c.append(
                DeletionCandidate(
                    p,
                    "confidence PNGs (only needed to re-train 3DGS with --use_confidence)",
                )
            )

    if not keep_glb:
        p = scene_dir / "scene.glb"
        if p.is_file():
            c.append(DeletionCandidate(p, "Dust3R scene GLB preview"))

    if remove_masks:
        p = scene_dir / "masks"
        if p.exists():
            c.append(
                DeletionCandidate(
                    p,
                    "per-frame masks (not used by COLMAP reader in gaussian-splatting)",
                )
            )
    return c


def _plan_output_dir(
    out_dir: Path,
    *,
    aggressive: bool,
    keep_iterations: int,
) -> List[DeletionCandidate]:
    c: List[DeletionCandidate] = []

    render_dir = out_dir / "render"
    if render_dir.exists():
        c.append(
            DeletionCandidate(
                render_dir,
                "3dgs.py preview renders / videos (train_renders, interpolation, camera_trajectory)",
            )
        )

    for p in out_dir.iterdir():
        if p.is_file() and _is_tensorboard_event_file(p):
            c.append(DeletionCandidate(p, "tensorboard event log"))

    for p in out_dir.glob("chkpnt*.pth"):
        c.append(DeletionCandidate(p, "optimizer state checkpoint (resume training only)"))

    if aggressive:
        inp = out_dir / "input.ply"
        if inp.is_file():
            c.append(
                DeletionCandidate(
                    inp,
                    "copy of initial sparse point cloud (duplicate of sparse/0/points3D.ply)",
                )
            )

    pairs = _point_cloud_iteration_dirs(out_dir)
    if aggressive and pairs:
        if keep_iterations <= 0:
            keep_iters: Set[int] = {pairs[-1][0]}
        else:
            keep_iters = set(it for it, _ in pairs[-keep_iterations:])
        for it, p in pairs:
            if it not in keep_iters:
                c.append(
                    DeletionCandidate(
                        p,
                        f"intermediate 3DGS PLY checkpoint (iteration_{it}; keeping {sorted(keep_iters)})",
                    )
                )

    return c


def _plan_duplicate_extracted_frames(data_root: Path) -> List[DeletionCandidate]:
    """
    get_frame.py writes instantsplat/data/images/<tag>/; dust3r_inference duplicates into
    data/scenes/<tag>/images/. If the scene folder exists, the top-level copy is redundant.
    """
    c: List[DeletionCandidate] = []
    images_root = data_root / "images"
    scenes_root = data_root / "scenes"
    if not images_root.is_dir() or not scenes_root.is_dir():
        return c
    for p in sorted(images_root.iterdir()):
        if not p.is_dir():
            continue
        if (scenes_root / p.name / "images").is_dir():
            c.append(
                DeletionCandidate(
                    p,
                    "duplicate extracted frames (same tag exists under data/scenes/.../images)",
                )
            )
    return c


def plan_all(
    data_root: Path,
    *,
    aggressive: bool,
    keep_confidence: bool,
    keep_glb: bool,
    remove_masks: bool,
    keep_iterations: int,
) -> List[DeletionCandidate]:
    data_root = data_root.resolve()
    all_c: List[DeletionCandidate] = []

    for scene_dir in _iter_scene_dirs(data_root):
        all_c.extend(
            _plan_scene_scratch(
                scene_dir,
                keep_confidence=keep_confidence,
                keep_glb=keep_glb,
                remove_masks=remove_masks,
            )
        )
        for child in scene_dir.iterdir():
            if _is_output_training_dir(child):
                all_c.extend(
                    _plan_output_dir(
                        child,
                        aggressive=aggressive,
                        keep_iterations=keep_iterations,
                    )
                )

    if aggressive:
        all_c.extend(_plan_duplicate_extracted_frames(data_root))

    return _dedupe_under_parent(all_c)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Clean up InstantSplat data/ outputs: keep COLMAP source + 3DGS files needed to "
            "render/reload, drop heavy redundant artifacts."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("instantsplat/data"),
        help="Path to instantsplat/data (contains scenes/ and usually images/).",
    )
    parser.add_argument(
        "--aggressive",
        action="store_true",
        help=(
            "Also remove duplicate data/images/<tag> when scenes/<tag>/images exists, "
            "drop input.ply in output_* and all but the last N point_cloud iterations "
            "(see --keep-iterations)."
        ),
    )
    parser.add_argument(
        "--keep-iterations",
        type=int,
        default=1,
        help=(
            "With --aggressive, how many latest point_cloud/iteration_* folders to keep per "
            "output_* dir (default: 1 = final only). Ignored if <=0 (keeps final only)."
        ),
    )
    parser.add_argument(
        "--keep-confidence",
        action="store_true",
        help="Keep data/scenes/*/confidence_map/ (needed only for re-training with --use_confidence).",
    )
    parser.add_argument(
        "--keep-glb",
        action="store_true",
        help="Keep data/scenes/*/scene.glb (Dust3R preview).",
    )
    parser.add_argument(
        "--remove-masks",
        action="store_true",
        help="Also delete data/scenes/*/masks/ (not used by standard COLMAP loader).",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete; default is dry-run.",
    )

    args = parser.parse_args()
    root = args.root.resolve()
    if not root.exists():
        raise SystemExit(f"Root does not exist: {root}")

    candidates = plan_all(
        root,
        aggressive=args.aggressive,
        keep_confidence=args.keep_confidence,
        keep_glb=args.keep_glb,
        remove_masks=args.remove_masks,
        keep_iterations=max(0, args.keep_iterations),
    )

    total_bytes = sum(_dir_size_bytes(c.path) for c in candidates)
    mode = "APPLY" if args.apply else "DRY-RUN"
    plan = "aggressive" if args.aggressive else "conservative"

    print(f"[{mode}] Root: {root}")
    print(f"[{mode}] Plan: {plan}")
    print(
        f"[{mode}] Candidates: {len(candidates)}  Estimated reclaim: {_format_bytes(total_bytes)}"
    )
    print("")
    print(
        "Kept (not listed): data/scenes/<tag>/sparse/0/{cameras.txt,images.txt,points3D.ply}, "
        "data/scenes/<tag>/images/, output_*/{cfg_args,cameras.json,point_cloud/...} "
        "(trimmed only in aggressive mode)."
    )
    print("")

    for c in candidates:
        size_b = _dir_size_bytes(c.path)
        try:
            rel = c.path.relative_to(root)
        except ValueError:
            rel = c.path
        print(f"- {_format_bytes(size_b):>10}  {rel}  ({c.reason})")

    if not args.apply:
        print("")
        print(f"[DRY-RUN] Would reclaim approximately {_format_bytes(total_bytes)}")
        print("Re-run with --apply to delete.")
        return 0

    deleted = 0
    for c in candidates:
        if c.path.exists():
            _delete_path(c.path)
            deleted += 1

    print("")
    print(f"[APPLY] Deleted {deleted} paths.")
    print(f"[APPLY] Estimated reclaim: {_format_bytes(total_bytes)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
