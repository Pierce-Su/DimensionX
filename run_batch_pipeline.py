#!/usr/bin/env python3
"""
Batch pipeline for DimensionX: run the image-to-video (3D scene) pipeline on a
curated dataset. Stage 1 uses the SAT-based 145-frame 360° orbit model
(sample_video_lowR.py) with main model at checkpoints/1/ and T5/VAE under sat_weights.

Features (aligned with Matrix-3D batch pipeline):
  - Stage-wise execution via --only_stages
  - Preserved file structure: output_base/photorealistic/index_XXXX/, stylized/...
  - Enhanced prompts from metadata (style, scene_type, category)
  - Optional index filtering and continue-on-error

Usage:
  python run_batch_pipeline.py --dataset_dir data/curated_set --output_base output/batch \\
      --sat_checkpoint_dir checkpoints --sat_t5_dir sat_weights/t5-v1_1-xxl \\
      --sat_vae_ckpt sat_weights/vae/3d-vae.pt --cogvideo_root cogvideo

  python run_batch_pipeline.py --dataset_dir data/curated_set --only_stages 1 --indices 0-9

  # Stage 2 only: run 3DGS (InstantSplat) on existing videos
  python run_batch_pipeline.py --dataset_dir data/curated_set --output_base output/batch \\
      --only_stages 2 --instantsplat_root instantsplat
"""

import argparse
import json
import re
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set


@dataclass
class ReconStageConfig:
    """Configuration for Stage 2: InstantSplat (Dust3R + 3DGS)."""
    device: str
    num_frames: int
    gs_iter: int
    lambda_lpips: float
    use_confidence: bool


def _parse_indices(specs: List[str]) -> Set[int]:
    """Parse --indices into a set of integers (single values or inclusive ranges)."""
    out: Set[int] = set()
    range_re = re.compile(r"^(-?\d+)[-:](-?\d+)$")
    for spec in specs:
        s = str(spec).strip()
        m = range_re.match(s)
        if m:
            lo, hi = int(m.group(1)), int(m.group(2))
            if lo <= hi:
                out.update(range(lo, hi + 1))
            else:
                out.update(range(hi, lo + 1))
        else:
            out.add(int(s))
    return out


def enhance_prompt_with_metadata(
    base_prompt: Optional[str],
    style: str,
    scene_type: str,
    category: str,
) -> Optional[str]:
    """Enhance prompt with style, scene_type, and category (Matrix-3D style)."""
    if not base_prompt:
        return None
    metadata_parts: List[str] = []
    if style and style != "unknown":
        metadata_parts.append(f"Style: {style.replace('_', ' ').title()}")
    if scene_type and scene_type != "unknown":
        metadata_parts.append(f"Scene type: {scene_type.replace('_', ' ').title()}")
    if category and category != "unknown":
        metadata_parts.append(f"Category: {category.replace('_', ' ').title()}")
    if metadata_parts:
        return ", ".join(metadata_parts) + ". " + base_prompt
    return base_prompt


def _ensure_checkpoint_latest(checkpoint_dir: Path) -> None:
    """
    SAT load_checkpoint expects checkpoint_dir/latest to exist (file containing
    the subdir name, e.g. "1"). If 1/mp_rank_00_model_states.pt exists but
    latest is missing, create latest with content "1".
    """
    checkpoint_dir = Path(checkpoint_dir).resolve()
    latest_file = checkpoint_dir / "latest"
    rank_dir = checkpoint_dir / "1"
    ckpt_file = rank_dir / "mp_rank_00_model_states.pt"
    if latest_file.exists():
        return
    if ckpt_file.exists():
        latest_file.write_text("1", encoding="utf-8")
        print(f"Created {latest_file} (content: 1) for SAT checkpoint loading.")


def _create_sat_patched_configs(
    cogvideo_root: Path,
    sat_t5_dir: Optional[str],
    sat_vae_ckpt: Optional[str],
    sat_checkpoint_dir: Optional[str],
) -> List[str]:
    """
    Create patched SAT config YAMLs under cogvideo_root/sat_batch_configs/ with
    the given paths. Return config paths relative to cogvideo_root.
    If no overrides are given, return default config paths (no patching).
    """
    if not sat_t5_dir and not sat_vae_ckpt and not sat_checkpoint_dir:
        return [
            "configs/cogvideox_5b_i2v_lora_145.yaml",
            "configs/inference_145.yaml",
        ]

    patch_dir = cogvideo_root / "sat_batch_configs"
    patch_dir.mkdir(parents=True, exist_ok=True)

    configs_src = [
        ("configs/cogvideox_5b_i2v_lora_145.yaml", "cogvideox_5b_i2v_lora_145.yaml"),
        ("configs/inference_145.yaml", "inference_145.yaml"),
    ]
    out_paths: List[str] = []

    for rel_src, name in configs_src:
        src_path = cogvideo_root / rel_src
        if not src_path.exists():
            raise FileNotFoundError(f"SAT config not found: {src_path}")
        text = src_path.read_text(encoding="utf-8")

        if name == "cogvideox_5b_i2v_lora_145.yaml":
            if sat_t5_dir is not None:
                text = re.sub(
                    r'model_dir:\s*["\'][^"\']*["\']',
                    f'model_dir: "{Path(sat_t5_dir).resolve()}"',
                    text,
                    count=1,
                )
            if sat_vae_ckpt is not None:
                text = re.sub(
                    r'ckpt_path:\s*["\'][^"\']*["\']',
                    f'ckpt_path: "{Path(sat_vae_ckpt).resolve()}"',
                    text,
                    count=1,
                )
        elif name == "inference_145.yaml":
            if sat_checkpoint_dir is not None:
                text = re.sub(
                    r'load:\s*["\'][^"\']*["\']',
                    f'load: "{Path(sat_checkpoint_dir).resolve()}"',
                    text,
                    count=1,
                )

        out_path = patch_dir / name
        out_path.write_text(text, encoding="utf-8")
        out_paths.append(f"sat_batch_configs/{name}")

    return out_paths


def run_sat_360_video_generation(
    image_path: Path,
    prompt: Optional[str],
    output_dir: Path,
    base_configs: List[str],
    seed: int,
    cogvideo_root: Path,
) -> bool:
    """
    Run the SAT 145-frame image-to-video pipeline for one image.
    Calls sample_video_lowR.py and copies the first generated .mp4 to output_dir/video.mp4.
    """
    if not cogvideo_root.exists():
        print(f"ERROR: cogvideo_root not found: {cogvideo_root}")
        return False

    sample_video_script = cogvideo_root / "sample_video_lowR.py"
    if not sample_video_script.exists():
        print(f"ERROR: sample_video_lowR.py not found at {sample_video_script}")
        return False

    output_dir_abs = output_dir.resolve()
    tmp_dir = output_dir_abs / "sat_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    input_file = tmp_dir / "i2v_input.txt"
    # Single line only: SAT reads one line per sample; newlines in prompt would cause extra diffusion runs and broken saves
    line_prompt = (prompt or "").replace("\r", " ").replace("\n", " ").strip()
    image_path_abs = image_path.resolve()
    input_file.write_text(f"{image_path_abs}@@{line_prompt}\n", encoding="utf-8")

    sat_output_dir = tmp_dir / "raw_outputs"
    sat_output_dir.mkdir(parents=True, exist_ok=True)

    cmd: List[str] = [
        sys.executable,
        sample_video_script.name,
        "--base",
    ]
    cmd.extend(base_configs)
    cmd.extend(
        [
            "--seed",
            str(seed),
            "--image2video",
            "--input-type",
            "txt",
            "--input-file",
            str(input_file.resolve()),
            "--output-dir",
            str(sat_output_dir.resolve()),
        ]
    )

    print(f"  Stage 1 (SAT 360): Generating video for {image_path.name}")
    try:
        from subprocess import CalledProcessError, run

        result = run(
            cmd,
            cwd=str(cogvideo_root),
            check=True,
            capture_output=False,
            text=True,
        )
        if result.returncode != 0:
            print(f"ERROR: SAT pipeline failed with code {result.returncode}")
            return False
    except CalledProcessError as e:
        print(f"ERROR: SAT pipeline failed with return code {e.returncode}")
        print("Run directly to debug:")
        print(f"  cd {cogvideo_root.resolve()} && {' '.join(cmd)}")
        return False
    except Exception as e:
        print(f"ERROR: SAT pipeline: {e}")
        return False

    sat_videos = list(sat_output_dir.rglob("*.mp4"))
    if not sat_videos:
        sat_videos = list(tmp_dir.rglob("*.mp4"))
    if not sat_videos:
        print(f"ERROR: No .mp4 produced under {sat_output_dir}")
        try:
            for p in sorted(sat_output_dir.rglob("*"))[:50]:
                print(f"  found: {p.relative_to(sat_output_dir)}")
        except Exception:
            pass
        return False

    final_video_path = output_dir / "video.mp4"
    try:
        shutil.copy2(sat_videos[0], final_video_path)
    except Exception as e:
        print(f"ERROR: Failed to copy video to {final_video_path}: {e}")
        return False

    print(f"Saved video to: {final_video_path}")
    return True


def _build_case_tag(index: int, variant: str) -> str:
    """Build a short, filesystem-friendly tag for InstantSplat (e.g. idx0005_photorealistic)."""
    return f"idx{index:04d}_{variant}"


def run_instantsplat_pipeline(
    video_path: Path,
    case_tag: str,
    instantsplat_root: Path,
    cfg: ReconStageConfig,
) -> bool:
    """
    Run the full InstantSplat pipeline (get_frame -> Dust3R -> 3DGS optimization)
    for a single video.
    """
    from subprocess import CalledProcessError, run

    if not video_path.exists():
        print(f"ERROR: Video not found for InstantSplat pipeline: {video_path}")
        return False

    print(f"  Stage 2: Running InstantSplat 3D reconstruction for tag: {case_tag}")

    # Use absolute paths so get_frame.py can find the video when run with cwd=instantsplat_root
    video_path_abs = video_path.resolve()
    images_dir = (instantsplat_root / "data" / "images" / case_tag).resolve()
    cmd_get_frame = [
        sys.executable,
        "get_frame.py",
        str(video_path_abs),
        str(images_dir),
        str(cfg.num_frames),
    ]
    cmd_dust3r = [
        sys.executable,
        "dust3r_inference.py",
        "--device",
        cfg.device,
        "--dataset",
        case_tag,
    ]
    cmd_3dgs: List[str] = [
        sys.executable,
        "3dgs.py",
        "--device",
        cfg.device,
        "--dataset",
        case_tag,
        "--iter",
        str(cfg.gs_iter),
        "--lambda_lpips",
        str(cfg.lambda_lpips),
    ]
    # Always export the final optimized Gaussian splats as a PLY when invoked
    # via the DimensionX batch pipeline so downstream consumers can use them.
    cmd_3dgs.append("--export_ply")
    if cfg.use_confidence:
        cmd_3dgs.append("--use_confidence")

    for cmd, desc in [
        (cmd_get_frame, "frame extraction"),
        (cmd_dust3r, "Dust3R reconstruction"),
        (cmd_3dgs, "3DGS optimization"),
    ]:
        print(f"    Running InstantSplat ({desc}): {' '.join(cmd)}")
        try:
            result = run(
                cmd,
                cwd=str(instantsplat_root),
                check=True,
                capture_output=False,
                text=True,
            )
            if result.returncode != 0:
                print(f"ERROR: InstantSplat step '{desc}' failed with code {result.returncode}")
                return False
        except CalledProcessError as e:
            print(f"ERROR: InstantSplat step '{desc}' failed with return code {e.returncode}")
            return False
        except Exception as e:
            print(f"ERROR: InstantSplat '{desc}': {e}")
            return False

    print(f"  Stage 2 done: 3DGS scene for {case_tag}")
    return True


def process_sample(
    sample: Dict,
    dataset_dir: Path,
    output_base: Path,
    only_stages: Optional[List[int]],
    base_configs: List[str],
    seed: int,
    cogvideo_root: Path,
    instantsplat_root: Optional[Path],
    recon_cfg: Optional[ReconStageConfig],
) -> Dict:
    """Process one sample: photorealistic and/or stylized variants."""
    index = sample["index"]
    style = sample.get("style", "unknown")
    scene_type = sample.get("scene_type", "unknown")
    category = sample.get("category", "unknown")

    results = {
        "index": index,
        "style": style,
        "scene_type": scene_type,
        "category": category,
        "processed": [],
        "failed": [],
    }

    skip_stages: List[int] = []
    if only_stages is not None:
        skip_stages = [s for s in [1, 2] if s not in only_stages]

    def run_for_variant(variant_key: str) -> None:
        if variant_key not in sample:
            return
        variant_data = sample[variant_key]
        image_path = dataset_dir / variant_key / variant_data["filename"]
        output_dir = output_base / variant_key / f"index_{index:04d}"
        video_path = output_dir / "video.mp4"

        # Stage 2 only: require existing video
        if only_stages == [2]:
            if not video_path.exists():
                print(f"WARNING: Video not found for Stage 2: {video_path}")
                results["failed"].append(f"{variant_key}: no video.mp4 (run Stage 1 first or provide video)")
                return
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            if not image_path.exists():
                print(f"WARNING: Image not found: {image_path}")
                results["failed"].append(f"{variant_key}: {variant_data['filename']} (file not found)")
                return
            output_dir.mkdir(parents=True, exist_ok=True)
            base_prompt = variant_data.get("prompt")
            prompt = enhance_prompt_with_metadata(base_prompt, style, scene_type, category)
            if prompt is not None:
                (output_dir / "prompt.txt").write_text(prompt, encoding="utf-8")

        case_tag = _build_case_tag(index, variant_key)

        # Stage 1: SAT 360° video generation
        if 1 not in skip_stages:
            ok = run_sat_360_video_generation(
                image_path=image_path,
                prompt=prompt,
                output_dir=output_dir,
                base_configs=base_configs,
                seed=seed,
                cogvideo_root=cogvideo_root,
            )
            if not ok:
                results["failed"].append(f"{variant_key}: {variant_data['filename']} (Stage 1 failed)")
                return
        else:
            print(f"  Skipping Stage 1 for {variant_key} index {index}")

        # Stage 2: InstantSplat (Dust3R + 3DGS)
        if 2 in skip_stages:
            print(f"  Skipping Stage 2 for {variant_key} index {index}")
        elif instantsplat_root is None or recon_cfg is None:
            print(f"  Skipping Stage 2 (instantsplat not configured) for {variant_key} index {index}")
        else:
            ok_3d = run_instantsplat_pipeline(
                video_path=video_path,
                case_tag=case_tag,
                instantsplat_root=instantsplat_root,
                cfg=recon_cfg,
            )
            if not ok_3d:
                results["failed"].append(f"{variant_key}: {variant_data['filename']} (Stage 2 failed)")
                return

        results["processed"].append(f"{variant_key}: {variant_data['filename']}")

    run_for_variant("photorealistic")
    run_for_variant("stylized")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run DimensionX image-to-video batch pipeline (SAT 360° Stage 1).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full batch with checkpoints/1 and sat_weights
  python run_batch_pipeline.py --dataset_dir data/curated_set --output_base output/batch \\
      --sat_checkpoint_dir checkpoints --sat_t5_dir sat_weights/t5-v1_1-xxl \\
      --sat_vae_ckpt sat_weights/vae/3d-vae.pt --cogvideo_root cogvideo

  # Only Stage 1, subset of indices
  python run_batch_pipeline.py --dataset_dir data/curated_set --only_stages 1 --indices 0-9

  # Only Stage 2 (3DGS from existing videos)
  python run_batch_pipeline.py --dataset_dir data/curated_set --output_base output/batch \\
      --only_stages 2 --instantsplat_root instantsplat

  # Continue on error
  python run_batch_pipeline.py --dataset_dir data/curated_set --output_base output/batch \\
      --sat_checkpoint_dir checkpoints --sat_t5_dir sat_weights/t5-v1_1-xxl \\
      --sat_vae_ckpt sat_weights/vae/3d-vae.pt --cogvideo_root cogvideo --continue_on_error
        """,
    )

    parser.add_argument("--dataset_dir", type=str, required=True,
                        help="Path to curated_set with metadata.json and image folders")
    parser.add_argument("--output_base", type=str, default="output/batch",
                        help="Base output dir: output_base/photorealistic/index_XXXX/ etc.")
    parser.add_argument("--indices", type=str, nargs="+", default=None,
                        help="Only process these indices (e.g. 0 5 14 or 0-24)")
    parser.add_argument("--only_stages", type=int, nargs="+", choices=[1, 2], default=None,
                        help="Only run these stages: 1=SAT 360° video, 2=InstantSplat 3DGS. Default: run stage 1. Use 2 to run 3DGS on existing videos.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for SAT pipeline")
    parser.add_argument("--sat_checkpoint_dir", type=str, default=None,
                        help="(Required for stage 1.) Path to main model checkpoint dir: must contain "
                        "1/mp_rank_00_model_states.pt and a 'latest' file (script creates 'latest' with content '1' if missing).")
    parser.add_argument("--sat_t5_dir", type=str, default=None,
                        help="Path to T5 text encoder (t5-v1_1-xxl), e.g. sat_weights/t5-v1_1-xxl")
    parser.add_argument("--sat_vae_ckpt", type=str, default=None,
                        help="Path to 3D VAE checkpoint, e.g. sat_weights/vae/3d-vae.pt")
    parser.add_argument("--cogvideo_root", type=str, default="cogvideo",
                        help="Path to cogvideo dir containing sample_video_lowR.py")
    # Stage 2: InstantSplat (Dust3R + 3DGS)
    parser.add_argument("--instantsplat_root", type=str, default=None,
                        help="Path to InstantSplat dir (required for stage 2). Enables 3DGS from video.")
    parser.add_argument("--num_frames", type=int, default=50,
                        help="Frames to extract per video for Dust3R (stage 2).")
    parser.add_argument("--gs_iter", type=int, default=10000,
                        help="3DGS optimization iterations (stage 2).")
    parser.add_argument("--lambda_lpips", type=float, default=0.3,
                        help="LPIPS weight for 3DGS loss (stage 2).")
    parser.add_argument("--use_confidence", action="store_true",
                        help="Use Dust3R confidence in 3DGS (stage 2).")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device for InstantSplat/Dust3R/3DGS (stage 2).")
    parser.add_argument("--continue_on_error", action="store_true",
                        help="Continue processing if one sample fails")

    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    metadata_path = dataset_dir / "metadata.json"

    if not dataset_dir.exists():
        print(f"ERROR: Dataset directory not found: {dataset_dir}")
        sys.exit(1)
    if not metadata_path.exists():
        print(f"ERROR: metadata.json not found: {metadata_path}")
        sys.exit(1)

    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    samples = metadata.get("samples", [])

    if args.indices is not None:
        idx_set = _parse_indices(args.indices)
        samples = [s for s in samples if s.get("index") in idx_set]
        print(f"Filtered to {len(samples)} samples (indices: {sorted(idx_set)})")

    if not samples:
        print("ERROR: No samples to process")
        sys.exit(1)

    output_base = Path(args.output_base)
    output_base.mkdir(parents=True, exist_ok=True)
    cogvideo_root = Path(args.cogvideo_root)

    stages_enabled = args.only_stages if args.only_stages is not None else [1]
    if 1 in stages_enabled and not cogvideo_root.exists():
        print(f"ERROR: cogvideo_root not found: {cogvideo_root}")
        sys.exit(1)
    if 1 in stages_enabled and not args.sat_checkpoint_dir:
        print(
            "ERROR: Stage 1 (SAT image-to-video) requires --sat_checkpoint_dir. "
            "Pass the directory that contains 1/mp_rank_00_model_states.pt (e.g. checkpoints). "
            "The repo's default config points to an author path that may not exist on your machine."
        )
        sys.exit(1)
    if 2 in stages_enabled and not args.instantsplat_root:
        print(
            "ERROR: Stage 2 (InstantSplat 3DGS) requires --instantsplat_root. "
            "Pass the path to the InstantSplat directory (e.g. instantsplat)."
        )
        sys.exit(1)

    if args.sat_checkpoint_dir:
        _ensure_checkpoint_latest(Path(args.sat_checkpoint_dir))

    instantsplat_root = Path(args.instantsplat_root) if args.instantsplat_root else None
    recon_cfg = None
    if instantsplat_root and 2 in stages_enabled:
        recon_cfg = ReconStageConfig(
            device=args.device,
            num_frames=args.num_frames,
            gs_iter=args.gs_iter,
            lambda_lpips=args.lambda_lpips,
            use_confidence=args.use_confidence,
        )
        if not instantsplat_root.exists():
            print(f"ERROR: instantsplat_root not found: {instantsplat_root}")
            sys.exit(1)
        # Require InstantSplat submodules (dust3r, gaussian-splatting) to be cloned
        dust3r_pkg = instantsplat_root / "dust3r" / "dust3r"
        dust3r_alt = instantsplat_root / "dust3r" / "inference.py"
        gs_scene = instantsplat_root / "gaussian-splatting" / "scene"
        if not (dust3r_pkg.is_dir() or dust3r_alt.exists()) or not gs_scene.is_dir():
            print(
                "ERROR: Stage 2 (InstantSplat) requires the dust3r and gaussian-splatting submodules to be initialized.\n"
                "From the InstantSplat directory, run:\n"
                "  cd instantsplat && git submodule update --init --recursive\n"
                "Then install any dependencies required by those submodules (e.g. dust3r, 3D Gaussian Splatting)."
            )
            sys.exit(1)

    if 1 in stages_enabled:
        base_configs = _create_sat_patched_configs(
            cogvideo_root,
            args.sat_t5_dir,
            args.sat_vae_ckpt,
            args.sat_checkpoint_dir,
        )
    else:
        base_configs = []

    print(f"\n{'='*80}")
    print("DimensionX Batch Pipeline Configuration")
    print(f"{'='*80}")
    print(f"Dataset: {dataset_dir}")
    print(f"Output base: {output_base}")
    print(f"Total samples: {len(samples)}")
    print(f"Stages: {stages_enabled}")
    print(f"CogVideo root: {cogvideo_root}")
    if args.sat_checkpoint_dir or args.sat_t5_dir or args.sat_vae_ckpt:
        print(f"SAT paths: checkpoint={args.sat_checkpoint_dir}, T5={args.sat_t5_dir}, VAE={args.sat_vae_ckpt}")
    print(f"{'='*80}\n")

    all_results: List[Dict] = []
    start_time = time.time()

    for i, sample in enumerate(samples, 1):
        index = sample["index"]
        print(f"\n{'='*80}")
        print(f"Processing sample {i}/{len(samples)}: Index {index}")
        print(f"{'='*80}")

        try:
            result = process_sample(
                sample=sample,
                dataset_dir=dataset_dir,
                output_base=output_base,
                only_stages=stages_enabled,
                base_configs=base_configs,
                seed=args.seed,
                cogvideo_root=cogvideo_root,
                instantsplat_root=instantsplat_root,
                recon_cfg=recon_cfg,
            )
            all_results.append(result)
            if result["processed"]:
                print(f"✓ Processed: {', '.join(result['processed'])}")
            if result["failed"]:
                print(f"✗ Failed: {', '.join(result['failed'])}")
                if not args.continue_on_error:
                    print("Stopping (use --continue_on_error to continue)")
                    break
        except Exception as e:
            print(f"ERROR processing sample {index}: {e}")
            all_results.append({
                "index": index,
                "style": sample.get("style", "unknown"),
                "scene_type": sample.get("scene_type", "unknown"),
                "category": sample.get("category", "unknown"),
                "processed": [],
                "failed": [f"exception: {e}"],
            })
            if not args.continue_on_error:
                break

    elapsed = time.time() - start_time
    total_processed = sum(len(r["processed"]) for r in all_results)
    total_failed = sum(len(r["failed"]) for r in all_results)
    failed_results = [r for r in all_results if r["failed"]]
    failed_indices = sorted({r["index"] for r in failed_results})
    failed_details = [{"index": r["index"], "failed": r["failed"]} for r in failed_results]

    print(f"\n{'='*80}")
    print("Batch Processing Summary")
    print(f"{'='*80}")
    print(f"Total samples: {len(all_results)}")
    print(f"Successfully processed: {total_processed}")
    print(f"Failed: {total_failed}")
    if failed_indices:
        print(f"Failed indices: {', '.join(str(i) for i in failed_indices)}")
    print(f"Time: {elapsed/60:.1f} min ({elapsed:.1f} s)")
    print(f"{'='*80}\n")

    summary_path = output_base / "batch_summary.json"
    summary = {
        "total_samples": len(all_results),
        "total_processed": total_processed,
        "total_failed": total_failed,
        "failed_sample_indices": failed_indices,
        "failed_details": failed_details,
        "elapsed_time_seconds": elapsed,
        "configuration": {
            "only_stages": stages_enabled,
            "seed": args.seed,
            "sat_checkpoint_dir": args.sat_checkpoint_dir,
            "sat_t5_dir": args.sat_t5_dir,
            "sat_vae_ckpt": args.sat_vae_ckpt,
            "cogvideo_root": str(cogvideo_root),
            "instantsplat_root": str(instantsplat_root) if instantsplat_root else None,
            "dataset_dir": str(dataset_dir),
            "output_base": str(output_base),
        },
        "results": all_results,
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {summary_path}")
    sys.exit(0 if total_failed == 0 else 1)


if __name__ == "__main__":
    main()
