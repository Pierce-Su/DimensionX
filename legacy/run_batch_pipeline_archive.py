#!/usr/bin/env python3
"""
Batch pipeline for DimensionX: process a curated dataset of single images
into 3D scenes using:

  1. DimensionX video generation (CogVideoX + S-Director LoRA)
  2. 3D scene reconstruction via Dust3R + InstantSplat (Gaussian Splatting)

Key features (mirrors Matrix-3D batch pipeline behavior):
  - Reads enhanced prompts and metadata from dataset `metadata.json`
  - Supports stage-wise execution via --only_stages
  - Processes both photorealistic and stylized variants when present
  - Organizes outputs in a dataset-like structure:

        output_base/
            photorealistic/
                index_0000/
                index_0005/
                ...
            stylized/
                index_0005/
                index_0017/
                ...

Usage examples:
  python run_batch_pipeline.py --dataset_dir data/curated_set \\
      --output_base output/dimensionx_batch \\
      --lora_path path/to/orbit_lora.safetensors

  # Only run 3D reconstruction (Stage 2) for samples whose videos already exist
  python run_batch_pipeline.py --dataset_dir data/curated_set \\
      --output_base output/dimensionx_batch \\
      --only_stages 2
"""

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set

import shutil

import torch
from diffusers import CogVideoXImageToVideoPipeline
from diffusers.utils import export_to_video, load_image


def _parse_indices(specs: List[str]) -> Set[int]:
    """Parse --indices specs into a set of integers (single values or inclusive ranges)."""
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
    """
    Enhance the base prompt using metadata from the dataset (style, scene_type, category).

    Mirrors Matrix-3D behavior so that the same curated metadata can be reused.
    """
    if not base_prompt:
        return None

    metadata_parts: List[str] = []

    if style and style != "unknown":
        style_formatted = style.replace("_", " ").title()
        metadata_parts.append(f"Style: {style_formatted}")

    if scene_type and scene_type != "unknown":
        scene_formatted = scene_type.replace("_", " ").title()
        metadata_parts.append(f"Scene type: {scene_formatted}")

    if category and category != "unknown":
        category_formatted = category.replace("_", " ").title()
        metadata_parts.append(f"Category: {category_formatted}")

    if metadata_parts:
        metadata_prefix = ", ".join(metadata_parts)
        return f"{metadata_prefix}. {base_prompt}"
    return base_prompt


@dataclass
class VideoStageConfig:
    lora_path: Optional[str]
    device: str
    fps: int
    lora_rank: int = 256


@dataclass
class ReconStageConfig:
    device: str
    num_frames: int
    gs_iter: int
    lambda_lpips: float
    use_confidence: bool


class DimensionXVideoGenerator:
    """
    Thin wrapper around CogVideoXImageToVideoPipeline so we only load the model once.
    """

    def __init__(self, cfg: VideoStageConfig) -> None:
        if cfg.lora_path is None:
            raise ValueError("lora_path must be provided when Stage 1 is enabled.")

        self.cfg = cfg
        print("Loading DimensionX CogVideoXImageToVideoPipeline...")
        self.pipe = CogVideoXImageToVideoPipeline.from_pretrained(
            "THUDM/CogVideoX-5b-I2V", torch_dtype=torch.bfloat16
        )

        lora_directory = os.path.dirname(cfg.lora_path)
        lora_filename = os.path.basename(cfg.lora_path)
        print(f"Loading LoRA weights from {cfg.lora_path} ...")
        self.pipe.load_lora_weights(
            lora_directory,
            weight_name=lora_filename,
            adapter_name="dimensionx_lora",
        )
        self.pipe.fuse_lora(lora_scale=1.0 / cfg.lora_rank)
        self.pipe.to(cfg.device)

    def generate_video(
    self,
        image_path: Path,
        prompt: Optional[str],
        output_video_path: Path,
    ) -> bool:
        try:
            print(f"Stage 1: Generating video for image: {image_path}")
            image = load_image(str(image_path))
            # If prompt is None, CogVideoX will still run but with a default internal behavior.
            text_prompt = prompt or ""
            result = self.pipe(image, text_prompt, use_dynamic_cfg=True)
            frames = result.frames[0]

            output_video_path.parent.mkdir(parents=True, exist_ok=True)
            export_to_video(frames, str(output_video_path), fps=self.cfg.fps)
            print(f"Saved video to: {output_video_path}")
            return True
        except Exception as e:
            print(f"ERROR during DimensionX video generation: {e}")
            return False


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

    print(f"Stage 2: Running InstantSplat 3D reconstruction for tag: {case_tag}")

    # 1) Extract frames from the video
    images_dir = instantsplat_root / "data" / "images" / case_tag
    cmd_get_frame = [
        sys.executable,
        "get_frame.py",
        str(video_path),
        str(images_dir),
        str(cfg.num_frames),
    ]

    # 2) Dust3R camera + point cloud estimation
    cmd_dust3r = [
        sys.executable,
        "dust3r_inference.py",
        "--device",
        cfg.device,
        "--dataset",
        case_tag,
    ]

    # 3) 3D Gaussian Splatting optimization
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
    if cfg.use_confidence:
        cmd_3dgs.append("--use_confidence")

    for cmd, desc in [
        (cmd_get_frame, "frame extraction"),
        (cmd_dust3r, "Dust3R reconstruction"),
        (cmd_3dgs, "3DGS optimization"),
    ]:
        print(f"\nRunning InstantSplat step ({desc}): {' '.join(cmd)}")
        try:
            result = run(
                cmd,
                cwd=instantsplat_root,
                check=True,
                capture_output=False,
                text=True,
            )
            if result.returncode != 0:
                print(f"ERROR: Command failed (step={desc}) with code {result.returncode}")
                return False
        except CalledProcessError as e:
            print(
                f"ERROR: InstantSplat step '{desc}' failed with return code "
                f"{e.returncode}"
            )
            return False
        except Exception as e:  # noqa: BLE001
            print(f"ERROR: Unexpected error during '{desc}': {e}")
            return False

    print("InstantSplat pipeline completed successfully.")
    return True


def _build_case_tag(index: int, variant: str) -> str:
    """
    Build a short, filesystem-friendly tag for InstantSplat datasets.

    Example: index 5, variant "photorealistic" -> "idx0005_photorealistic"
    """
    return f"idx{index:04d}_{variant}"


# Default paths hardcoded in the repo's SAT configs (author's machine)
_SAT_DEFAULT_T5_DIR = "/home/chenshuo/DimensionX_code/CogVideoX-2b-sat/t5-v1_1-xxl"
_SAT_DEFAULT_VAE_CKPT = "/home/chenshuo/DimensionX_code/CogVideoX-2b-sat/vae/3d-vae.pt"
_SAT_DEFAULT_CHECKPOINT_DIR = "/home/chenshuo/DimensionX_code/checkpoints/"


@dataclass
class SatBackendConfig:
    """
    Configuration for the SAT-based 145-frame 360-degree orbit pipeline.

    This wraps the existing `cogvideo/sample_video_lowR.py` + 145-frame configs.
    """

    seed: int
    base_configs: List[str]


def _create_sat_patched_configs(
    cogvideo_root: Path,
    t5_dir: Optional[str],
    vae_ckpt: Optional[str],
    checkpoint_dir: Optional[str],
) -> List[str]:
    """
    Create patched SAT config YAMLs under cogvideo_root/sat_batch_configs/ with
    the given paths, and return config paths relative to cogvideo_root.
    If all overrides are None, returns the default config paths (no patching).
    """
    if not t5_dir and not vae_ckpt and not checkpoint_dir:
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
        if t5_dir:
            text = text.replace(_SAT_DEFAULT_T5_DIR, str(Path(t5_dir).resolve()))
        if vae_ckpt:
            text = text.replace(_SAT_DEFAULT_VAE_CKPT, str(Path(vae_ckpt).resolve()))
        if checkpoint_dir:
            text = text.replace(_SAT_DEFAULT_CHECKPOINT_DIR, str(Path(checkpoint_dir).resolve()))
        out_path = patch_dir / name
        out_path.write_text(text, encoding="utf-8")
        out_paths.append(f"sat_batch_configs/{name}")
    return out_paths


def run_sat_360_video_generation(
    image_path: Path,
    prompt: Optional[str],
    output_dir: Path,
    cfg: SatBackendConfig,
    cogvideo_root: Path,
) -> bool:
    """
    Run the SAT-based 145-frame 360-degree orbit pipeline for a single image.

    This calls:
        python sample_video_lowR.py --base <config1> <config2> --seed <seed> --image2video
            --input-type txt --input-file <tmp> --output-dir <tmp_out>

    and then moves the first generated .mp4 into `output_dir / "video.mp4"`.
    """
    from subprocess import CalledProcessError, run

    if not cogvideo_root.exists():
        print(f"ERROR: cogvideo_root not found: {cogvideo_root}")
        return False

    sample_video_script = cogvideo_root / "sample_video_lowR.py"
    if not sample_video_script.exists():
        print(f"ERROR: sample_video_lowR.py not found at {sample_video_script}")
        return False

    # Prepare temporary input file: "image_path@@prompt"
    # Use absolute paths so the SAT subprocess (run with cwd=cogvideo_root) finds the file and image.
    output_dir_abs = output_dir.resolve()
    tmp_dir = output_dir_abs / "sat_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    input_file = tmp_dir / "i2v_input.txt"
    line_prompt = prompt or ""
    image_path_abs = image_path.resolve()
    input_file.write_text(f"{image_path_abs}@@{line_prompt}\n", encoding="utf-8")

    # Where SAT pipeline will dump its outputs
    sat_output_dir = tmp_dir / "raw_outputs"
    sat_output_dir.mkdir(parents=True, exist_ok=True)

    # When we set cwd=cogvideo_root below, pass absolute paths so the subprocess finds input and writes output correctly.
    script_name = sample_video_script.name
    cmd: List[str] = [
        sys.executable,
        script_name,
        "--base",
    ]
    cmd.extend(cfg.base_configs)
    cmd.extend(
        [
            "--seed",
            str(cfg.seed),
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
    print(f"\nRunning SAT 145-frame pipeline: {' '.join(cmd)}")
    try:
        result = run(
            cmd,
            cwd=cogvideo_root,
            check=True,
            capture_output=False,
            text=True,
        )
        if result.returncode != 0:
            print(f"ERROR: SAT pipeline failed with code {result.returncode}")
            return False
    except CalledProcessError as e:
        print(f"ERROR: SAT pipeline failed with return code {e.returncode}")
        print("Run the SAT script directly to see the full traceback (e.g. OOM or CUDA error):")
        print(f"  cd {cogvideo_root.resolve()} && {' '.join(cmd)}")
        return False
    except Exception as e:  # noqa: BLE001
        print(f"ERROR: Unexpected error during SAT pipeline: {e}")
        return False

    # Find the first .mp4 in the SAT output directory and move it to video.mp4
    sat_videos = list(sat_output_dir.rglob("*.mp4"))
    if not sat_videos:
        print(f"ERROR: No .mp4 files produced by SAT pipeline under {sat_output_dir}")
        print("The subprocess may have crashed (e.g. OOM) or been killed before writing. Run it directly to debug:")
        print(f"  cd {cogvideo_root.resolve()} && {' '.join(cmd)}")
        return False

    sat_video_path = sat_videos[0]
    final_video_path = output_dir / "video.mp4"
    try:
        shutil.copy2(sat_video_path, final_video_path)
    except Exception as e:  # noqa: BLE001
        print(f"ERROR: Failed to copy SAT video to {final_video_path}: {e}")
        return False

    print(f"Saved SAT 145-frame video to: {final_video_path}")
    return True


def process_sample(
    sample: Dict,
    dataset_dir: Path,
    output_base: Path,
    only_stages: Optional[List[int]],
    video_gen: Optional[DimensionXVideoGenerator],
    video_backend: str,
    sat_cfg: Optional[SatBackendConfig],
    cogvideo_root: Path,
    recon_cfg: ReconStageConfig,
    instantsplat_root: Path,
) -> Dict:
    """
    Process a single sample from metadata.json for both photorealistic
    and stylized variants (if present).
    """
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

    # Determine which stages are skipped
    skip_stages: List[int] = []
    if only_stages is not None:
        skip_stages = [s for s in [1, 2] if s not in only_stages]

    def _run_for_variant(variant_key: str) -> None:
        if variant_key not in sample:
            return

        variant_data = sample[variant_key]
        image_path = dataset_dir / variant_key / variant_data["filename"]
        if not image_path.exists():
            print(f"WARNING: Image not found: {image_path}")
            results["failed"].append(f"{variant_key}: {variant_data['filename']} (file not found)")
            return

        output_dir = output_base / variant_key / f"index_{index:04d}"
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"  [{variant_key}] Image: {variant_data['filename']} -> {output_dir}")

        base_prompt = variant_data.get("prompt")
        prompt = enhance_prompt_with_metadata(base_prompt, style, scene_type, category)

        # Save the enhanced prompt for reference
        if prompt is not None:
            prompt_file = output_dir / "prompt.txt"
            prompt_file.write_text(prompt, encoding="utf-8")

        # Build a simple case tag used by InstantSplat
        case_tag = _build_case_tag(index, variant_key)

        # Stage 1: DimensionX video generation
        video_path = output_dir / "video.mp4"
        if 1 in skip_stages:
            print(f"  Skipping Stage 1 (video generation) for {variant_key} index {index}")
        else:
            print(f"  Stage 1: Generating video from {variant_data['filename']} ...")
            if video_backend == "diffusers":
                if video_gen is None:
                    raise RuntimeError(
                        "Stage 1 is enabled with backend='diffusers' but video generator is not initialized."
                    )
                ok_video = video_gen.generate_video(
                    image_path=image_path,
                    prompt=prompt,
                    output_video_path=video_path,
                )
            elif video_backend == "sat_360":
                if sat_cfg is None:
                    raise RuntimeError(
                        "Stage 1 is enabled with backend='sat_360' but SAT config is not initialized."
                    )
                ok_video = run_sat_360_video_generation(
                    image_path=image_path,
                    prompt=prompt,
                    output_dir=output_dir,
                    cfg=sat_cfg,
                    cogvideo_root=cogvideo_root,
                )
            else:
                raise ValueError(f"Unknown video backend: {video_backend}")

            if not ok_video:
                results["failed"].append(f"{variant_key}: {variant_data['filename']} (Stage 1 failed)")
                return
            print(f"  Stage 1 done: video -> {video_path}")

        # Stage 2: InstantSplat 3D reconstruction
        if 2 in skip_stages:
            print(f"Skipping Stage 2 (3D reconstruction) for {variant_key} index {index}")
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

    # Run for both variants if present
    _run_for_variant("photorealistic")
    _run_for_variant("stylized")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run DimensionX pipeline on a batch dataset (single-image → video → 3D scene).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all samples through both stages
  python run_batch_pipeline.py --dataset_dir data/curated_set \\
      --output_base output/dimensionx_batch \\
      --lora_path path/to/orbit_lora.safetensors

  # Only run Stage 2 (3D reconstruction) assuming videos already exist
  python run_batch_pipeline.py --dataset_dir data/curated_set \\
      --output_base output/dimensionx_batch \\
      --only_stages 2

  # Process a subset of indices
  python run_batch_pipeline.py --dataset_dir data/curated_set \\
      --output_base output/dimensionx_batch \\
      --lora_path path/to/orbit_lora.safetensors \\
      --indices 0-9 20 30-32
        """,
    )

    # Dataset arguments
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Path to curated dataset directory containing metadata.json and image folders",
    )
    parser.add_argument(
        "--output_base",
        type=str,
        default="output/dimensionx_batch",
        help="Base output directory. Output structure: "
        "output_base/photorealistic/index_XXXX/ and output_base/stylized/index_XXXX/",
    )
    parser.add_argument(
        "--indices",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Only process these sample indices. Each item can be an integer "
            "(e.g. 0, 5) or an inclusive range (e.g. 0-24 or 0:24). "
            "Examples: --indices 0 2 5, --indices 0-9, --indices 0-24 30-39"
        ),
    )

    # Stage filtering
    parser.add_argument(
        "--only_stages",
        type=int,
        nargs="+",
        choices=[1, 2],
        default=None,
        help="Only run these stages (e.g., --only_stages 1 2). If not specified, runs both stages.",
    )

    # Stage 1: Video generation
    g1 = parser.add_argument_group("Stage 1: Video Generation")
    g1.add_argument(
        "--video_backend",
        type=str,
        choices=["diffusers", "sat_360"],
        default="diffusers",
        help=(
            "Video backend to use for Stage 1. "
            "'diffusers' uses CogVideoXImageToVideoPipeline + LoRA (short ~48-frame videos). "
            "'sat_360' uses the SAT-based 145-frame 360-degree orbit pipeline (sample_video_lowR.py)."
        ),
    )
    g1.add_argument(
        "--lora_path",
        type=str,
        default=None,
        help=(
            "Full path to the S-Director LoRA .safetensors file used for camera control "
            "when --video_backend=diffusers."
        ),
    )
    g1.add_argument(
        "--fps",
        type=int,
        default=8,
        help="Frames per second when exporting the video.",
    )
    g1.add_argument(
        "--sat_seed",
        type=int,
        default=42,
        help="Random seed for the SAT-based 145-frame pipeline when --video_backend=sat_360.",
    )
    g1.add_argument(
        "--sat_t5_dir",
        type=str,
        default=None,
        help="Path to T5 text encoder dir (t5-v1_1-xxl). Overrides hardcoded path in SAT configs for --video_backend=sat_360.",
    )
    g1.add_argument(
        "--sat_vae_ckpt",
        type=str,
        default=None,
        help="Path to CogVideoX 3D VAE checkpoint (3d-vae.pt). Overrides hardcoded path in SAT configs for --video_backend=sat_360.",
    )
    g1.add_argument(
        "--sat_checkpoint_dir",
        type=str,
        default=None,
        help="Path to DimensionX_360orbit checkpoint dir (containing 1/mp_rank_00_model_states.pt). Overrides hardcoded path in SAT configs for --video_backend=sat_360.",
    )

    # Stage 2: 3D reconstruction via InstantSplat
    g2 = parser.add_argument_group("Stage 2: 3D Reconstruction (Dust3R + InstantSplat)")
    g2.add_argument(
        "--num_frames",
        type=int,
        default=50,
        help="Number of frames to sample from each video for Dust3R.",
    )
    g2.add_argument(
        "--gs_iter",
        type=int,
        default=10000,
        help="Number of optimization iterations for 3DGS (InstantSplat).",
    )
    g2.add_argument(
        "--lambda_lpips",
        type=float,
        default=0.3,
        help="LPIPS weight for 3DGS optimization loss.",
    )
    g2.add_argument(
        "--use_confidence",
        action="store_true",
        help="Use Dust3R confidence maps during 3DGS optimization.",
    )

    # General settings
    g3 = parser.add_argument_group("General")
    g3.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="CUDA device to use for both DimensionX and InstantSplat stages.",
    )
    g3.add_argument(
        "--instantsplat_root",
        type=str,
        default="instantsplat",
        help="Path to the InstantSplat directory (relative to DimensionX root by default).",
    )
    g3.add_argument(
        "--cogvideo_root",
        type=str,
        default="cogvideo",
        help="Path to the CogVideoX SAT directory containing sample_video_lowR.py (for --video_backend=sat_360).",
    )
    g3.add_argument(
        "--continue_on_error",
        action="store_true",
        help="Continue processing other samples if one fails.",
    )

    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    metadata_path = dataset_dir / "metadata.json"

    if not dataset_dir.exists():
        print(f"ERROR: Dataset directory not found: {dataset_dir}")
        sys.exit(1)

    if not metadata_path.exists():
        print(f"ERROR: metadata.json not found: {metadata_path}")
        sys.exit(1)

    print(f"Loading metadata from {metadata_path}...")
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

    instantsplat_root = Path(args.instantsplat_root)
    if not instantsplat_root.exists():
        print(
            f"WARNING: InstantSplat directory not found at {instantsplat_root}. "
            "Stage 2 will fail unless this path is corrected."
        )

    cogvideo_root = Path(args.cogvideo_root)

    # Determine which stages are enabled
    stages_enabled = args.only_stages or [1, 2]
    stage1_enabled = 1 in stages_enabled
    stage2_enabled = 2 in stages_enabled

    # Prepare Stage 1 config + video generator / SAT config (if needed)
    video_gen: Optional[DimensionXVideoGenerator] = None
    sat_cfg: Optional[SatBackendConfig] = None
    video_backend = args.video_backend
    if stage1_enabled:
        if video_backend == "diffusers":
            if not args.lora_path:
                print(
                    "ERROR: --lora_path must be provided when Stage 1 is enabled "
                    "with --video_backend=diffusers."
                )
                sys.exit(1)
            video_cfg = VideoStageConfig(
                lora_path=args.lora_path,
                device=args.device,
                fps=args.fps,
            )
            video_gen = DimensionXVideoGenerator(video_cfg)
        elif video_backend == "sat_360":
            # SAT backend: use patched configs if user provided path overrides
            base_configs = _create_sat_patched_configs(
                cogvideo_root,
                args.sat_t5_dir,
                args.sat_vae_ckpt,
                args.sat_checkpoint_dir,
            )
            sat_cfg = SatBackendConfig(seed=args.sat_seed, base_configs=base_configs)
        else:
            print(f"ERROR: Unknown video backend: {video_backend}")
            sys.exit(1)

    # Prepare Stage 2 config
    recon_cfg = ReconStageConfig(
        device=args.device,
        num_frames=args.num_frames,
        gs_iter=args.gs_iter,
        lambda_lpips=args.lambda_lpips,
        use_confidence=args.use_confidence,
    )

    print(f"\n{'=' * 80}")
    print("DimensionX Batch Pipeline Configuration")
    print(f"{'=' * 80}")
    print(f"Dataset: {dataset_dir}")
    print(f"Output base: {output_base}")
    print(f"Total samples to process: {len(samples)}")
    print(f"Enabled stages: {sorted(stages_enabled)}")
    print(f"Device: {args.device}")
    print(f"Video backend (Stage 1): {video_backend}")
    if stage1_enabled and video_backend == "diffusers":
        print(f"Stage 1 LoRA: {args.lora_path}")
    if stage1_enabled and video_backend == "sat_360":
        print(f"CogVideo root: {cogvideo_root}")
        if args.sat_t5_dir or args.sat_vae_ckpt or args.sat_checkpoint_dir:
            print(f"SAT path overrides: T5={args.sat_t5_dir}, VAE={args.sat_vae_ckpt}, checkpoint={args.sat_checkpoint_dir}")
    print(f"InstantSplat root: {instantsplat_root}")
    print(f"{'=' * 80}\n")

    all_results: List[Dict] = []
    start_time = time.time()

    for i, sample in enumerate(samples, 1):
        index = sample["index"]
        print(f"\n{'=' * 80}")
        print(f"Processing sample {i}/{len(samples)}: Index {index}")
        print(f"{'=' * 80}")

        try:
            result = process_sample(
                sample=sample,
                dataset_dir=dataset_dir,
                output_base=output_base,
                only_stages=stages_enabled,
                video_gen=video_gen,
                video_backend=video_backend,
                sat_cfg=sat_cfg,
                cogvideo_root=cogvideo_root,
                recon_cfg=recon_cfg,
                instantsplat_root=instantsplat_root,
            )
            all_results.append(result)

            if result["processed"]:
                print(f"✓ Successfully processed: {', '.join(result['processed'])}")
            if result["failed"]:
                print(f"✗ Failed: {', '.join(result['failed'])}")
                if not args.continue_on_error:
                    print("Stopping due to error (use --continue_on_error to continue)")
                    break
        except Exception as e:  # noqa: BLE001
            print(f"ERROR processing sample {index}: {e}")
            all_results.append(
                {
                    "index": index,
                    "style": sample.get("style", "unknown"),
                    "scene_type": sample.get("scene_type", "unknown"),
                    "category": sample.get("category", "unknown"),
                    "processed": [],
                    "failed": [f"exception: {e}"],
                }
            )
            if not args.continue_on_error:
                print("Stopping due to error (use --continue_on_error to continue)")
                break

    # Summary
    elapsed = time.time() - start_time
    total_processed = sum(len(r["processed"]) for r in all_results)
    total_failed = sum(len(r["failed"]) for r in all_results)
    failed_results = [r for r in all_results if r["failed"]]
    failed_indices = sorted({r["index"] for r in failed_results})
    failed_details = [{"index": r["index"], "failed": r["failed"]} for r in failed_results]

    print(f"\n{'=' * 80}")
    print("Batch Processing Summary")
    print(f"{'=' * 80}")
    print(f"Total samples processed: {len(all_results)}")
    print(f"Successfully processed variants: {total_processed}")
    print(f"Failed variants: {total_failed}")
    if failed_indices:
        print(f"Failed sample indices: {', '.join(str(i) for i in failed_indices)}")
        for r in failed_results:
            print(f"  Index {r['index']}: {', '.join(r['failed'])}")
    print(f"Total time: {elapsed / 60:.1f} minutes ({elapsed:.1f} seconds)")
    print(f"{'=' * 80}\n")

    # Save summary JSON
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
            "device": args.device,
            "output_base": str(output_base),
            "dataset_dir": str(dataset_dir),
            "instantsplat_root": str(instantsplat_root),
            "video": {
                "backend": video_backend,
                "lora_path": args.lora_path,
                "fps": args.fps,
                "sat_seed": args.sat_seed,
                "sat_t5_dir": args.sat_t5_dir,
                "sat_vae_ckpt": args.sat_vae_ckpt,
                "sat_checkpoint_dir": args.sat_checkpoint_dir,
                "cogvideo_root": str(cogvideo_root),
            },
            "reconstruction": {
                "num_frames": args.num_frames,
                "gs_iter": args.gs_iter,
                "lambda_lpips": args.lambda_lpips,
                "use_confidence": args.use_confidence,
            },
        },
        "results": all_results,
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Summary saved to: {summary_path}")
    sys.exit(0 if total_failed == 0 else 1)


if __name__ == "__main__":
    main()

