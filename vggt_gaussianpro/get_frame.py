import cv2
import os
import argparse
import numpy as np


def extract_frames_from_video(video_path, output_dir, num_frames=None):
    """
    Extract evenly-spaced frames from a video and save them as 0.png, 1.png, …

    Args:
        video_path (str): Path to the input video file.
        output_dir (str): Directory where extracted frames are saved.
        num_frames (int | None): How many frames to extract.
            When None (default) every frame in the video is extracted.
    """
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        raise RuntimeError(f"Video reports 0 frames: {video_path}")

    if num_frames is None or num_frames >= total_frames:
        frame_indices = set(range(total_frames))
    else:
        frame_indices = set(np.linspace(0, total_frames - 1, num_frames, dtype=int).tolist())

    frame_count = 0
    extracted_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count in frame_indices:
            out_path = os.path.join(output_dir, f"{extracted_count}.png")
            cv2.imwrite(out_path, frame)
            print(f"  frame {frame_count:>5d} → {out_path}")
            extracted_count += 1

        frame_count += 1

    cap.release()
    print(f"Extracted {extracted_count} / {total_frames} frames → {output_dir}")
    return extracted_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from a video file.")
    parser.add_argument("video_path", type=str, help="Path to the input video file.")
    parser.add_argument("output_dir", type=str, help="Directory where extracted frames will be saved.")
    parser.add_argument(
        "num_frames",
        type=int,
        nargs="?",
        default=None,
        help="Number of frames to sample (default: all frames).",
    )
    args = parser.parse_args()
    extract_frames_from_video(args.video_path, args.output_dir, args.num_frames)
