#!/usr/bin/env python3
"""
CLI script: Video egg segmentation and persistent multi-object tracking (Task 4).
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from eggcv.common.io import load_yaml
from eggcv.common.logging_utils import get_logger
from eggcv.segmentation.classical_segmenter import ClassicalEggSegmenter
from eggcv.tracking.tracker import EggTracker
from eggcv.tracking.video_pipeline import VideoTrackingPipeline

logger = get_logger("track_video")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run egg segmentation and tracking pipeline on video (Task 4).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "tracking.yaml"),
        help="Path to video tracking configuration YAML",
    )
    parser.add_argument(
        "--video-path",
        type=str,
        default=None,
        help="Path to input video file (overrides tracking config)",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Path to save annotated output video MP4",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Limit number of video frames to process (useful for smoke testing)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_yaml(args.config) if Path(args.config).is_file() else {}

    video_in = args.video_path or PROJECT_ROOT / cfg.get("video", {}).get("local_video_path", "data/raw/video/egg_video.mp4")
    video_out = args.output_path or PROJECT_ROOT / cfg.get("outputs", {}).get("output_annotated_video", "outputs/tracking/annotated_egg_video.mp4")

    video_p = Path(video_in).resolve()
    if not video_p.is_file():
        yt_url = cfg.get("video", {}).get("assigned_youtube_url", "https://www.youtube.com/watch?v=Y4JyV94y2_A")
        print("\n" + "!" * 65)
        print("VIDEO FILE REQUIRED FOR TASK 4")
        print("!" * 65)
        print(f"Target video file not found at: {video_p}")
        print(f"Please download the assigned video from:")
        print(f"  {yt_url}")
        print(f"and place it at:")
        print(f"  {video_p}")
        print("or specify its location using: --video-path <path_to_video.mp4>")
        print("!" * 65 + "\n")
        sys.exit(1)

    # Initialize classical segmenter with configured parameters
    seg_cfg_path = PROJECT_ROOT / cfg.get("detection", {}).get("config_path", "configs/segmentation.yaml")
    seg_cfg = load_yaml(seg_cfg_path) if seg_cfg_path.is_file() else {}
    segmenter = ClassicalEggSegmenter(seg_cfg)

    # Initialize tracker
    trk_cfg = cfg.get("tracker", {})
    tracker = EggTracker(
        max_distance=float(trk_cfg.get("max_distance_pixels", 75.0)),
        max_consecutive_lost=int(trk_cfg.get("max_consecutive_lost_frames", 20)),
        min_hits_to_confirm=int(trk_cfg.get("min_hits_to_confirm", 2)),
    )

    pipeline = VideoTrackingPipeline(segmenter=segmenter, tracker=tracker)

    stats = pipeline.process_video(
        video_input_path=video_p,
        video_output_path=video_out,
        max_frames=args.max_frames,
    )

    drive_placeholder = cfg.get("video", {}).get("google_drive_public_url", "")

    print("\n" + "=" * 65)
    print("VIDEO TRACKING COMPLETED")
    print("=" * 65)
    print(f"Frames processed          : {stats['total_frames_processed']}")
    print(f"Unique egg tracks identified: {stats['total_unique_tracks']}")
    print(f"Annotated output video    : {stats['video_output_path']}")
    print(f"Trajectory logs           : {stats['trajectory_csv']}")
    print(f"Submission Google Drive URL: {drive_placeholder}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
