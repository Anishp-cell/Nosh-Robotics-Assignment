#!/usr/bin/env python3
"""
QA Tool: Periodic video frame sampling (every 1.5s) and segmentation grid generation.

This script:
  1. Opens the target video (640x360 @ 24 FPS).
  2. Samples frames every 1.5 seconds (interval = 36 frames, 40 total sample points).
  3. Runs ClassicalEggSegmenter on each sampled frame.
  4. Generates a 3-panel comparison for each sample (Raw Frame | Binary Mask | Detection Overlay).
  5. Assembles all sample panels into structured contact sheet galleries
     (similar to outputs/evaluation/gallery/best_5_successes.jpg).
  6. Outputs diagnostic statistics (egg counts per sample, false positive indicators).
"""

import argparse
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from eggcv.common.io import load_yaml, save_image
from eggcv.common.logging_utils import get_logger
from eggcv.common.visualization import overlay_mask
from eggcv.segmentation.classical_segmenter import ClassicalEggSegmenter

logger = get_logger("video_qa_grid")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate 1.5s interval video segmentation QA grids.")
    parser.add_argument(
        "--video-path",
        type=str,
        default=str(PROJECT_ROOT / "data" / "raw" / "video" / "egg_video.mp4"),
        help="Path to input video file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "segmentation.yaml"),
        help="Path to segmentation config YAML",
    )
    parser.add_argument(
        "--interval-seconds",
        type=float,
        default=1.5,
        help="Sampling interval in seconds (default: 1.5s)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "outputs" / "tracking" / "qa_gallery"),
        help="Directory to save contact sheets and individual inspection snapshots",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="baseline",
        help="Tag name for the output grids (e.g. baseline, optimized)",
    )
    return parser.parse_args()


def make_sample_panel(
    frame: np.ndarray,
    mask: np.ndarray,
    components: List[Dict],
    frame_idx: int,
    timestamp_s: float,
) -> np.ndarray:
    """
    Create a clean 3-view panel: [Raw Frame | Binary Mask | Detection Overlay]
    with informative top badge.
    """
    h, w = frame.shape[:2]

    # 1. Binary mask in 3 channels
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # 2. Overlay with contours, bboxes, centroids
    overlay = overlay_mask(frame, mask, color=(0, 255, 0), alpha=0.4, draw_contour=True)
    num_eggs = len(components)

    for i, comp in enumerate(components):
        bx, by, bw, bh = comp["bbox"]
        cx, cy = comp["centroid"]
        cv2.rectangle(overlay, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
        cv2.circle(overlay, (cx, cy), 4, (0, 0, 255), -1)
        label = f"Egg #{i+1} ({int(comp['area'])}px)"
        cv2.putText(
            overlay,
            label,
            (bx, max(15, by - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    # Add header banner across each view
    views = [frame.copy(), mask_bgr, overlay]
    titles = [
        f"Raw Frame #{frame_idx} ({timestamp_s:.1f}s)",
        f"Binary Mask ({int(np.count_nonzero(mask))} px)",
        f"Detected: {num_eggs} egg{'s' if num_eggs != 1 else ''}",
    ]
    colors = [(255, 255, 255), (200, 200, 200), (0, 255, 0) if (1 <= num_eggs <= 2) else (50, 50, 255)]

    for view, title, col in zip(views, titles, colors):
        banner = np.zeros((26, w, 3), dtype=np.uint8)
        cv2.putText(banner, title, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
        cv2.line(banner, (0, 25), (w, 25), (60, 60, 60), 1)

    trio = np.hstack([np.vstack([np.zeros((26, w, 3), dtype=np.uint8), v]) for v in views])
    # Put actual text in top bar
    for idx, (title, col) in enumerate(zip(titles, colors)):
        x_offset = idx * w
        cv2.putText(trio, title, (x_offset + 8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)

    return trio


def assemble_grid(
    panels: List[np.ndarray],
    cols: int = 2,
    header_text: str = "Video QA Contact Sheet",
) -> np.ndarray:
    """
    Stack panels into a multi-row grid with a distinct title banner.
    """
    if not panels:
        raise ValueError("Cannot assemble grid with zero panels.")

    target_w = 640  # downscale each 3-view panel width for clean grid rendering
    resized_panels = []
    for p in panels:
        ph, pw = p.shape[:2]
        new_h = int(ph * (target_w / pw))
        resized_panels.append(cv2.resize(p, (target_w, new_h), interpolation=cv2.INTER_AREA))

    n = len(resized_panels)
    rows = (n + cols - 1) // cols
    tile_h = resized_panels[0].shape[0]

    grid_w = cols * target_w
    grid_h = rows * tile_h + 50
    canvas = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

    # Master title banner
    cv2.rectangle(canvas, (0, 0), (grid_w, 50), (35, 35, 35), -1)
    cv2.putText(
        canvas,
        header_text,
        (20, 33),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (0, 255, 200),
        2,
        cv2.LINE_AA,
    )

    for idx, panel in enumerate(resized_panels):
        r = idx // cols
        c = idx % cols
        y0 = 50 + r * tile_h
        x0 = c * target_w
        canvas[y0 : y0 + tile_h, x0 : x0 + target_w] = panel

    return canvas


def main():
    args = parse_args()
    video_path = Path(args.video_path)
    if not video_path.is_file():
        logger.error(f"Video file not found: {video_path}")
        sys.exit(1)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_yaml(args.config) if Path(args.config).is_file() else {}
    segmenter = ClassicalEggSegmenter(cfg)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s = total_frames / fps
    step_frames = max(1, int(round(args.interval_seconds * fps)))

    sample_frame_indices = list(range(0, total_frames, step_frames))
    logger.info(
        f"Video: {video_path.name} | {total_frames} frames ({duration_s:.1f}s @ {fps:.1f} FPS) | "
        f"Sampling every {args.interval_seconds}s ({step_frames} frames) -> {len(sample_frame_indices)} snapshots."
    )

    panels: List[np.ndarray] = []
    stats: List[Dict] = []

    for frame_idx in sample_frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        timestamp_s = frame_idx / fps
        mask, meta = segmenter.segment(frame)
        components = meta.get("components", [])

        panel = make_sample_panel(
            frame=frame,
            mask=mask,
            components=components,
            frame_idx=frame_idx,
            timestamp_s=timestamp_s,
        )
        panels.append(panel)

        stats.append({
            "frame_idx": frame_idx,
            "timestamp_s": round(timestamp_s, 2),
            "num_detected_eggs": len(components),
            "total_fg_pixels": int(meta.get("total_foreground_pixels", 0)),
            "component_areas": [int(c["area"]) for c in components],
        })

    cap.release()

    # Split into 2 logical sheets (Part 1: 0-30s, Part 2: 30-60s) for readable resolution
    half = len(panels) // 2
    sheet1 = assemble_grid(
        panels[:half],
        cols=2,
        header_text=f"Detection QA Grid ({args.tag.upper()}) — Part 1: 0.0s to {stats[half-1]['timestamp_s']}s",
    )
    sheet2 = assemble_grid(
        panels[half:],
        cols=2,
        header_text=f"Detection QA Grid ({args.tag.upper()}) — Part 2: {stats[half]['timestamp_s']}s to {stats[-1]['timestamp_s']}s",
    )

    out_p1 = out_dir / f"detection_grid_{args.tag}_part1.jpg"
    out_p2 = out_dir / f"detection_grid_{args.tag}_part2.jpg"
    save_image(sheet1, out_p1)
    save_image(sheet2, out_p2)

    logger.info(f"Saved contact sheets:\n  -> {out_p1}\n  -> {out_p2}")

    # Print diagnostic metrics
    counts = [s["num_detected_eggs"] for s in stats]
    zero_eggs = sum(1 for c in counts if c == 0)
    exact_eggs = sum(1 for c in counts if 1 <= c <= 2)
    over_detected = sum(1 for c in counts if c > 2)

    print("\n" + "=" * 65)
    print(f"DETECTION QA EVALUATION SUMMARY: [{args.tag.upper()}]")
    print("=" * 65)
    print(f"Sampled frames inspected       : {len(stats)} (every {args.interval_seconds}s)")
    print(f"Frames with 1 or 2 eggs (Good) : {exact_eggs}/{len(stats)} ({exact_eggs/len(stats)*100:.1f}%)")
    print(f"Frames with > 2 detections (FP): {over_detected}/{len(stats)} ({over_detected/len(stats)*100:.1f}%)")
    print(f"Frames with 0 detections       : {zero_eggs}/{len(stats)} ({zero_eggs/len(stats)*100:.1f}%)")
    print(f"Max detected in single frame   : {max(counts) if counts else 0}")
    print(f"Output Gallery Part 1          : {out_p1}")
    print(f"Output Gallery Part 2          : {out_p2}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
