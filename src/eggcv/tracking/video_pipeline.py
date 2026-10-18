"""
End-to-end video processing pipeline for egg detection, tracking, and visualization (Task 4).
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from eggcv.common.logging_utils import get_logger
from eggcv.segmentation.classical_segmenter import ClassicalEggSegmenter
from eggcv.tracking.association import Detection
from eggcv.tracking.tracker import EggTracker

logger = get_logger("eggcv.tracking.pipeline")


class VideoTrackingPipeline:
    """Pipelines video frame reading, segmentation, association, and video writing."""

    def __init__(
        self,
        segmenter: Optional[ClassicalEggSegmenter] = None,
        tracker: Optional[EggTracker] = None,
    ):
        self.segmenter = segmenter or ClassicalEggSegmenter()
        self.tracker = tracker or EggTracker()

    def process_video(
        self,
        video_input_path: Union[str, Path],
        video_output_path: Union[str, Path],
        max_frames: Optional[int] = None,
        draw_trails: bool = True,
        trail_length: int = 30,
    ) -> Dict[str, Any]:
        """
        Process an input video file, run tracking on all frames, and write annotated video.

        Args:
            video_input_path: Path to raw input video
            video_output_path: Path for annotated output video
            max_frames: Optional limit on number of frames to process
            draw_trails: If True, draws trajectory motion history lines
            trail_length: Number of past centroid points to draw

        Returns:
            Dict containing video and tracking statistics
        """
        in_path = Path(video_input_path)
        if not in_path.is_file():
            raise FileNotFoundError(
                f"Video input file not found: {in_path.resolve()}\n"
                f"Download the assigned YouTube video (https://www.youtube.com/watch?v=Y4JyV94y2_A) "
                f"and place it at {in_path}."
            )

        cap = cv2.VideoCapture(str(in_path))
        if not cap.isOpened():
            raise IOError(f"Failed to open video file: {in_path.resolve()}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        limit = min(total_frames, max_frames) if max_frames else total_frames

        out_path = Path(video_output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

        logger.info(f"Processing video: {in_path.name} ({width}x{height} @ {fps:.1f} fps, frames: {limit})")

        frame_count = 0
        trajectory_records: List[Dict[str, Any]] = []

        try:
            with tqdm(total=limit, desc="Tracking Video Frames") as pbar:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret or (limit and frame_count >= limit):
                        break

                    frame_count += 1

                    # 1. Detect eggs using classical segmenter
                    mask, meta = self.segmenter.segment(frame)

                    # 2. Extract Detections
                    detections: List[Detection] = []
                    for comp in meta.get("components", []):
                        detections.append(Detection(
                            centroid=comp["centroid"],
                            bbox=comp["bbox"],
                            area=comp["area"],
                        ))

                    # 3. Update tracker
                    confirmed_tracks = self.tracker.update(detections)

                    # 4. Annotate frame
                    annotated_frame = frame.copy()

                    # Draw trails
                    if draw_trails:
                        for trk in confirmed_tracks:
                            pts = trk.centroids[-trail_length:]
                            for i in range(1, len(pts)):
                                pt1 = (int(pts[i - 1][0]), int(pts[i - 1][1]))
                                pt2 = (int(pts[i][0]), int(pts[i][1]))
                                cv2.line(annotated_frame, pt1, pt2, (0, 255, 255), 2)

                    # Draw boxes and IDs
                    for trk in confirmed_tracks:
                        cx, cy = int(trk.current_centroid[0]), int(trk.current_centroid[1])
                        x, y, w, h = trk.current_bbox

                        # Bounding box
                        cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        # Centroid marker
                        cv2.circle(annotated_frame, (cx, cy), 4, (0, 0, 255), -1)
                        # ID label badge
                        label = f"Egg ID: {trk.track_id}"
                        cv2.putText(
                            annotated_frame, label, (x, max(20, y - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA
                        )

                        trajectory_records.append({
                            "frame": frame_count,
                            "track_id": trk.track_id,
                            "centroid_x": cx,
                            "centroid_y": cy,
                            "bbox_x": x,
                            "bbox_y": y,
                            "bbox_w": w,
                            "bbox_h": h,
                            "area": trk.areas[-1],
                        })

                    # Frame counter badge
                    cv2.putText(
                        annotated_frame, f"Frame: {frame_count}/{limit}", (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
                    )

                    writer.write(annotated_frame)
                    pbar.update(1)

        finally:
            cap.release()
            writer.release()

        # Save trajectory log
        log_csv_path = out_path.parent / "tracking_trajectories.csv"
        if trajectory_records:
            pd.DataFrame(trajectory_records).to_csv(log_csv_path, index=False)

        all_tracks = self.tracker.get_all_tracks()
        stats = {
            "total_frames_processed": frame_count,
            "total_unique_tracks": len(all_tracks),
            "video_output_path": str(out_path),
            "trajectory_csv": str(log_csv_path) if trajectory_records else None,
        }
        logger.info(f"Video tracking complete: {frame_count} frames, {len(all_tracks)} unique egg tracks.")
        return stats
