"""
Persistent multi-object egg tracker maintaining consistent IDs across video frames.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from eggcv.tracking.association import Detection, associate_detections_to_tracks


class EggTrack:
    """Represents a single egg instance trajectory across video frames."""

    def __init__(self, track_id: int, initial_detection: Detection, start_frame: int):
        self.track_id = track_id
        self.centroids: List[Tuple[float, float]] = [initial_detection.centroid]
        self.bboxes: List[Tuple[int, int, int, int]] = [initial_detection.bbox]
        self.areas: List[float] = [initial_detection.area]
        self.frame_indices: List[int] = [start_frame]

        self.hits: int = 1
        self.consecutive_lost: int = 0
        self.status: str = "active"  # "active", "lost", or "terminated"

    @property
    def current_centroid(self) -> Tuple[float, float]:
        return self.centroids[-1]

    @property
    def current_bbox(self) -> Tuple[int, int, int, int]:
        return self.bboxes[-1]

    def update(self, detection: Detection, frame_idx: int) -> None:
        """Update track state with matching detection."""
        self.centroids.append(detection.centroid)
        self.bboxes.append(detection.bbox)
        self.areas.append(detection.area)
        self.frame_indices.append(frame_idx)
        self.hits += 1
        self.consecutive_lost = 0
        self.status = "active"

    def mark_missed(self) -> None:
        """Mark track as missed in the current frame."""
        self.consecutive_lost += 1
        self.status = "lost"


class EggTracker:
    """
    Multi-object tracker for persistent egg tracking across frames.
    Uses centroid Euclidean distance and Hungarian assignment.
    """

    def __init__(
        self,
        max_distance: float = 75.0,
        max_consecutive_lost: int = 20,
        min_hits_to_confirm: int = 2,
    ):
        self.max_distance = max_distance
        self.max_consecutive_lost = max_consecutive_lost
        self.min_hits_to_confirm = min_hits_to_confirm

        self.tracks: Dict[int, EggTrack] = {}
        self.next_track_id: int = 1
        self.current_frame_idx: int = 0

    def update(self, detections: List[Detection]) -> List[EggTrack]:
        """
        Process detections for current frame, update tracks, spawn new tracks,
        and terminate stale tracks.

        Returns:
            List[EggTrack]: Active tracks confirmed in the current frame
        """
        self.current_frame_idx += 1

        active_track_ids = [
            tid for tid, trk in self.tracks.items()
            if trk.status in ("active", "lost")
        ]
        active_centroids = [self.tracks[tid].current_centroid for tid in active_track_ids]

        matches, unmatched_track_indices, unmatched_det_indices = associate_detections_to_tracks(
            track_centroids=active_centroids,
            detections=detections,
            max_distance=self.max_distance,
        )

        # 1. Update matched tracks
        for trk_idx, det_idx in matches:
            tid = active_track_ids[trk_idx]
            self.tracks[tid].update(detections[det_idx], self.current_frame_idx)

        # 2. Mark missed tracks
        for trk_idx in unmatched_track_indices:
            tid = active_track_ids[trk_idx]
            trk = self.tracks[tid]
            trk.mark_missed()
            if trk.consecutive_lost > self.max_consecutive_lost:
                trk.status = "terminated"

        # 3. Spawn new tracks for unmatched detections
        for det_idx in unmatched_det_indices:
            new_id = self.next_track_id
            self.next_track_id += 1
            new_track = EggTrack(
                track_id=new_id,
                initial_detection=detections[det_idx],
                start_frame=self.current_frame_idx,
            )
            self.tracks[new_id] = new_track

        # Return tracks confirmed in current frame
        confirmed = [
            trk for trk in self.tracks.values()
            if trk.status == "active" and trk.hits >= self.min_hits_to_confirm
        ]
        return confirmed

    def get_all_tracks(self) -> Dict[int, EggTrack]:
        return self.tracks
