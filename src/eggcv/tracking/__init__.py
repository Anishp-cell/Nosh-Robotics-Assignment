"""
Video tracking package: detection association, multi-object tracking, and video pipeline.
"""

from eggcv.tracking.association import (
    Detection,
    associate_detections_to_tracks,
    compute_centroid_distance_matrix,
)
from eggcv.tracking.tracker import EggTrack, EggTracker
from eggcv.tracking.video_pipeline import VideoTrackingPipeline

__all__ = [
    "Detection",
    "compute_centroid_distance_matrix",
    "associate_detections_to_tracks",
    "EggTrack",
    "EggTracker",
    "VideoTrackingPipeline",
]
