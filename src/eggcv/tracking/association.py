"""
Track-to-detection data association algorithms (centroid distance and IoU matching).
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class Detection:
    """Represents an egg instance detection in a single frame."""
    centroid: Tuple[float, float]
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    area: float
    confidence: float = 1.0
    contour: Optional[np.ndarray] = None


def compute_centroid_distance_matrix(
    track_centroids: List[Tuple[float, float]],
    detection_centroids: List[Tuple[float, float]],
) -> np.ndarray:
    """
    Compute pairwise Euclidean distance matrix between existing track centroids
    and new frame detection centroids.

    Returns:
        np.ndarray: Matrix of shape (len(tracks), len(detections))
    """
    n_tracks = len(track_centroids)
    n_dets = len(detection_centroids)

    if n_tracks == 0 or n_dets == 0:
        return np.zeros((n_tracks, n_dets), dtype=np.float32)

    t_arr = np.array(track_centroids, dtype=np.float32)  # (N, 2)
    d_arr = np.array(detection_centroids, dtype=np.float32)  # (M, 2)

    diff = t_arr[:, np.newaxis, :] - d_arr[np.newaxis, :, :]  # (N, M, 2)
    dist_matrix = np.linalg.norm(diff, axis=2)  # (N, M)
    return dist_matrix


def associate_detections_to_tracks(
    track_centroids: List[Tuple[float, float]],
    detections: List[Detection],
    max_distance: float = 75.0,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    Associate detections to active tracks using Hungarian / Munkres minimum cost assignment.

    Args:
        track_centroids: List of (x, y) coordinates for active tracks
        detections: List of Detection objects in current frame
        max_distance: Maximum allowable Euclidean pixel distance for a valid match

    Returns:
        Tuple:
            - matches: List of (track_idx, detection_idx) pairs
            - unmatched_tracks: List of track indices with no valid detection
            - unmatched_detections: List of detection indices with no matching track
    """
    if not track_centroids:
        return [], [], list(range(len(detections)))

    if not detections:
        return [], list(range(len(track_centroids))), []

    det_centroids = [d.centroid for d in detections]
    cost_matrix = compute_centroid_distance_matrix(track_centroids, det_centroids)

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matches: List[Tuple[int, int]] = []
    unmatched_tracks = set(range(len(track_centroids)))
    unmatched_dets = set(range(len(detections)))

    for r, c in zip(row_ind, col_ind):
        dist = cost_matrix[r, c]
        if dist <= max_distance:
            matches.append((r, c))
            unmatched_tracks.discard(r)
            unmatched_dets.discard(c)

    return matches, sorted(list(unmatched_tracks)), sorted(list(unmatched_dets))
