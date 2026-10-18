"""
Unit tests for data association and persistent tracking on synthetic detections.
"""

from eggcv.tracking.association import (
    Detection,
    associate_detections_to_tracks,
    compute_centroid_distance_matrix,
)
from eggcv.tracking.tracker import EggTracker


def test_distance_matrix_computation():
    tracks = [(0.0, 0.0), (100.0, 100.0)]
    dets = [(0.0, 10.0), (100.0, 105.0)]

    mat = compute_centroid_distance_matrix(tracks, dets)
    assert mat.shape == (2, 2)
    assert mat[0, 0] == 10.0
    assert mat[1, 1] == 5.0
    assert mat[0, 1] > 100.0


def test_hungarian_association():
    tracks = [(50.0, 50.0), (150.0, 150.0)]
    dets = [
        Detection(centroid=(152.0, 149.0), bbox=(130, 130, 40, 40), area=1600),
        Detection(centroid=(51.0, 49.0), bbox=(30, 30, 40, 40), area=1600),
    ]

    matches, unmatched_trks, unmatched_dets = associate_detections_to_tracks(
        track_centroids=tracks,
        detections=dets,
        max_distance=20.0,
    )

    # Track 0 (50, 50) matches Detection 1 (51, 49)
    # Track 1 (150, 150) matches Detection 0 (152, 149)
    assert len(matches) == 2
    assert (0, 1) in matches
    assert (1, 0) in matches
    assert len(unmatched_trks) == 0
    assert len(unmatched_dets) == 0


def test_tracker_identity_persistence():
    tracker = EggTracker(max_distance=30.0, max_consecutive_lost=5, min_hits_to_confirm=2)

    # Frame 1: Two eggs detected
    f1_dets = [
        Detection(centroid=(100.0, 100.0), bbox=(80, 80, 40, 40), area=1600),
        Detection(centroid=(250.0, 250.0), bbox=(230, 230, 40, 40), area=1600),
    ]
    tracks_f1 = tracker.update(f1_dets)
    # First frame has hits=1, min_hits_to_confirm=2 so not yet confirmed
    assert len(tracker.get_all_tracks()) == 2

    # Frame 2: Eggs move slightly
    f2_dets = [
        Detection(centroid=(102.0, 101.0), bbox=(82, 81, 40, 40), area=1600),
        Detection(centroid=(253.0, 248.0), bbox=(233, 228, 40, 40), area=1600),
    ]
    confirmed_f2 = tracker.update(f2_dets)
    assert len(confirmed_f2) == 2
    ids_f2 = {t.track_id for t in confirmed_f2}
    assert ids_f2 == {1, 2}

    # Frame 3: Eggs move slightly again
    f3_dets = [
        Detection(centroid=(105.0, 103.0), bbox=(85, 83, 40, 40), area=1600),
        Detection(centroid=(256.0, 246.0), bbox=(236, 226, 40, 40), area=1600),
    ]
    confirmed_f3 = tracker.update(f3_dets)
    assert len(confirmed_f3) == 2
    ids_f3 = {t.track_id for t in confirmed_f3}
    # Track IDs MUST stay identical (persistent)
    assert ids_f3 == {1, 2}


def test_tracker_lost_and_recovery():
    tracker = EggTracker(max_distance=40.0, max_consecutive_lost=3, min_hits_to_confirm=1)

    # Frame 1: Single egg
    tracker.update([Detection(centroid=(100.0, 100.0), bbox=(80, 80, 40, 40), area=1600)])
    assert tracker.tracks[1].status == "active"

    # Frame 2: Occluded / Missed detection
    tracker.update([])
    assert tracker.tracks[1].status == "lost"
    assert tracker.tracks[1].consecutive_lost == 1

    # Frame 3: Reappears nearby
    confirmed = tracker.update([Detection(centroid=(102.0, 101.0), bbox=(82, 81, 40, 40), area=1600)])
    assert len(confirmed) == 1
    assert confirmed[0].track_id == 1
    assert confirmed[0].status == "active"
    assert confirmed[0].consecutive_lost == 0


def test_tracker_stale_termination():
    tracker = EggTracker(max_distance=40.0, max_consecutive_lost=2, min_hits_to_confirm=1)

    tracker.update([Detection(centroid=(100.0, 100.0), bbox=(80, 80, 40, 40), area=1600)])
    assert tracker.tracks[1].status == "active"

    # Miss 1
    tracker.update([])
    assert tracker.tracks[1].status == "lost"

    # Miss 2
    tracker.update([])
    assert tracker.tracks[1].status == "lost"

    # Miss 3 (> max_consecutive_lost)
    tracker.update([])
    assert tracker.tracks[1].status == "terminated"
