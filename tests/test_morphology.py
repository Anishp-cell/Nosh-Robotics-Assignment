"""
Unit tests for morphological cleaning, hole filling, and geometric filtering.
"""

import cv2
import numpy as np
import pytest

from eggcv.segmentation.morphology import (
    fill_holes,
    filter_connected_components,
    morphological_clean,
    separate_touching_objects,
)


def test_morphological_clean_removes_noise():
    # Large object with isolated salt noise
    canvas = np.zeros((100, 100), dtype=np.uint8)
    # Main circular mass (radius 20)
    cv2.circle(canvas, (50, 50), 20, 255, -1)
    # Speckle noise (1-2 px dots)
    canvas[10, 10] = 255
    canvas[90, 15] = 255

    cleaned = morphological_clean(canvas, open_ksize=5, open_iters=1, close_ksize=5, close_iters=1)

    # Specks should be removed
    assert cleaned[10, 10] == 0
    assert cleaned[90, 15] == 0
    # Core should be preserved
    assert cleaned[50, 50] == 255


def test_fill_holes():
    # Create an annular ring (a circle with a hole in the center)
    canvas = np.zeros((100, 100), dtype=np.uint8)
    cv2.circle(canvas, (50, 50), 30, 255, -1)  # Outer circle
    cv2.circle(canvas, (50, 50), 10, 0, -1)    # Hole in middle

    assert canvas[50, 50] == 0  # Initial hole

    filled = fill_holes(canvas)

    assert filled[50, 50] == 255  # Hole successfully filled
    assert filled[0, 0] == 0      # Background outside remains 0


def test_filter_connected_components_by_area_and_circularity():
    canvas = np.zeros((200, 200), dtype=np.uint8)

    # 1. Egg-like ellipse (large area, high circularity)
    cv2.ellipse(canvas, (60, 60), (35, 25), 0, 0, 360, 255, -1)

    # 2. Tiny speck (area < 100)
    canvas[10:15, 10:15] = 255

    # 3. Thin long wire (aspect ratio > 10, circularity < 0.1)
    canvas[180:183, 10:190] = 255

    filtered, comps = filter_connected_components(
        canvas,
        min_area=500.0,
        min_circularity=0.20,
        max_aspect_ratio=3.0,
    )

    # Only the egg-like ellipse should survive
    assert len(comps) == 1
    assert comps[0]["area"] > 1000
    assert comps[0]["circularity"] > 0.5
    assert filtered[60, 60] == 255
    # Wire and speck must be removed
    assert filtered[12, 12] == 0
    assert filtered[181, 100] == 0


def test_separate_touching_objects():
    canvas = np.zeros((150, 250), dtype=np.uint8)
    # Two touching circles with slight overlap
    cv2.circle(canvas, (80, 75), 35, 255, -1)
    cv2.circle(canvas, (145, 75), 35, 255, -1)

    # Initially single connected component
    n_init, _ = cv2.connectedComponents(canvas)
    assert n_init == 2  # Background + 1 merged blob

    separated = separate_touching_objects(canvas, distance_threshold_ratio=0.55, min_distance_px=15)

    # After watershed separation, distance boundary divides them
    n_sep, _ = cv2.connectedComponents(separated)
    assert n_sep >= 3  # Background + at least 2 distinct components
