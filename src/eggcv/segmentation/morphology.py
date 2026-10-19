"""
Morphological operations, hole filling, connected components filtering,
and watershed separation for egg segmentation.
"""

import math
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


def morphological_clean(
    mask: np.ndarray,
    open_ksize: int = 5,
    open_iters: int = 1,
    close_ksize: int = 7,
    close_iters: int = 2,
) -> np.ndarray:
    """
    Clean binary mask using mathematical morphology.
    Performs opening (erosion then dilation) to eliminate speckle noise,
    followed by closing (dilation then erosion) to seal internal cavities.

    Args:
        mask: 2D binary mask array (uint8 with 0/255 or bool)
        open_ksize: Kernel size for opening operation
        open_iters: Iterations of opening
        close_ksize: Kernel size for closing operation
        close_iters: Iterations of closing

    Returns:
        np.ndarray: Cleaned binary mask (uint8 with 0 or 255)
    """
    binary = np.ascontiguousarray((mask > 0).astype(np.uint8) * 255)

    if open_iters > 0 and open_ksize > 1:
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ksize, open_ksize))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open, iterations=open_iters)

    if close_iters > 0 and close_ksize > 1:
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close, iterations=close_iters)

    return binary


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """
    Fill internal holes (e.g. from specular glare/reflections) in binary foreground objects.
    Uses border padding and flood fill from boundaries to distinguish outer background
    from internal voids, even if foreground objects touch the image edge.

    Args:
        mask: 2D binary mask array (uint8 or bool)

    Returns:
        np.ndarray: Binary mask with internal holes filled (uint8 with 0 or 255)
    """
    binary = np.ascontiguousarray((mask > 0).astype(np.uint8) * 255)
    h, w = binary.shape

    # Pad border with 0s to guarantee (0, 0) is background
    padded = cv2.copyMakeBorder(binary, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    flood_canvas = np.zeros((h + 4, w + 4), np.uint8)
    im_floodfill = padded.copy()
    cv2.floodFill(im_floodfill, flood_canvas, (0, 0), 255)

    # Invert floodfilled image to isolate internal holes
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)

    # Crop out the padding
    holes = im_floodfill_inv[1:-1, 1:-1]

    # Combine original mask with internal holes
    filled_mask = binary | holes
    return filled_mask


def filter_connected_components(
    mask: np.ndarray,
    min_area: float = 2000.0,
    max_area: Optional[float] = 320000.0,
    min_circularity: float = 0.20,
    min_aspect_ratio: float = 0.35,
    max_aspect_ratio: float = 2.85,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Filter binary components using egg-like geometric priors (area, circularity, aspect ratio).

    Circularity is defined as: 4 * pi * area / (perimeter^2)
    For a perfect circle, circularity = 1.0; for typical eggs, it is ~0.7 to 0.9.

    Args:
        mask: 2D binary mask
        min_area: Minimum pixel area to eliminate dust/noise
        max_area: Maximum pixel area to avoid full-canvas background flooding
        min_circularity: Minimum isoperimetric quotient
        min_aspect_ratio: Minimum ratio of min(width, height) / max(width, height)
        max_aspect_ratio: Maximum aspect ratio width/height

    Returns:
        Tuple:
            - np.ndarray: Cleaned binary mask containing only valid components
            - List[Dict]: Metadata for each accepted component (area, centroid, bbox, circularity)
    """
    binary = np.ascontiguousarray((mask > 0).astype(np.uint8) * 255)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    filtered_mask = np.zeros(binary.shape[:2], dtype=np.uint8, order="C")
    accepted_components: List[Dict[str, Any]] = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        if max_area is not None and area > max_area:
            continue

        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue

        circularity = (4.0 * math.pi * area) / (perimeter * perimeter)
        if circularity < min_circularity:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / float(h) if h > 0 else 0.0
        # Check aspect ratio
        if aspect_ratio < min_aspect_ratio or aspect_ratio > max_aspect_ratio:
            continue

        # Compute centroid
        moments = cv2.moments(contour)
        cx = int(moments["m10"] / moments["m00"]) if moments["m00"] != 0 else x + w // 2
        cy = int(moments["m01"] / moments["m00"]) if moments["m00"] != 0 else y + h // 2

        # Draw accepted contour onto filtered mask
        cv2.drawContours(filtered_mask, [contour], -1, 255, -1)

        accepted_components.append({
            "area": float(area),
            "perimeter": float(perimeter),
            "circularity": float(circularity),
            "bbox": (int(x), int(y), int(w), int(h)),
            "centroid": (cx, cy),
            "aspect_ratio": float(aspect_ratio),
        })

    return filtered_mask, accepted_components


def separate_touching_objects(
    mask: np.ndarray,
    distance_threshold_ratio: float = 0.55,
    min_distance_px: int = 20,
) -> np.ndarray:
    """
    Separate touching egg instances using Euclidean Distance Transform and Watershed segmentation.

    Args:
        mask: Binary foreground mask (uint8 0/255)
        distance_threshold_ratio: Threshold multiplier for distance peaks (0.0 - 1.0)
        min_distance_px: Minimum Euclidean distance in pixels between candidate instance centers

    Returns:
        np.ndarray: Mask with separated instances
    """
    binary = np.ascontiguousarray((mask > 0).astype(np.uint8) * 255)
    if not np.any(binary):
        return binary

    # Exact Euclidean Distance Transform
    dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)

    max_dist = dist_transform.max()
    if max_dist <= 0:
        return binary

    # Identify sure foreground seeds
    sure_fg_thresh = max_dist * distance_threshold_ratio
    _, sure_fg = cv2.threshold(dist_transform, sure_fg_thresh, 255, 0)
    sure_fg = sure_fg.astype(np.uint8)

    # Sure background area via dilation
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    sure_bg = cv2.dilate(binary, kernel, iterations=3)

    # Unknown region between background and sure foreground
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Label markers
    num_markers, markers = cv2.connectedComponents(sure_fg)
    # Add one to all labels so that background is 1, not 0
    markers = markers + 1
    # Mark the unknown region with 0
    markers[unknown == 255] = 0

    # Watershed requires 3-channel input
    color_stub = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(color_stub, markers)

    # Boundaries are assigned -1 by watershed; reconstruct separated foreground mask
    separated = np.where((markers > 1), 255, 0).astype(np.uint8)
    return separated
