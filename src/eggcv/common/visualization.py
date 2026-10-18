"""
Visualization utilities: mask overlays, error maps, contact sheets, and galleries.
"""

from typing import List, Optional, Tuple, Union

import cv2
import numpy as np


def overlay_mask(
    image: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int] = (0, 255, 0),  # BGR Green
    alpha: float = 0.45,
    draw_contour: bool = True,
    contour_thickness: int = 2,
) -> np.ndarray:
    """
    Create a semi-transparent colored overlay of a binary mask onto a BGR image.

    Args:
        image: Original BGR image (H, W, 3) uint8
        mask: Binary mask (H, W) where foreground > 0
        color: BGR tuple for foreground color
        alpha: Blending weight for mask color (0.0 to 1.0)
        draw_contour: Whether to draw a solid outline around mask boundary
        contour_thickness: Thickness of boundary contour line

    Returns:
        np.ndarray: Blended BGR image (H, W, 3)
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected BGR image with shape (H, W, 3), got {image.shape}")

    h, w = image.shape[:2]
    binary_mask = (mask > 0).astype(np.uint8)

    if binary_mask.shape[:2] != (h, w):
        binary_mask = cv2.resize(binary_mask, (w, h), interpolation=cv2.INTER_NEAREST)

    overlay = image.copy()
    colored_layer = np.zeros_like(image, dtype=np.uint8)
    colored_layer[binary_mask == 1] = color

    # Blend foreground pixels
    fg_mask = binary_mask == 1
    if np.any(fg_mask):
        blended = cv2.addWeighted(image, 1.0 - alpha, colored_layer, alpha, 0)
        overlay[fg_mask] = blended[fg_mask]

    if draw_contour and np.any(binary_mask):
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, color, contour_thickness)

    return overlay


def visualize_error_map(
    reference_mask: np.ndarray,
    predicted_mask: np.ndarray,
    image: Optional[np.ndarray] = None,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Generate an RGB/BGR visual error map comparing prediction against reference.
    Colors:
      - Green (0, 255, 0): True Positive (TP) - Predicted and in reference
      - Red   (0, 0, 255): False Positive (FP) - Predicted but absent in reference
      - Blue  (255, 0, 0): False Negative (FN) - In reference but missed by prediction

    Args:
        reference_mask: Binary reference mask (H, W)
        predicted_mask: Binary predicted mask (H, W)
        image: Optional BGR image to blend in background
        alpha: Opacity of color map overlay

    Returns:
        np.ndarray: BGR error map visualization
    """
    ref = (reference_mask > 0).astype(np.uint8)
    pred = (predicted_mask > 0).astype(np.uint8)

    tp = (ref == 1) & (pred == 1)
    fp = (ref == 0) & (pred == 1)
    fn = (ref == 1) & (pred == 0)

    h, w = ref.shape[:2]

    # Create colored error canvas in BGR
    error_layer = np.zeros((h, w, 3), dtype=np.uint8)
    error_layer[tp] = [0, 220, 0]    # Green: TP
    error_layer[fp] = [0, 0, 230]    # Red: FP
    error_layer[fn] = [230, 80, 0]   # Blue: FN

    active_pixels = tp | fp | fn

    if image is not None:
        base = image.copy()
        if base.shape[:2] != (h, w):
            base = cv2.resize(base, (w, h))
        output = base.copy()
        if np.any(active_pixels):
            blended = cv2.addWeighted(base, 1.0 - alpha, error_layer, alpha, 0)
            output[active_pixels] = blended[active_pixels]
    else:
        output = error_layer

    # Add legend badge in the top-left corner
    legend_h, legend_w = 48, 200
    cv2.rectangle(output, (5, 5), (5 + legend_w, 5 + legend_h), (30, 30, 30), -1)
    cv2.putText(output, "TP (Match)", (12, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    cv2.putText(output, "FP (False Alarm)", (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    cv2.putText(output, "FN (Missed)", (12, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 120, 0), 1)

    return output


def create_contact_sheet(
    images: List[np.ndarray],
    titles: Optional[List[str]] = None,
    cols: int = 5,
    tile_size: Tuple[int, int] = (160, 160),
    padding: int = 4,
    bg_color: Tuple[int, int, int] = (25, 25, 25),
) -> np.ndarray:
    """
    Arrange a list of images into a rectangular contact sheet grid with optional titles.

    Args:
        images: List of BGR images
        titles: Optional list of text labels for each tile
        cols: Number of columns in grid
        tile_size: (width, height) for each thumbnail tile
        padding: Pixel padding between thumbnails
        bg_color: BGR tuple for canvas background

    Returns:
        np.ndarray: Large composite BGR image
    """
    if not images:
        raise ValueError("Cannot create contact sheet from an empty list of images.")

    n = len(images)
    rows = (n + cols - 1) // cols
    tile_w, tile_h = tile_size

    banner_h = 18 if titles else 0
    cell_w = tile_w + padding * 2
    cell_h = tile_h + banner_h + padding * 2

    canvas_w = cols * cell_w
    canvas_h = rows * cell_h
    canvas = np.full((canvas_h, canvas_w, 3), bg_color, dtype=np.uint8)

    for idx, img in enumerate(images):
        r = idx // cols
        c = idx % cols

        x0 = c * cell_w + padding
        y0 = r * cell_h + padding

        # Resize image maintaining aspect ratio
        ih, iw = img.shape[:2]
        scale = min(tile_w / iw, tile_h / ih)
        nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

        # Center within tile
        px = x0 + (tile_w - nw) // 2
        py = y0 + (tile_h - nh) // 2
        canvas[py : py + nh, px : px + nw] = resized

        # Add title text if available
        if titles and idx < len(titles):
            title = titles[idx]
            text_y = y0 + tile_h + 14
            cv2.putText(
                canvas,
                str(title)[:24],
                (x0 + 2, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.32,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )

    return canvas
