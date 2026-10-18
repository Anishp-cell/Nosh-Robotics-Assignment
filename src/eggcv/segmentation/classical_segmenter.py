"""
Classical Computer Vision Egg Segmentation Algorithm using OpenCV and NumPy.
Modular, configurable, leakage-aware pipeline tuned strictly on the development split.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from eggcv.common.logging_utils import get_logger
from eggcv.segmentation.morphology import (
    fill_holes,
    filter_connected_components,
    morphological_clean,
    separate_touching_objects,
)

logger = get_logger("eggcv.segmentation.classical")


class BaseSegmenter(ABC):
    """Abstract base class for all egg segmentation models."""

    @abstractmethod
    def segment(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Segment egg(s) in a BGR image.

        Args:
            image: BGR image array (H, W, 3) uint8

        Returns:
            Tuple of:
                - binary_mask: 2D uint8 array with 0 (background) and 255 (egg)
                - metadata: Dictionary containing detected component attributes
        """
        pass


class ClassicalEggSegmenter(BaseSegmenter):
    """
    Classical egg segmentation pipeline utilizing color space transformations,
    adaptive/Otsu thresholding, morphological refinement, and geometric shape filtering.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the segmenter with modular configuration parameters.
        """
        self.config = config or {}
        self.preprocessing_cfg = self.config.get("preprocessing", {})
        self.color_cfg = self.config.get("color_analysis", {})
        self.thresh_cfg = self.config.get("thresholding", {})
        self.morph_cfg = self.config.get("morphology", {})
        self.filter_cfg = self.config.get("component_filtering", {})
        self.watershed_cfg = self.config.get("watershed", {})

    def segment(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Execute the full classical segmentation pipeline on a single BGR image.
        """
        if image is None or image.size == 0:
            raise ValueError("Input image is empty or None.")

        # 1. Preprocessing / Smoothing
        preprocessed = self._apply_preprocessing(image)

        # 2. Color Space Conversion & Feature Channel Selection
        channel = self._extract_feature_channel(preprocessed)

        # 3. Thresholding
        raw_mask = self._apply_threshold(channel)

        # 4. Morphological Cleaning & Hole Filling
        cleaned_mask = self._apply_morphology(raw_mask)

        # 5. Optional Watershed for Touching Objects
        if self.watershed_cfg.get("enabled", False):
            cleaned_mask = separate_touching_objects(
                cleaned_mask,
                distance_threshold_ratio=self.watershed_cfg.get("distance_threshold_ratio", 0.4),
                min_distance_px=self.watershed_cfg.get("min_distance_px", 20),
            )

        # 6. Connected Component Geometric Filtering
        final_mask, components = filter_connected_components(
            cleaned_mask,
            min_area=float(self.filter_cfg.get("min_area", 2000.0)),
            max_area=float(self.filter_cfg.get("max_area", 320000.0)) if self.filter_cfg.get("max_area") else None,
            min_circularity=float(self.filter_cfg.get("min_circularity", 0.25)),
            min_aspect_ratio=float(self.filter_cfg.get("min_aspect_ratio", 0.35)),
            max_aspect_ratio=float(self.filter_cfg.get("max_aspect_ratio", 2.80)),
        )

        metadata = {
            "num_detected_eggs": len(components),
            "components": components,
            "total_foreground_pixels": int(np.count_nonzero(final_mask)),
        }

        return final_mask, metadata

    def _apply_preprocessing(self, image: np.ndarray) -> np.ndarray:
        blur_method = self.preprocessing_cfg.get("blur_method", "gaussian")
        ksize = tuple(self.preprocessing_cfg.get("blur_ksize", [5, 5]))
        sigma = float(self.preprocessing_cfg.get("blur_sigma", 1.0))

        if blur_method == "gaussian":
            return cv2.GaussianBlur(image, ksize, sigma)
        elif blur_method == "median":
            k = max(3, ksize[0] if ksize[0] % 2 != 0 else ksize[0] + 1)
            return cv2.medianBlur(image, k)
        elif blur_method == "bilateral":
            return cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
        return image

    def _extract_feature_channel(self, image: np.ndarray) -> np.ndarray:
        space = self.color_cfg.get("color_space", "lab").lower()
        channel_name = self.color_cfg.get("channel", "l").lower()

        if space == "lab":
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            ch_idx = {"l": 0, "a": 1, "b": 2}.get(channel_name, 0)
            ch = lab[:, :, ch_idx]
        elif space == "hsv":
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            ch_idx = {"h": 0, "s": 1, "v": 2}.get(channel_name, 2)
            ch = hsv[:, :, ch_idx]
        elif space == "ycrcb":
            ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
            ch_idx = {"y": 0, "cr": 1, "cb": 2}.get(channel_name, 0)
            ch = ycrcb[:, :, ch_idx]
        else:
            ch = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Optional CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe_cfg = self.color_cfg.get("clahe", {})
        if clahe_cfg.get("enabled", False):
            clip = float(clahe_cfg.get("clip_limit", 2.0))
            grid = tuple(clahe_cfg.get("tile_grid_size", [8, 8]))
            clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=grid)
            ch = clahe.apply(ch)

        return ch

    def _apply_threshold(self, channel: np.ndarray) -> np.ndarray:
        method = self.thresh_cfg.get("method", "otsu").lower()
        invert = self.thresh_cfg.get("invert", False)

        thresh_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY

        if method == "otsu":
            _, binary = cv2.threshold(channel, 0, 255, thresh_type | cv2.THRESH_OTSU)
        elif method == "adaptive_gaussian":
            block = int(self.thresh_cfg.get("adaptive_block_size", 31))
            if block % 2 == 0:
                block += 1
            c = float(self.thresh_cfg.get("adaptive_c", 4))
            binary = cv2.adaptiveThreshold(channel, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, thresh_type, block, c)
        elif method == "adaptive_mean":
            block = int(self.thresh_cfg.get("adaptive_block_size", 31))
            if block % 2 == 0:
                block += 1
            c = float(self.thresh_cfg.get("adaptive_c", 4))
            binary = cv2.adaptiveThreshold(channel, 255, cv2.ADAPTIVE_THRESH_MEAN_C, thresh_type, block, c)
        else:
            val = float(self.thresh_cfg.get("manual_threshold", 120))
            _, binary = cv2.threshold(channel, val, 255, thresh_type)

        return binary

    def _apply_morphology(self, mask: np.ndarray) -> np.ndarray:
        open_k = self.morph_cfg.get("open_kernel_size", [5, 5])
        open_ksize = open_k[0] if isinstance(open_k, list) else int(open_k)
        open_iters = int(self.morph_cfg.get("open_iterations", 1))

        close_k = self.morph_cfg.get("close_kernel_size", [7, 7])
        close_ksize = close_k[0] if isinstance(close_k, list) else int(close_k)
        close_iters = int(self.morph_cfg.get("close_iterations", 2))

        cleaned = morphological_clean(
            mask,
            open_ksize=open_ksize,
            open_iters=open_iters,
            close_ksize=close_ksize,
            close_iters=close_iters,
        )

        if self.morph_cfg.get("fill_holes", True):
            cleaned = fill_holes(cleaned)

        return cleaned
