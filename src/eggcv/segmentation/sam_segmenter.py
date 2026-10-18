"""
Meta Segment Anything Model (SAM) integration interface (Task 1).
Provides transparent dependency inspection, checkpoint verification,
and mask generation scaffolding.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch

from eggcv.common.logging_utils import get_logger
from eggcv.segmentation.classical_segmenter import BaseSegmenter
from eggcv.segmentation.morphology import filter_connected_components

logger = get_logger("eggcv.segmentation.sam")


class SAMSegmenter(BaseSegmenter):
    """
    Interface for Meta's Segment Anything Model (SAM).
    Manages model loading, device selection, automatic mask generation,
    and geometric candidate filtering.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.model_cfg = self.config.get("model", {})
        self.gen_cfg = self.config.get("generator", {})
        self.filter_cfg = self.config.get("filtering", {})

        self.model_type = self.model_cfg.get("type", "vit_b")
        self.checkpoint_path = Path(self.model_cfg.get("checkpoint_path", "checkpoints/sam_vit_b_01ec64.pth"))
        self.device = self._resolve_device(self.model_cfg.get("device", "cuda"))

        self.sam = None
        self.mask_generator = None

    @staticmethod
    def _resolve_device(requested_device: str) -> str:
        if requested_device == "cuda":
            if torch.cuda.is_available():
                return "cuda"
            logger.warning("CUDA was requested but is not available. Falling back to CPU.")
            return "cpu"
        return "cpu"

    def is_available(self) -> Tuple[bool, str]:
        """
        Check if SAM package and model weights checkpoint are installed and present.

        Returns:
            Tuple of (is_ready: bool, message: str)
        """
        try:
            import segment_anything  # noqa: F401
        except ImportError:
            return False, (
                "The 'segment-anything' package is not installed. "
                "Install it using: pip install git+https://github.com/facebookresearch/segment-anything.git"
            )

        if not self.checkpoint_path.is_file():
            return False, (
                f"SAM checkpoint weight file not found at: {self.checkpoint_path.resolve()}\n"
                f"Download the ViT-B checkpoint ({self.model_type}) using:\n"
                f"  python scripts/generate_sam_masks.py --download-checkpoint\n"
                f"or directly from: {self.model_cfg.get('checkpoint_url', 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth')}"
            )

        return True, "SAM is installed and checkpoint weights are verified."

    def load_model(self) -> None:
        """
        Instantiate SAM model and automatic mask generator.
        Raises descriptive error if dependencies or checkpoints are missing.
        """
        ready, msg = self.is_available()
        if not ready:
            raise RuntimeError(f"Cannot load SAM model: {msg}")

        from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

        logger.info(f"Loading SAM ({self.model_type}) from {self.checkpoint_path} onto {self.device}...")
        self.sam = sam_model_registry[self.model_type](checkpoint=str(self.checkpoint_path))
        self.sam.to(device=self.device)

        self.mask_generator = SamAutomaticMaskGenerator(
            model=self.sam,
            points_per_side=int(self.gen_cfg.get("points_per_side", 32)),
            pred_iou_thresh=float(self.gen_cfg.get("pred_iou_thresh", 0.88)),
            stability_score_thresh=float(self.gen_cfg.get("stability_score_thresh", 0.95)),
            min_mask_region_area=int(self.gen_cfg.get("min_mask_region_area", 200)),
        )
        logger.info("SAM model initialized successfully.")

    def segment(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Run SAM mask generation and geometric candidate filtering.

        Args:
            image: BGR image array (H, W, 3) uint8

        Returns:
            Tuple of:
                - binary_mask: Consolidated egg foreground mask (uint8 0/255)
                - metadata: Dict with proposal count, egg instance masks, and IoU scores
        """
        if self.mask_generator is None:
            self.load_model()

        # SAM expects RGB format
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        proposals = self.mask_generator.generate(rgb_image)

        h, w = image.shape[:2]
        composite_mask = np.zeros((h, w), dtype=np.uint8)
        egg_instances: List[Dict[str, Any]] = []

        min_area = float(self.filter_cfg.get("min_egg_area", 2000.0))
        max_area = float(self.filter_cfg.get("max_egg_area", 320000.0))
        min_circ = float(self.filter_cfg.get("min_circularity", 0.30))

        for prop in proposals:
            m = prop["segmentation"].astype(np.uint8) * 255
            area = float(prop.get("area", np.count_nonzero(m)))

            if area < min_area or area > max_area:
                continue

            # Check circularity of proposal
            filtered, comps = filter_connected_components(m, min_area=min_area, max_area=max_area, min_circularity=min_circ)
            if np.any(filtered):
                composite_mask[filtered > 0] = 255
                egg_instances.append({
                    "bbox": prop.get("bbox"),
                    "predicted_iou": float(prop.get("predicted_iou", 0.0)),
                    "stability_score": float(prop.get("stability_score", 0.0)),
                    "area": area,
                })

        metadata = {
            "num_proposals": len(proposals),
            "num_egg_instances": len(egg_instances),
            "egg_instances": egg_instances,
        }

        return composite_mask, metadata
