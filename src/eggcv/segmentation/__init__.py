"""
Egg segmentation models: classical computer vision and Meta SAM.
"""

from eggcv.segmentation.classical_segmenter import BaseSegmenter, ClassicalEggSegmenter
from eggcv.segmentation.morphology import (
    fill_holes,
    filter_connected_components,
    morphological_clean,
    separate_touching_objects,
)
from eggcv.segmentation.sam_segmenter import SAMSegmenter

__all__ = [
    "BaseSegmenter",
    "ClassicalEggSegmenter",
    "SAMSegmenter",
    "morphological_clean",
    "fill_holes",
    "filter_connected_components",
    "separate_touching_objects",
]
