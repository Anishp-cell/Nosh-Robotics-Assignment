"""
Dataset inspection, duplicate detection, and splitting tools.
"""

from eggcv.dataset.create_split import SplitManager
from eggcv.dataset.duplicate_detection import (
    build_duplicate_groups,
    compute_ahash,
    compute_dhash,
    compute_phash,
    find_duplicate_candidates,
    hamming_distance,
)
from eggcv.dataset.inspect_dataset import DatasetAuditor

__all__ = [
    "DatasetAuditor",
    "SplitManager",
    "compute_dhash",
    "compute_ahash",
    "compute_phash",
    "hamming_distance",
    "find_duplicate_candidates",
    "build_duplicate_groups",
]
