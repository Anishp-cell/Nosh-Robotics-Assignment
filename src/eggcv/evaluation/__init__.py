"""
Evaluation and comparative analysis package.
"""

from eggcv.evaluation.error_analysis import (
    extract_top_failure_cases,
    generate_failure_case_gallery,
    plot_metric_distributions,
)
from eggcv.evaluation.evaluate import SegmentationEvaluator
from eggcv.evaluation.metrics import (
    compute_all_metrics,
    compute_dice,
    compute_iou,
    compute_precision_recall,
)

__all__ = [
    "compute_iou",
    "compute_dice",
    "compute_precision_recall",
    "compute_all_metrics",
    "SegmentationEvaluator",
    "extract_top_failure_cases",
    "generate_failure_case_gallery",
    "plot_metric_distributions",
]
