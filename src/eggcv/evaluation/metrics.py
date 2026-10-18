"""
Evaluation metrics for binary segmentation masks:
Intersection over Union (IoU / Jaccard), Dice coefficient (F1 score),
Pixel Precision, and Pixel Recall with well-defined empty-mask semantics.
"""

from typing import Dict, Tuple, Union

import numpy as np


def compute_confusion_matrix_elements(
    pred_mask: np.ndarray,
    ref_mask: np.ndarray,
) -> Tuple[int, int, int, int]:
    """
    Compute pixel-level TP, FP, FN, TN counts between predicted and reference binary masks.

    Args:
        pred_mask: Binary prediction mask (H, W)
        ref_mask: Binary reference mask (H, W)

    Returns:
        Tuple of (tp, fp, fn, tn) pixel counts
    """
    if pred_mask.shape != ref_mask.shape:
        raise ValueError(
            f"Shape mismatch: pred_mask {pred_mask.shape} vs ref_mask {ref_mask.shape}"
        )

    pred_bool = pred_mask > 0
    ref_bool = ref_mask > 0

    tp = int(np.count_nonzero(pred_bool & ref_bool))
    fp = int(np.count_nonzero(pred_bool & ~ref_bool))
    fn = int(np.count_nonzero(~pred_bool & ref_bool))
    tn = int(np.count_nonzero(~pred_bool & ~ref_bool))

    return tp, fp, fn, tn


def compute_iou(
    pred_mask: np.ndarray,
    ref_mask: np.ndarray,
    both_empty_val: float = 1.0,
) -> float:
    """
    Compute Intersection over Union (IoU / Jaccard Index).
    Formula: TP / (TP + FP + FN)

    Empty-mask convention:
      - If both pred and ref are completely empty: returns both_empty_val (default 1.0)
      - If one is empty and the other is not: returns 0.0
    """
    tp, fp, fn, _ = compute_confusion_matrix_elements(pred_mask, ref_mask)
    union = tp + fp + fn
    if union == 0:
        return both_empty_val
    return float(tp / union)


def compute_dice(
    pred_mask: np.ndarray,
    ref_mask: np.ndarray,
    both_empty_val: float = 1.0,
) -> float:
    """
    Compute Dice Coefficient (F1-score / Sorensen-Dice Index).
    Formula: 2 * TP / (2 * TP + FP + FN)

    Empty-mask convention:
      - If both pred and ref are completely empty: returns both_empty_val (default 1.0)
      - If one is empty and the other is not: returns 0.0
    """
    tp, fp, fn, _ = compute_confusion_matrix_elements(pred_mask, ref_mask)
    denom = 2 * tp + fp + fn
    if denom == 0:
        return both_empty_val
    return float((2.0 * tp) / denom)


def compute_precision_recall(
    pred_mask: np.ndarray,
    ref_mask: np.ndarray,
    zero_division_val: float = 0.0,
) -> Tuple[float, float]:
    """
    Compute pixel-level Precision and Recall.
    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)

    Empty-mask conventions:
      - Both empty: Precision=1.0, Recall=1.0 (perfect negative agreement)
      - Pred empty, Ref non-empty: Precision=0.0 (no true positives made), Recall=0.0
      - Pred non-empty, Ref empty: Precision=0.0 (all predictions are false alarms), Recall=0.0
    """
    tp, fp, fn, _ = compute_confusion_matrix_elements(pred_mask, ref_mask)

    if (tp + fp + fn) == 0:
        # Both masks are completely blank
        return 1.0, 1.0

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else zero_division_val
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else zero_division_val

    return precision, recall


def compute_all_metrics(
    pred_mask: np.ndarray,
    ref_mask: np.ndarray,
) -> Dict[str, float]:
    """
    Compute comprehensive metrics dictionary for an individual image.
    """
    tp, fp, fn, tn = compute_confusion_matrix_elements(pred_mask, ref_mask)
    iou = compute_iou(pred_mask, ref_mask)
    dice = compute_dice(pred_mask, ref_mask)
    precision, recall = compute_precision_recall(pred_mask, ref_mask)

    return {
        "iou": round(iou, 5),
        "dice": round(dice, 5),
        "precision": round(precision, 5),
        "recall": round(recall, 5),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
    }
