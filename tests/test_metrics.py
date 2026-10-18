"""
Unit tests for segmentation evaluation metrics: IoU, Dice, Precision, and Recall.
"""

import numpy as np
import pytest

from eggcv.evaluation.metrics import (
    compute_all_metrics,
    compute_dice,
    compute_iou,
    compute_precision_recall,
)


def test_perfect_overlap():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:70, 20:70] = 255  # 50x50 foreground = 2500 px

    iou = compute_iou(mask, mask)
    dice = compute_dice(mask, mask)
    prec, rec = compute_precision_recall(mask, mask)

    assert iou == pytest.approx(1.0, abs=1e-5)
    assert dice == pytest.approx(1.0, abs=1e-5)
    assert prec == pytest.approx(1.0, abs=1e-5)
    assert rec == pytest.approx(1.0, abs=1e-5)


def test_partial_overlap():
    pred = np.zeros((100, 100), dtype=np.uint8)
    ref = np.zeros((100, 100), dtype=np.uint8)

    pred[0:50, 0:50] = 255     # Area: 2500
    ref[0:50, 25:75] = 255     # Area: 2500
    # Overlap is [0:50, 25:50] -> Area: 1250
    # Union is [0:50, 0:75]   -> Area: 3750

    iou = compute_iou(pred, ref)
    dice = compute_dice(pred, ref)
    prec, rec = compute_precision_recall(pred, ref)

    assert iou == pytest.approx(1250 / 3750, abs=1e-4)   # ~0.3333
    assert dice == pytest.approx(2500 / 5000, abs=1e-4)  # 0.5
    assert prec == pytest.approx(1250 / 2500, abs=1e-4)  # 0.5
    assert rec == pytest.approx(1250 / 2500, abs=1e-4)   # 0.5


def test_completely_disjoint():
    pred = np.zeros((100, 100), dtype=np.uint8)
    ref = np.zeros((100, 100), dtype=np.uint8)

    pred[0:20, 0:20] = 255
    ref[80:100, 80:100] = 255

    iou = compute_iou(pred, ref)
    dice = compute_dice(pred, ref)
    prec, rec = compute_precision_recall(pred, ref)

    assert iou == 0.0
    assert dice == 0.0
    assert prec == 0.0
    assert rec == 0.0


def test_both_empty_masks():
    pred = np.zeros((100, 100), dtype=np.uint8)
    ref = np.zeros((100, 100), dtype=np.uint8)

    iou = compute_iou(pred, ref)
    dice = compute_dice(pred, ref)
    prec, rec = compute_precision_recall(pred, ref)

    # Empty reference + empty prediction = perfect true negative match
    assert iou == 1.0
    assert dice == 1.0
    assert prec == 1.0
    assert rec == 1.0


def test_prediction_empty_reference_nonempty():
    pred = np.zeros((100, 100), dtype=np.uint8)
    ref = np.zeros((100, 100), dtype=np.uint8)
    ref[10:50, 10:50] = 255

    iou = compute_iou(pred, ref)
    dice = compute_dice(pred, ref)
    prec, rec = compute_precision_recall(pred, ref)

    assert iou == 0.0
    assert dice == 0.0
    assert prec == 0.0
    assert rec == 0.0


def test_prediction_nonempty_reference_empty():
    pred = np.zeros((100, 100), dtype=np.uint8)
    pred[10:50, 10:50] = 255
    ref = np.zeros((100, 100), dtype=np.uint8)

    iou = compute_iou(pred, ref)
    dice = compute_dice(pred, ref)
    prec, rec = compute_precision_recall(pred, ref)

    assert iou == 0.0
    assert dice == 0.0
    assert prec == 0.0
    assert rec == 0.0


def test_shape_mismatch_raises_error():
    pred = np.zeros((100, 100), dtype=np.uint8)
    ref = np.zeros((50, 50), dtype=np.uint8)

    with pytest.raises(ValueError):
        compute_iou(pred, ref)


def test_compute_all_metrics_dict():
    pred = np.zeros((64, 64), dtype=np.uint8)
    ref = np.zeros((64, 64), dtype=np.uint8)
    pred[10:30, 10:30] = 255
    ref[10:30, 10:30] = 255

    metrics = compute_all_metrics(pred, ref)
    assert "iou" in metrics
    assert "dice" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert metrics["iou"] == 1.0
    assert metrics["true_positives"] == 400
    assert metrics["false_positives"] == 0
    assert metrics["false_negatives"] == 0
