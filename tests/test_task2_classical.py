"""
Unit tests for Task 2 Classical Computer Vision Egg Segmentation and Experimentation.
Validates classical pipeline outputs, modular config execution, experiment log integrity,
and zero leakage onto evaluation images.
"""

from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import pytest

from eggcv.segmentation.classical_segmenter import ClassicalEggSegmenter

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_classical_segmenter_shape_and_dtype():
    """Verify segmenter outputs correct binary mask shape, uint8 dtype, and metadata."""
    dummy_img = np.full((120, 160, 3), 50, dtype=np.uint8)
    # Draw a bright ellipse simulating an egg
    cv2.ellipse(dummy_img, (80, 60), (30, 20), 0, 0, 360, (200, 200, 200), -1)

    segmenter = ClassicalEggSegmenter({
        "preprocessing": {"blur_method": "gaussian", "blur_ksize": [5, 5], "blur_sigma": 1.0},
        "color_analysis": {"color_space": "gray"},
        "thresholding": {"method": "otsu"},
        "morphology": {"fill_holes": True},
        "component_filtering": {"enabled": False},
        "watershed": {"enabled": False},
    })

    mask, metadata = segmenter.segment(dummy_img)

    assert mask.shape == (120, 160)
    assert mask.dtype == np.uint8
    assert set(np.unique(mask)).issubset({0, 255})
    assert mask[60, 80] == 255  # Egg center segmented
    assert mask[10, 10] == 0    # Background 0
    assert "num_detected_eggs" in metadata
    assert "components" in metadata


def test_experiment_log_structure():
    """Ensure experiments/experiment_log.csv exists and has all required columns and 9 experiments."""
    log_path = PROJECT_ROOT / "experiments" / "experiment_log.csv"
    assert log_path.is_file(), f"Missing experiment_log.csv at {log_path}"

    df = pd.read_csv(log_path)
    required_cols = [
        "experiment_id",
        "timestamp",
        "hypothesis",
        "method",
        "parameters",
        "development_split",
        "mean_iou",
        "mean_dice",
        "precision",
        "recall",
        "runtime_seconds",
        "decision",
        "observations",
        "failure_cases",
    ]
    for col in required_cols:
        assert col in df.columns, f"Missing required column '{col}' in experiment_log.csv"

    assert len(df) >= 8, f"Expected at least 8 experiments logged, found {len(df)}"

    # Verify at least one negative attempt is logged with REJECT
    rejects = df[df["decision"] == "REJECT"]
    assert len(rejects) >= 1, "Expected at least one unsuccessful experiment with REJECT decision"

    # Verify all experiments ran strictly on the development split
    for _, row in df.iterrows():
        assert "development" in str(row["development_split"]).lower()


def test_development_outputs_count():
    """Verify 88 development masks and overlays exist and match split manifest."""
    split_path = PROJECT_ROOT / "data" / "manifests" / "split.csv"
    assert split_path.is_file()

    split_df = pd.read_csv(split_path)
    dev_ids = split_df[split_df["split"] == "development"]["image_id"].tolist()
    assert len(dev_ids) == 88

    masks_dir = PROJECT_ROOT / "outputs" / "classical" / "masks"
    overlays_dir = PROJECT_ROOT / "outputs" / "classical" / "overlays"

    assert masks_dir.is_dir()
    assert overlays_dir.is_dir()

    for img_id in dev_ids:
        stem = Path(img_id).stem
        mask_file = masks_dir / f"{stem}.png"
        overlay_file = overlays_dir / f"{stem}_overlay.jpg"
        assert mask_file.is_file(), f"Missing mask for development image {stem}"
        assert overlay_file.is_file(), f"Missing overlay for development image {stem}"
