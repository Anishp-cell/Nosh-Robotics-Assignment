"""
Automated validation test for Task 1 outputs (Meta SAM reference segmentations).
Verifies file counts, mask dimensions, binary values, and manifest integrity.
"""

from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_task1_file_counts():
    """Verify that all 110 images have corresponding masks and overlays."""
    raw_zip = PROJECT_ROOT.parent / "egg.v1i.folder.zip"
    mask_dir = PROJECT_ROOT / "outputs" / "sam" / "binary_masks"
    overlay_dir = PROJECT_ROOT / "outputs" / "sam" / "overlays"
    instance_dir = PROJECT_ROOT / "outputs" / "sam" / "instance_masks"

    assert mask_dir.is_dir(), f"Missing binary masks directory: {mask_dir}"
    assert overlay_dir.is_dir(), f"Missing overlays directory: {overlay_dir}"

    bin_masks = list(mask_dir.glob("*.png"))
    overlays = list(overlay_dir.glob("*.jpg"))
    inst_masks = list(instance_dir.glob("*.png"))

    assert len(bin_masks) == 110, f"Expected 110 binary masks, got {len(bin_masks)}"
    assert len(overlays) == 110, f"Expected 110 overlays, got {len(overlays)}"
    assert len(inst_masks) == 690, f"Expected 690 instance masks, got {len(inst_masks)}"


def test_task1_binary_mask_format():
    """Verify binary masks are 640x640, single-channel, and strictly binary (0 or 255)."""
    mask_dir = PROJECT_ROOT / "outputs" / "sam" / "binary_masks"
    sample_masks = list(mask_dir.glob("*.png"))[:10]  # Check first 10

    for m_path in sample_masks:
        mask = cv2.imread(str(m_path), cv2.IMREAD_GRAYSCALE)
        assert mask is not None, f"Could not read mask: {m_path}"
        assert mask.shape == (640, 640), f"Incorrect shape {mask.shape} for {m_path.name}"
        unique_vals = set(np.unique(mask))
        assert unique_vals.issubset({0, 255}), f"Mask {m_path.name} contains non-binary values: {unique_vals}"


def test_task1_manifest_integrity():
    """Verify the SAM manifest contains 110 rows and required metadata columns."""
    manifest_path = PROJECT_ROOT / "outputs" / "sam" / "sam_manifest.csv"
    assert manifest_path.is_file(), f"Manifest file missing: {manifest_path}"

    df = pd.read_csv(manifest_path)
    assert len(df) == 110, f"Expected 110 manifest entries, got {len(df)}"

    required_cols = {"image_id", "mask_path", "num_proposals", "num_egg_instances", "foreground_pixels"}
    assert required_cols.issubset(set(df.columns)), f"Missing columns in manifest: {required_cols - set(df.columns)}"
    assert df["foreground_pixels"].min() > 0, "Found image with 0 foreground pixels in SAM mask"


def test_task1_qc_contact_sheet():
    """Verify the visual Quality Control contact sheet exists and is readable."""
    qc_sheet_path = PROJECT_ROOT / "outputs" / "sam" / "qc_gallery" / "sam_qc_contact_sheet.jpg"
    assert qc_sheet_path.is_file(), f"QC sheet missing: {qc_sheet_path}"

    img = cv2.imread(str(qc_sheet_path))
    assert img is not None, "Failed to decode QC contact sheet"
    assert img.shape[0] > 1000 and img.shape[1] > 1000, f"QC sheet dimensions too small: {img.shape}"
