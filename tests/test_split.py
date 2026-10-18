"""
Unit tests for leakage-aware reproducible splitting and duplicate group constraints.
"""

from pathlib import Path

import pandas as pd
import pytest

from eggcv.dataset.create_split import SplitManager


@pytest.fixture
def mock_audit_csv(tmp_path: Path) -> Path:
    """Create a mock dataset_audit.csv with duplicate groups."""
    csv_path = tmp_path / "mock_audit.csv"

    records = []
    # 20 total images:
    # Group 1: 3 images (duplicates)
    for i in range(1, 4):
        records.append({
            "image_id": f"img_g1_{i}.jpg",
            "original_folder": "train/egg",
            "duplicate_group_id": 1,
            "is_corrupted": False,
            "sha256": "hash_g1",
            "width": 640,
            "height": 640,
        })
    # Group 2: 2 images (near-duplicates)
    for i in range(1, 3):
        records.append({
            "image_id": f"img_g2_{i}.jpg",
            "original_folder": "train/egg",
            "duplicate_group_id": 2,
            "is_corrupted": False,
            "sha256": f"hash_g2_{i}",
            "width": 640,
            "height": 640,
        })
    # Group 3 to 17: singletons (15 images)
    for gid in range(3, 18):
        records.append({
            "image_id": f"img_single_{gid}.jpg",
            "original_folder": "valid/egg",
            "duplicate_group_id": gid,
            "is_corrupted": False,
            "sha256": f"hash_single_{gid}",
            "width": 640,
            "height": 640,
        })

    df = pd.DataFrame(records)
    df.to_csv(csv_path, index=False)
    return csv_path


def test_zero_leakage_across_groups(mock_audit_csv: Path, tmp_path: Path):
    manager = SplitManager(random_seed=42, target_dev_ratio=0.80)
    out_csv = tmp_path / "split_test.csv"

    split_df, summary = manager.create_split(
        audit_csv_path=mock_audit_csv,
        output_csv_path=out_csv,
        force=True,
    )

    # Assert that all images in Group 1 are in the same split
    g1_splits = split_df[split_df["duplicate_group_id"] == 1]["split"].unique()
    assert len(g1_splits) == 1

    # Assert that all images in Group 2 are in the same split
    g2_splits = split_df[split_df["duplicate_group_id"] == 2]["split"].unique()
    assert len(g2_splits) == 1

    # Assert every group is atomic across entire split
    leakage = split_df.groupby("duplicate_group_id")["split"].nunique()
    assert (leakage == 1).all()


def test_split_reproducibility(mock_audit_csv: Path, tmp_path: Path):
    manager1 = SplitManager(random_seed=42)
    manager2 = SplitManager(random_seed=42)

    out1 = tmp_path / "split_run1.csv"
    out2 = tmp_path / "split_run2.csv"

    df1, _ = manager1.create_split(mock_audit_csv, out1, force=True)
    df2, _ = manager2.create_split(mock_audit_csv, out2, force=True)

    pd.testing.assert_frame_equal(df1, df2)


def test_overwrite_protection(mock_audit_csv: Path, tmp_path: Path):
    manager = SplitManager(random_seed=42)
    out_csv = tmp_path / "split_protected.csv"

    manager.create_split(mock_audit_csv, out_csv, force=True)
    assert out_csv.exists()

    # Second call without force=True must raise FileExistsError
    with pytest.raises(FileExistsError):
        manager.create_split(mock_audit_csv, out_csv, force=False)
