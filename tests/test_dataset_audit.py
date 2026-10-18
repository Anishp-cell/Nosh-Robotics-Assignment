"""
Unit tests for dataset auditing, corrupt image detection, and manifest creation.
"""

import zipfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from eggcv.dataset.inspect_dataset import DatasetAuditor


@pytest.fixture
def temp_dataset_dir(tmp_path: Path) -> Path:
    """Create a temporary dataset folder containing valid and corrupted images."""
    data_dir = tmp_path / "raw_data"
    sub_egg = data_dir / "train" / "egg"
    sub_egg.mkdir(parents=True, exist_ok=True)

    # 1. Valid image
    valid_img = np.full((64, 64, 3), 180, dtype=np.uint8)
    cv2.imwrite(str(sub_egg / "egg_valid_01.jpg"), valid_img)

    # 2. Another valid image
    valid_img2 = np.full((64, 64, 3), 120, dtype=np.uint8)
    cv2.imwrite(str(sub_egg / "egg_valid_02.jpg"), valid_img2)

    # 3. Corrupted image file (broken header bytes)
    corrupt_file = sub_egg / "egg_corrupted.jpg"
    with open(corrupt_file, "wb") as f:
        f.write(b"NOT_A_VALID_JPEG_HEADER_RANDOM_GARBAGE_BYTES")

    return data_dir


def test_audit_detects_corrupted_image(temp_dataset_dir: Path, tmp_path: Path):
    out_dir = tmp_path / "audit_out"
    auditor = DatasetAuditor(source_path=temp_dataset_dir)

    audit_df, duplicates_df, summary = auditor.run_audit(
        output_dir=out_dir,
        hamming_threshold=6,
        create_sheet=False,
    )

    assert len(audit_df) == 3
    assert summary["total_images"] == 3
    assert summary["corrupted_images_count"] == 1

    corrupted_rows = audit_df[audit_df["is_corrupted"]]
    assert len(corrupted_rows) == 1
    assert corrupted_rows.iloc[0]["image_id"] == "egg_corrupted.jpg"


def test_audit_zip_archive(tmp_path: Path):
    # Create a mock zip archive
    zip_path = tmp_path / "mock_eggs.zip"
    with zipfile.ZipFile(zip_path, "w") as z:
        # Add image 1
        img1 = np.full((64, 64, 3), 200, dtype=np.uint8)
        _, enc1 = cv2.imencode(".jpg", img1)
        z.writestr("train/egg/test_01.jpg", enc1.tobytes())

        # Add image 2 (exact duplicate of 1)
        z.writestr("valid/egg/test_02_dupe.jpg", enc1.tobytes())

        # Add README
        z.writestr("README.dataset.txt", "Roboflow mock dataset.")

    auditor = DatasetAuditor(source_path=zip_path)
    out_dir = tmp_path / "zip_audit_out"
    audit_df, duplicates_df, summary = auditor.run_audit(
        output_dir=out_dir,
        hamming_threshold=6,
        create_sheet=False,
    )

    assert summary["total_images"] == 2
    assert summary["corrupted_images_count"] == 0
    assert summary["exact_duplicate_pairs_count"] == 1
    # Both images should be grouped into the same duplicate group
    gids = audit_df["duplicate_group_id"].tolist()
    assert len(gids) == 2
    assert gids[0] == gids[1]
