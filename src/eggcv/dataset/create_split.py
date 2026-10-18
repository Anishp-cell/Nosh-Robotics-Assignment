"""
Leakage-aware, reproducible train/development and evaluation splitting utility.
Guarantees zero-leakage by keeping duplicate and near-duplicate groups strictly intact.
"""

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

from eggcv.common.logging_utils import get_logger

logger = get_logger("eggcv.dataset.split")


class SplitManager:
    """Creates and validates reproducible 80:20 development/evaluation splits."""

    def __init__(self, random_seed: int = 42, target_dev_ratio: float = 0.80):
        self.random_seed = random_seed
        self.target_dev_ratio = target_dev_ratio

    def create_split(
        self,
        audit_csv_path: Union[str, Path],
        output_csv_path: Union[str, Path],
        force: bool = False,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Partition images into development and evaluation sets using group constraints.

        Args:
            audit_csv_path: Path to dataset_audit.csv produced by DatasetAuditor
            output_csv_path: Target path for split.csv
            force: If False, prevents overwriting existing split file

        Returns:
            Tuple of (split_df, summary_dict)
        """
        output_csv = Path(output_csv_path)
        if output_csv.exists() and not force:
            raise FileExistsError(
                f"Split manifest already exists at {output_csv.resolve()}. "
                "Use --force to overwrite an existing split."
            )

        audit_path = Path(audit_csv_path)
        if not audit_path.is_file():
            raise FileNotFoundError(f"Audit CSV not found: {audit_path.resolve()}")

        df = pd.read_csv(audit_path)
        if "duplicate_group_id" not in df.columns:
            raise ValueError("Audit CSV must contain 'duplicate_group_id' column.")

        # Filter out any corrupted files if found
        if "is_corrupted" in df.columns:
            valid_df = df[~df["is_corrupted"]].copy()
            corrupted_count = int(df["is_corrupted"].sum())
        else:
            valid_df = df.copy()
            corrupted_count = 0

        total_images = len(valid_df)
        target_dev_count = round(total_images * self.target_dev_ratio)
        target_eval_count = total_images - target_dev_count

        logger.info(
            f"Partitioning {total_images} valid images into ~{self.target_dev_ratio*100:.0f}% dev "
            f"({target_dev_count}) and ~{(1-self.target_dev_ratio)*100:.0f}% eval ({target_eval_count}) "
            f"with random seed {self.random_seed}."
        )

        # Aggregate by duplicate groups
        group_sizes: Dict[int, List[str]] = {}
        for _, row in valid_df.iterrows():
            gid = int(row["duplicate_group_id"])
            group_sizes.setdefault(gid, []).append(str(row["image_id"]))

        # Shuffle groups deterministically
        rng = random.Random(self.random_seed)
        unique_groups = sorted(list(group_sizes.keys()))
        rng.shuffle(unique_groups)

        dev_groups: List[int] = []
        eval_groups: List[int] = []
        dev_images: List[str] = []
        eval_images: List[str] = []

        # Greedy allocation targeting the development count while keeping groups atomic
        for gid in unique_groups:
            imgs = group_sizes[gid]
            if len(dev_images) + len(imgs) <= target_dev_count:
                dev_groups.append(gid)
                dev_images.extend(imgs)
            elif len(eval_images) + len(imgs) <= target_eval_count:
                eval_groups.append(gid)
                eval_images.extend(imgs)
            else:
                # Add to whichever side is farther from its target
                dev_deficit = target_dev_count - len(dev_images)
                eval_deficit = target_eval_count - len(eval_images)
                if dev_deficit >= eval_deficit:
                    dev_groups.append(gid)
                    dev_images.extend(imgs)
                else:
                    eval_groups.append(gid)
                    eval_images.extend(imgs)

        # Build split mapping
        split_map: Dict[str, str] = {}
        for img_id in dev_images:
            split_map[img_id] = "development"
        for img_id in eval_images:
            split_map[img_id] = "evaluation"

        valid_df["split"] = valid_df["image_id"].map(split_map)

        # STRICT VALIDATION: Check for leakage across groups
        self._validate_zero_leakage(valid_df)

        actual_dev_count = len(dev_images)
        actual_eval_count = len(eval_images)

        summary: Dict[str, Any] = {
            "random_seed": self.random_seed,
            "target_dev_count": target_dev_count,
            "target_eval_count": target_eval_count,
            "actual_dev_count": actual_dev_count,
            "actual_eval_count": actual_eval_count,
            "actual_dev_ratio": round(actual_dev_count / total_images, 4),
            "actual_eval_ratio": round(actual_eval_count / total_images, 4),
            "total_images": total_images,
            "corrupted_images_excluded": corrupted_count,
            "duplicate_groups_total": len(unique_groups),
            "duplicate_groups_in_dev": len(dev_groups),
            "duplicate_groups_in_eval": len(eval_groups),
            "folder_distribution": valid_df.groupby(["original_folder", "split"]).size().unstack(fill_value=0).to_dict(),
        }

        # Save outputs
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        cols_to_save = [
            "image_id",
            "relative_path",
            "original_folder",
            "duplicate_group_id",
            "split",
            "sha256",
            "file_size_bytes",
            "width",
            "height",
        ]
        available_cols = [c for c in cols_to_save if c in valid_df.columns]
        valid_df[available_cols].to_csv(output_csv, index=False)

        summary_json_path = output_csv.parent / "split_summary.json"
        with open(summary_json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        logger.info(
            f"Split created successfully: {actual_dev_count} dev, {actual_eval_count} eval. "
            f"Zero-leakage validated across {len(unique_groups)} groups."
        )
        logger.info(f"Saved split manifest: {output_csv.resolve()}")

        return valid_df, summary

    @staticmethod
    def _validate_zero_leakage(df: pd.DataFrame) -> None:
        """
        Asserts that every duplicate group exists strictly in one split (zero leakage).
        Raises AssertionError if any group spans both development and evaluation.
        """
        group_split_counts = df.groupby("duplicate_group_id")["split"].nunique()
        leaking_groups = group_split_counts[group_split_counts > 1]

        if not leaking_groups.empty:
            leak_ids = leaking_groups.index.tolist()
            raise AssertionError(
                f"DATA LEAKAGE DETECTED! The following duplicate groups span both splits: {leak_ids}"
            )
        logger.info("Validation passed: ZERO leakage across all duplicate groups.")
