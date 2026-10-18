"""
Segmentation evaluation engine (Task 3).
Evaluates predicted masks against reference masks strictly respecting the split boundary.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
from tqdm import tqdm

from eggcv.common.io import load_mask
from eggcv.common.logging_utils import get_logger
from eggcv.evaluation.metrics import compute_all_metrics

logger = get_logger("eggcv.evaluation.evaluate")


class SegmentationEvaluator:
    """Evaluates predicted binary segmentation masks against reference masks."""

    def __init__(
        self,
        reference_masks_dir: Union[str, Path],
        predicted_masks_dir: Union[str, Path],
        split_manifest_path: Union[str, Path],
    ):
        self.reference_dir = Path(reference_masks_dir)
        self.predicted_dir = Path(predicted_masks_dir)
        self.split_manifest = Path(split_manifest_path)

        if not self.split_manifest.is_file():
            raise FileNotFoundError(f"Split manifest not found: {self.split_manifest.resolve()}")

    def evaluate_split(
        self,
        split_name: str = "development",
        output_dir: Optional[Union[str, Path]] = None,
        max_images: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Run evaluation across images belonging to the specified split.

        Args:
            split_name: "development" or "evaluation"
            output_dir: Optional directory to save metrics CSV and summary JSON
            max_images: Optional limit for quick smoke testing

        Returns:
            Tuple of (per_image_df, summary_stats_dict)
        """
        df_split = pd.read_csv(self.split_manifest)
        if "split" not in df_split.columns:
            raise ValueError("Split manifest must have 'split' column.")

        subset = df_split[df_split["split"] == split_name].copy()
        if subset.empty:
            raise ValueError(f"No images found for split: '{split_name}'. Available: {df_split['split'].unique()}")

        if max_images is not None and max_images > 0:
            subset = subset.head(max_images)

        logger.info(f"Evaluating {len(subset)} images on '{split_name}' split...")

        results: List[Dict[str, Any]] = []

        for _, row in tqdm(subset.iterrows(), total=len(subset), desc=f"Evaluating {split_name}"):
            img_id = str(row["image_id"])
            stem = Path(img_id).stem

            # Reference mask path (supports stem.png or img_id.png)
            ref_path = self.reference_dir / f"{stem}.png"
            if not ref_path.exists():
                ref_path = self.reference_dir / f"{img_id}.png"

            # Predicted mask path
            pred_path = self.predicted_dir / f"{stem}.png"
            if not pred_path.exists():
                pred_path = self.predicted_dir / f"{img_id}.png"

            if not ref_path.is_file():
                logger.warning(f"Reference mask missing for {img_id} at {ref_path}")
                continue
            if not pred_path.is_file():
                logger.warning(f"Predicted mask missing for {img_id} at {pred_path}")
                continue

            ref_mask = load_mask(ref_path)
            pred_mask = load_mask(pred_path)

            metrics = compute_all_metrics(pred_mask, ref_mask)
            record = {
                "image_id": img_id,
                "split": split_name,
                "original_folder": row.get("original_folder", ""),
                "duplicate_group_id": row.get("duplicate_group_id", -1),
                **metrics,
            }
            results.append(record)

        if not results:
            logger.warning(f"No mask pairs were evaluated for split '{split_name}'.")
            return pd.DataFrame(), {}

        metrics_df = pd.DataFrame(results)

        # Aggregate summary
        summary: Dict[str, Any] = {
            "split": split_name,
            "images_evaluated": len(metrics_df),
            "mean_iou": round(float(metrics_df["iou"].mean()), 5),
            "median_iou": round(float(metrics_df["iou"].median()), 5),
            "std_iou": round(float(metrics_df["iou"].std()), 5),
            "mean_dice": round(float(metrics_df["dice"].mean()), 5),
            "median_dice": round(float(metrics_df["dice"].median()), 5),
            "mean_precision": round(float(metrics_df["precision"].mean()), 5),
            "mean_recall": round(float(metrics_df["recall"].mean()), 5),
        }

        if output_dir:
            out_p = Path(output_dir)
            out_p.mkdir(parents=True, exist_ok=True)
            csv_path = out_p / f"metrics_{split_name}.csv"
            json_path = out_p / f"summary_{split_name}.json"
            metrics_df.to_csv(csv_path, index=False)
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            logger.info(f"Saved evaluation metrics to: {csv_path}")

        return metrics_df, summary
