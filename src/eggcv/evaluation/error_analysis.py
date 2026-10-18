"""
Error analysis and failure-case visualization tools (Task 3).
Generates error galleries (TP, FP, FN) and score distributions.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eggcv.common.io import load_image, load_mask, save_image
from eggcv.common.logging_utils import get_logger
from eggcv.common.visualization import visualize_error_map

logger = get_logger("eggcv.evaluation.analysis")


def extract_top_failure_cases(
    metrics_df: pd.DataFrame,
    top_n: int = 10,
    sort_by: str = "iou",
    ascending: bool = True,
) -> pd.DataFrame:
    """
    Extract the top N worst-performing cases from the evaluation metrics DataFrame.
    """
    if metrics_df.empty or sort_by not in metrics_df.columns:
        return pd.DataFrame()
    return metrics_df.sort_values(by=sort_by, ascending=ascending).head(top_n).copy()


def generate_failure_case_gallery(
    worst_df: pd.DataFrame,
    images_dir_or_source: Union[str, Path],
    reference_masks_dir: Union[str, Path],
    predicted_masks_dir: Union[str, Path],
    output_dir: Union[str, Path],
) -> List[Path]:
    """
    Generate visual inspection error maps for worst failure cases.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ref_dir = Path(reference_masks_dir)
    pred_dir = Path(predicted_masks_dir)

    saved_paths: List[Path] = []

    for rank, (_, row) in enumerate(worst_df.iterrows(), start=1):
        img_id = str(row["image_id"])
        stem = Path(img_id).stem
        iou = float(row.get("iou", 0.0))

        ref_path = ref_dir / f"{stem}.png"
        pred_path = pred_dir / f"{stem}.png"

        if not ref_path.is_file() or not pred_path.is_file():
            continue

        ref_m = load_mask(ref_path)
        pred_m = load_mask(pred_path)

        # Try to load original image
        orig_img = None
        img_source_path = Path(images_dir_or_source) / img_id
        if img_source_path.is_file():
            try:
                orig_img = load_image(img_source_path)
            except Exception:
                orig_img = None

        error_map = visualize_error_map(ref_m, pred_m, image=orig_img)

        # Header with rank and IoU
        banner = np.zeros((30, error_map.shape[1], 3), dtype=np.uint8)
        text = f"Rank #{rank} Failure | {img_id[:25]} | IoU: {iou:.3f} | Dice: {row.get('dice', 0):.3f}"
        cv2.putText(banner, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        composite = np.vstack([banner, error_map])

        save_name = f"failure_rank_{rank:02d}_iou_{iou:.3f}_{stem[:12]}.jpg"
        save_path = out_dir / save_name
        save_image(composite, save_path)
        saved_paths.append(save_path)

    logger.info(f"Generated {len(saved_paths)} failure-case visualizations in: {out_dir}")
    return saved_paths


def plot_metric_distributions(
    metrics_df: pd.DataFrame,
    output_path: Union[str, Path],
    split_name: str = "development",
) -> Path:
    """
    Create distribution histograms for IoU and Dice scores.
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=150)

    # IoU distribution
    axes[0].hist(metrics_df["iou"], bins=20, range=(0, 1), color="#2b5c8f", edgecolor="black", alpha=0.85)
    axes[0].axvline(metrics_df["iou"].mean(), color="red", linestyle="--", linewidth=1.5, label=f"Mean: {metrics_df['iou'].mean():.3f}")
    axes[0].axvline(metrics_df["iou"].median(), color="orange", linestyle=":", linewidth=1.5, label=f"Median: {metrics_df['iou'].median():.3f}")
    axes[0].set_title(f"IoU Distribution ({split_name})", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Intersection over Union (IoU)")
    axes[0].set_ylabel("Image Count")
    axes[0].set_xlim(0, 1.0)
    axes[0].legend(loc="upper left")
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # Dice distribution
    axes[1].hist(metrics_df["dice"], bins=20, range=(0, 1), color="#2b8f5c", edgecolor="black", alpha=0.85)
    axes[1].axvline(metrics_df["dice"].mean(), color="red", linestyle="--", linewidth=1.5, label=f"Mean: {metrics_df['dice'].mean():.3f}")
    axes[1].axvline(metrics_df["dice"].median(), color="orange", linestyle=":", linewidth=1.5, label=f"Median: {metrics_df['dice'].median():.3f}")
    axes[1].set_title(f"Dice Coefficient Distribution ({split_name})", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Dice Score (F1)")
    axes[1].set_ylabel("Image Count")
    axes[1].set_xlim(0, 1.0)
    axes[1].legend(loc="upper left")
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(out_file, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved metric distribution plot to: {out_file}")
    return out_file
