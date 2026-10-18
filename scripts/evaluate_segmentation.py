#!/usr/bin/env python3
"""
CLI script: Evaluate segmentation predictions against SAM reference masks (Task 3).
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from eggcv.common.io import load_yaml
from eggcv.common.logging_utils import get_logger
from eggcv.evaluation.error_analysis import (
    extract_top_failure_cases,
    generate_failure_case_gallery,
    plot_metric_distributions,
)
from eggcv.evaluation.evaluate import SegmentationEvaluator

logger = get_logger("evaluate_segmentation")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate classical egg segmentation masks against SAM reference masks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "evaluation.yaml"),
        help="Path to evaluation configuration YAML",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        choices=["development", "evaluation"],
        help="Split to evaluate ('development' for tuning; 'evaluation' ONLY for final benchmarking)",
    )
    parser.add_argument(
        "--reference-dir",
        type=str,
        default=None,
        help="Directory containing reference masks (e.g. outputs/sam/binary_masks)",
    )
    parser.add_argument(
        "--predicted-dir",
        type=str,
        default=None,
        help="Directory containing predicted masks (e.g. outputs/classical/masks)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save evaluation results, failure galleries, and figures",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Limit number of images evaluated (for quick testing)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_yaml(args.config) if Path(args.config).is_file() else {}

    split_name = args.split or cfg.get("target_split", "development")
    ref_dir = args.reference_dir or PROJECT_ROOT / cfg.get("inputs", {}).get("reference_masks_dir", "outputs/sam/binary_masks")
    pred_dir = args.predicted_dir or PROJECT_ROOT / cfg.get("inputs", {}).get("predicted_masks_dir", "outputs/classical/masks")
    manifest_p = PROJECT_ROOT / cfg.get("inputs", {}).get("split_manifest", "data/manifests/split.csv")
    out_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "outputs" / "evaluation"

    evaluator = SegmentationEvaluator(
        reference_masks_dir=ref_dir,
        predicted_masks_dir=pred_dir,
        split_manifest_path=manifest_p,
    )

    metrics_df, summary = evaluator.evaluate_split(
        split_name=split_name,
        output_dir=out_dir,
        max_images=args.max_images,
    )

    if metrics_df.empty:
        logger.warning("No metrics were generated. Ensure both reference and predicted masks exist.")
        return

    # Error analysis and visualizations
    failures = extract_top_failure_cases(metrics_df, top_n=cfg.get("analysis", {}).get("top_n_failure_cases", 10))
    if not failures.empty:
        failures_dir = out_dir / "failure_cases"
        generate_failure_case_gallery(
            worst_df=failures,
            images_dir_or_source=PROJECT_ROOT / "data" / "raw",
            reference_masks_dir=ref_dir,
            predicted_masks_dir=pred_dir,
            output_dir=failures_dir,
        )

    # Plot metric distributions
    fig_dir = PROJECT_ROOT / "outputs" / "figures"
    dist_plot_path = fig_dir / f"metric_distributions_{split_name}.png"
    plot_metric_distributions(metrics_df, output_path=dist_plot_path, split_name=split_name)

    print("\n" + "=" * 65)
    print(f"EVALUATION SUMMARY ({split_name.upper()} SET)")
    print("=" * 65)
    print(f"Images evaluated : {summary['images_evaluated']}")
    print(f"Mean IoU         : {summary['mean_iou']:.4f} (std: {summary['std_iou']:.4f})")
    print(f"Median IoU       : {summary['median_iou']:.4f}")
    print(f"Mean Dice (F1)   : {summary['mean_dice']:.4f} (median: {summary['median_dice']:.4f})")
    print(f"Mean Precision   : {summary['mean_precision']:.4f}")
    print(f"Mean Recall      : {summary['mean_recall']:.4f}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
