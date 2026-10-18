#!/usr/bin/env python3
"""
CLI script: Execute classical CV egg segmentation experiment on development split (Task 2).
"""

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd
from tqdm import tqdm

from eggcv.common.io import load_image, load_yaml, save_image, save_mask
from eggcv.common.logging_utils import get_logger
from eggcv.common.visualization import overlay_mask
from eggcv.dataset.inspect_dataset import DatasetAuditor
from eggcv.segmentation.classical_segmenter import ClassicalEggSegmenter

logger = get_logger("segmentation_experiment")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run classical egg segmentation experiment on development set.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "segmentation.yaml"),
        help="Path to classical segmentation configuration YAML",
    )
    parser.add_argument(
        "--dataset-config",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "dataset.yaml"),
        help="Path to dataset configuration YAML",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="development",
        choices=["development", "evaluation"],
        help="Split to process (Tuning MUST be performed strictly on development set)",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Limit number of images processed (for quick testing)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save predicted masks and overlays",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="exp_baseline_classical",
        help="Identifier for this experiment run",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    seg_cfg = load_yaml(args.config) if Path(args.config).is_file() else {}
    ds_cfg = load_yaml(args.dataset_config) if Path(args.dataset_config).is_file() else {}

    split_file = PROJECT_ROOT / ds_cfg.get("manifests", {}).get("split_file", "data/manifests/split.csv")
    if not split_file.is_file():
        logger.error(f"Split file missing at: {split_file}. Run 'python scripts/create_split.py' first.")
        sys.exit(1)

    split_df = pd.read_csv(split_file)
    target_subset = split_df[split_df["split"] == args.split].copy()

    if target_subset.empty:
        logger.error(f"No images found for split: '{args.split}'")
        sys.exit(1)

    if args.max_images:
        target_subset = target_subset.head(args.max_images)

    logger.info(f"Running experiment '{args.experiment_name}' on {len(target_subset)} images in '{args.split}' split.")

    segmenter = ClassicalEggSegmenter(seg_cfg)

    # Resolve dataset source
    archive_path = PROJECT_ROOT / ds_cfg.get("dataset", {}).get("archive_path", "../egg.v1i.folder.zip")
    auditor = DatasetAuditor(source_path=archive_path)

    # Output paths
    base_out = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "outputs" / "classical"
    masks_dir = base_out / "masks"
    overlays_dir = base_out / "overlays"
    for d in (masks_dir, overlays_dir):
        d.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    results = []

    for _, row in tqdm(target_subset.iterrows(), total=len(target_subset), desc="Segmenting Eggs"):
        img_id = str(row["image_id"])
        stem = Path(img_id).stem

        img = auditor.get_image(img_id)
        if img is None:
            logger.warning(f"Could not load image: {img_id}")
            continue

        mask, meta = segmenter.segment(img)

        # Save binary mask
        mask_path = masks_dir / f"{stem}.png"
        save_mask(mask, mask_path)

        # Save overlay
        overlay = overlay_mask(img, mask, color=(0, 220, 0), alpha=0.45)
        save_image(overlay, overlays_dir / f"{stem}_overlay.jpg")

        results.append({
            "image_id": img_id,
            "mask_path": str(mask_path),
            "num_detected_eggs": meta["num_detected_eggs"],
            "foreground_pixels": meta["total_foreground_pixels"],
        })

    elapsed_time = time.time() - start_time
    logger.info(f"Experiment completed in {elapsed_time:.2f} seconds ({len(results)} images).")
    logger.info(f"Masks written to: {masks_dir}")


if __name__ == "__main__":
    main()
