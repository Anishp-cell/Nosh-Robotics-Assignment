import argparse
import datetime
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd
from tqdm import tqdm

from eggcv.common.io import load_image, load_mask, load_yaml, save_image, save_mask
from eggcv.common.logging_utils import get_logger
from eggcv.common.visualization import overlay_mask
from eggcv.dataset.inspect_dataset import DatasetAuditor
from eggcv.evaluation.metrics import compute_all_metrics
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
        "--reference-dir",
        type=str,
        default=str(PROJECT_ROOT / "outputs" / "sam" / "binary_masks"),
        help="Directory of reference SAM masks for development evaluation",
    )
    parser.add_argument(
        "--experiment-id",
        type=str,
        default=None,
        help="Unique experiment identifier (e.g. EXP-001-baseline-otsu)",
    )
    parser.add_argument(
        "--hypothesis",
        type=str,
        default="Baseline classical segmentation trial",
        help="Engineering hypothesis motivating the experiment",
    )
    parser.add_argument(
        "--method-description",
        type=str,
        default="",
        help="Summary of methods used",
    )
    parser.add_argument(
        "--parameters-summary",
        type=str,
        default="",
        help="Key hyperparameter string",
    )
    parser.add_argument(
        "--decision",
        type=str,
        default="ITERATE",
        choices=["ADOPT", "REJECT", "ITERATE"],
        help="Engineering decision for this experiment",
    )
    parser.add_argument(
        "--observations",
        type=str,
        default="",
        help="Key qualitative observations",
    )
    parser.add_argument(
        "--log-csv",
        type=str,
        default=str(PROJECT_ROOT / "experiments" / "experiment_log.csv"),
        help="Path to experiment log CSV",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        default=True,
        help="Evaluate against reference masks and compute IoU/Dice/Precision/Recall",
    )
    return parser.parse_args()


def run_experiment(
    seg_cfg: Dict[str, Any],
    split: str = "development",
    experiment_id: Optional[str] = None,
    hypothesis: str = "",
    method_description: str = "",
    parameters_summary: str = "",
    decision: str = "ITERATE",
    observations: str = "",
    output_dir: Optional[Path] = None,
    reference_dir: Optional[Path] = None,
    log_csv: Optional[Path] = None,
    max_images: Optional[int] = None,
    evaluate: bool = True,
) -> Dict[str, Any]:
    """
    Execute a classical segmentation experiment on the specified split,
    compute metrics against reference masks, and log results to CSV.
    """
    ds_cfg = load_yaml(PROJECT_ROOT / "configs" / "dataset.yaml")
    split_file = PROJECT_ROOT / ds_cfg.get("manifests", {}).get("split_file", "data/manifests/split.csv")
    if not split_file.is_file():
        raise FileNotFoundError(f"Split file missing at: {split_file}")

    split_df = pd.read_csv(split_file)
    target_subset = split_df[split_df["split"] == split].copy()

    if target_subset.empty:
        raise ValueError(f"No images found for split: '{split}'")

    if max_images:
        target_subset = target_subset.head(max_images)

    logger.info(f"Running experiment '{experiment_id or 'ad-hoc'}' on {len(target_subset)} images in '{split}' split.")

    segmenter = ClassicalEggSegmenter(seg_cfg)

    archive_path = PROJECT_ROOT / ds_cfg.get("dataset", {}).get("archive_path", "../egg.v1i.folder.zip")
    auditor = DatasetAuditor(source_path=archive_path)

    base_out = output_dir or PROJECT_ROOT / "outputs" / "classical"
    masks_dir = base_out / "masks"
    overlays_dir = base_out / "overlays"
    for d in (masks_dir, overlays_dir):
        d.mkdir(parents=True, exist_ok=True)

    ref_p = reference_dir or PROJECT_ROOT / "outputs" / "sam" / "binary_masks"

    start_time = time.time()
    per_image_results = []

    for _, row in tqdm(target_subset.iterrows(), total=len(target_subset), desc=f"Segmenting ({split})"):
        img_id = str(row["image_id"])
        stem = Path(img_id).stem

        img = auditor.get_image(img_id)
        if img is None:
            logger.warning(f"Could not load image: {img_id}")
            continue

        mask, meta = segmenter.segment(img)

        mask_path = masks_dir / f"{stem}.png"
        save_mask(mask, mask_path)

        overlay = overlay_mask(img, mask, color=(0, 220, 0), alpha=0.45)
        save_image(overlay, overlays_dir / f"{stem}_overlay.jpg")

        record = {
            "image_id": img_id,
            "mask_path": str(mask_path),
            "num_detected_eggs": meta["num_detected_eggs"],
            "foreground_pixels": meta["total_foreground_pixels"],
        }

        if evaluate and ref_p.is_dir():
            ref_path = ref_p / f"{stem}.png"
            if not ref_path.is_file():
                ref_path = ref_p / f"{img_id}.png"
            if ref_path.is_file():
                ref_mask = load_mask(ref_path)
                metrics = compute_all_metrics(mask, ref_mask)
                record.update(metrics)

        per_image_results.append(record)

    runtime_seconds = round(time.time() - start_time, 2)
    logger.info(f"Experiment completed in {runtime_seconds:.2f} seconds ({len(per_image_results)} images).")

    res_df = pd.DataFrame(per_image_results)

    summary: Dict[str, Any] = {
        "experiment_id": experiment_id or "EXP-adhoc",
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "split": split,
        "images_evaluated": len(res_df),
        "runtime_seconds": runtime_seconds,
    }

    if evaluate and "iou" in res_df.columns:
        mean_iou = round(float(res_df["iou"].mean()), 4)
        mean_dice = round(float(res_df["dice"].mean()), 4)
        mean_prec = round(float(res_df["precision"].mean()), 4)
        mean_rec = round(float(res_df["recall"].mean()), 4)

        summary.update({
            "mean_iou": mean_iou,
            "mean_dice": mean_dice,
            "precision": mean_prec,
            "recall": mean_rec,
        })

        worst_images = (
            res_df.sort_values(by="iou", ascending=True)
            .head(3)["image_id"]
            .apply(lambda x: Path(x).stem)
            .tolist()
        )
        summary["failure_cases"] = ";".join(worst_images)

        print("\n" + "=" * 65)
        print(f"EXPERIMENT RESULTS: {summary['experiment_id']} ({split.upper()})")
        print("=" * 65)
        print(f"Images Evaluated : {summary['images_evaluated']}")
        print(f"Mean IoU         : {mean_iou:.4f}")
        print(f"Mean Dice (F1)   : {mean_dice:.4f}")
        print(f"Mean Precision   : {mean_prec:.4f}")
        print(f"Mean Recall      : {mean_rec:.4f}")
        print(f"Runtime (sec)    : {runtime_seconds:.2f}s")
        print(f"Worst Failures   : {summary['failure_cases']}")
        print("=" * 65 + "\n")

        # Append to experiment_log.csv if experiment_id is provided
        log_path = log_csv or PROJECT_ROOT / "experiments" / "experiment_log.csv"
        if experiment_id and log_path:
            log_path = Path(log_path)
            log_entry = {
                "experiment_id": experiment_id,
                "timestamp": summary["timestamp"],
                "hypothesis": hypothesis,
                "method": method_description or seg_cfg.get("thresholding", {}).get("method", "classical"),
                "parameters": parameters_summary or str(seg_cfg),
                "development_split": f"{split}-{len(res_df)}",
                "mean_iou": mean_iou,
                "mean_dice": mean_dice,
                "precision": mean_prec,
                "recall": mean_rec,
                "runtime_seconds": runtime_seconds,
                "decision": decision,
                "observations": observations,
                "failure_cases": summary["failure_cases"],
            }
            log_df = pd.DataFrame([log_entry])
            if log_path.is_file():
                log_df.to_csv(log_path, mode="a", header=False, index=False)
            else:
                log_df.to_csv(log_path, index=False)
            logger.info(f"Experiment logged to: {log_path}")

    return summary


def main():
    args = parse_args()
    seg_cfg = load_yaml(args.config) if Path(args.config).is_file() else {}

    output_dir = Path(args.output_dir) if args.output_dir else None
    ref_dir = Path(args.reference_dir) if args.reference_dir else None
    log_csv = Path(args.log_csv) if args.log_csv else None

    run_experiment(
        seg_cfg=seg_cfg,
        split=args.split,
        experiment_id=args.experiment_id,
        hypothesis=args.hypothesis,
        method_description=args.method_description,
        parameters_summary=args.parameters_summary,
        decision=args.decision,
        observations=args.observations,
        output_dir=output_dir,
        reference_dir=ref_dir,
        log_csv=log_csv,
        max_images=args.max_images,
        evaluate=args.evaluate,
    )


if __name__ == "__main__":
    main()
