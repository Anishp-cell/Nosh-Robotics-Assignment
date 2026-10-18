#!/usr/bin/env python3
"""
CLI script: Audit egg dataset V1, detect exact and perceptual duplicates,
and generate contact sheet and audit manifests.
"""

import argparse
import sys
from pathlib import Path

# Add src to path for direct invocation
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from eggcv.common.io import load_yaml
from eggcv.common.logging_utils import get_logger
from eggcv.dataset.inspect_dataset import DatasetAuditor

logger = get_logger("audit_dataset")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Audit Roboflow Egg Dataset V1, verify integrity, and detect duplicates.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "dataset.yaml"),
        help="Path to dataset configuration YAML file",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to raw ZIP archive or extracted dataset directory (overrides config archive_path)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Directory to save audit CSV, summary JSON, and contact sheet",
    )
    parser.add_argument(
        "--hamming-threshold",
        type=int,
        default=6,
        help="Hamming distance threshold for flagging near-duplicate candidate images",
    )
    parser.add_argument(
        "--no-contact-sheet",
        action="store_true",
        help="Skip generating visual contact sheet",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_yaml(args.config) if Path(args.config).is_file() else {}

    # Resolve input path
    input_path = args.input
    if not input_path:
        rel_archive = cfg.get("dataset", {}).get("archive_path", "../egg.v1i.folder.zip")
        # Check relative to project root or cwd
        p1 = PROJECT_ROOT / rel_archive
        p2 = Path(rel_archive)
        input_path = str(p1) if p1.exists() else str(p2)

    input_p = Path(input_path).resolve()
    if not input_p.exists():
        logger.error(
            f"Dataset archive not found at: {input_p}.\n"
            f"Please ensure egg.v1i.folder.zip is placed in the workspace or specify --input."
        )
        sys.exit(1)

    # Resolve output directory
    output_dir = args.output
    if not output_dir:
        output_dir = str(PROJECT_ROOT / cfg.get("outputs", {}).get("audit_dir", "outputs/dataset_audit"))

    threshold = args.hamming_threshold or cfg.get("duplicate_detection", {}).get("hamming_distance_threshold", 6)

    auditor = DatasetAuditor(source_path=input_p)
    audit_df, duplicates_df, summary = auditor.run_audit(
        output_dir=output_dir,
        hamming_threshold=threshold,
        create_sheet=not args.no_contact_sheet,
    )

    # Also mirror audit CSV and duplicates CSV to data/manifests
    manifest_dir = PROJECT_ROOT / "data" / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    audit_df.to_csv(manifest_dir / "dataset_audit.csv", index=False)
    duplicates_df.to_csv(manifest_dir / "duplicate_candidates.csv", index=False)
    logger.info(f"Manifests successfully mirrored to: {manifest_dir}")

    print("\n" + "=" * 65)
    print("DATASET AUDIT COMPLETE")
    print("=" * 65)
    print(f"Total images audited       : {summary['total_images']}")
    print(f"Original folder breakdown  : {summary['folder_counts']}")
    print(f"Corrupted images           : {summary['corrupted_images_count']}")
    print(f"Exact duplicate pairs      : {summary['exact_duplicate_pairs_count']}")
    print(f"Near duplicate pairs       : {summary['near_duplicate_pairs_count']}")
    print(f"Unique duplicate groups    : {summary['unique_duplicate_groups']}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
