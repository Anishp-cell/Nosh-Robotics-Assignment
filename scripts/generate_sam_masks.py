#!/usr/bin/env python3
"""
CLI script: Generate reference egg segmentation masks using Meta's Segment Anything Model (Task 1).
"""

import argparse
import sys
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from eggcv.common.io import load_image, load_yaml, save_image, save_mask
from eggcv.common.logging_utils import get_logger
from eggcv.common.visualization import create_contact_sheet, overlay_mask
from eggcv.dataset.inspect_dataset import DatasetAuditor
from eggcv.segmentation.sam_segmenter import SAMSegmenter

logger = get_logger("generate_sam_masks")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate reference egg segmentation masks using Meta's SAM (Task 1).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "sam.yaml"),
        help="Path to SAM configuration YAML",
    )
    parser.add_argument(
        "--dataset-config",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "dataset.yaml"),
        help="Path to dataset configuration YAML",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to raw dataset ZIP or extracted directory (overrides dataset config)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Directory to save masks and overlays (overrides config output dirs)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu"],
        help="Computation device ('cuda' or 'cpu')",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Limit number of images processed (useful for quick smoke tests)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        choices=["development", "evaluation", "all"],
        help="Filter to process only images from a specific split",
    )
    parser.add_argument(
        "--download-checkpoint",
        action="store_true",
        help="Download SAM ViT-B checkpoint weight file automatically",
    )
    return parser.parse_args()


def download_sam_checkpoint(url: str, target_path: Path):
    target_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading SAM checkpoint weights from:\n  {url}\nto:\n  {target_path} ...")
    
    with tqdm(unit="B", unit_scale=True, unit_divisor=1024, miniters=1, desc="SAM ViT-B Weights") as pbar:
        def update_progress(blocks, block_size, total_size):
            if total_size > 0:
                pbar.total = total_size
            pbar.update(blocks * block_size - pbar.n)

        urllib.request.urlretrieve(url, str(target_path), reporthook=update_progress)
    logger.info("Download completed successfully!")


def main():
    args = parse_args()
    sam_cfg = load_yaml(args.config) if Path(args.config).is_file() else {}
    ds_cfg = load_yaml(args.dataset_config) if Path(args.dataset_config).is_file() else {}

    ckpt_path = PROJECT_ROOT / sam_cfg.get("model", {}).get("checkpoint_path", "checkpoints/sam_vit_b_01ec64.pth")
    ckpt_url = sam_cfg.get("model", {}).get(
        "checkpoint_url",
        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
    )

    if args.download_checkpoint:
        download_sam_checkpoint(ckpt_url, ckpt_path)
        return

    # Check device override
    if args.device:
        sam_cfg.setdefault("model", {})["device"] = args.device

    segmenter = SAMSegmenter(sam_cfg)
    is_ready, msg = segmenter.is_available()

    if not is_ready:
        print("\n" + "!" * 65)
        print("SAM SETUP REQUIRED")
        print("!" * 65)
        print(msg)
        print("!" * 65 + "\n")
        sys.exit(1)

    # Initialize SAM
    segmenter.load_model()

    # Determine input images
    input_path = args.input or PROJECT_ROOT / ds_cfg.get("dataset", {}).get("archive_path", "../egg.v1i.folder.zip")
    auditor = DatasetAuditor(source_path=input_path)

    # Output directories
    # Output directories
    out_base = Path(args.output) if args.output else PROJECT_ROOT / "outputs" / "sam"
    masks_dir = out_base / "binary_masks"
    instance_masks_dir = out_base / "instance_masks"
    overlays_dir = out_base / "overlays"
    qc_dir = out_base / "qc_gallery"
    for d in (masks_dir, instance_masks_dir, overlays_dir, qc_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Read image list from audit or ZIP
    split_csv = PROJECT_ROOT / ds_cfg.get("manifests", {}).get("split_file", "data/manifests/split.csv")
    if split_csv.is_file() and args.split and args.split != "all":
        df_split = pd.read_csv(split_csv)
        image_ids = df_split[df_split["split"] == args.split]["image_id"].tolist()
    else:
        audit_csv = PROJECT_ROOT / ds_cfg.get("manifests", {}).get("audit_file", "data/manifests/dataset_audit.csv")
        if audit_csv.is_file():
            image_ids = pd.read_csv(audit_csv)["image_id"].tolist()
        else:
            logger.error("Neither split.csv nor dataset_audit.csv found. Run audit first.")
            sys.exit(1)

    if args.max_images:
        image_ids = image_ids[: args.max_images]

    logger.info(f"Running SAM inference on {len(image_ids)} images...")
    manifest_records = []
    qc_overlays = []
    qc_titles = []

    for img_id in tqdm(image_ids, desc="Generating SAM Masks"):
        img = auditor.get_image(img_id)
        if img is None:
            logger.warning(f"Could not load image: {img_id}")
            continue

        binary_mask, meta = segmenter.segment(img)
        stem = Path(img_id).stem

        # Save composite binary mask
        mask_path = masks_dir / f"{stem}.png"
        save_mask(binary_mask, mask_path)

        # Save individual instance masks where eggs exist
        instance_paths = []
        for inst_idx, inst in enumerate(meta.get("egg_instances", []), start=1):
            inst_path = instance_masks_dir / f"{stem}_inst_{inst_idx:02d}.png"
            if "mask" in inst:
                save_mask(inst["mask"], inst_path)
                instance_paths.append(str(inst_path))

        # Save overlay
        overlay = overlay_mask(img, binary_mask, color=(0, 255, 0), alpha=0.45)
        overlay_path = overlays_dir / f"{stem}_overlay.jpg"
        save_image(overlay, overlay_path)

        # Add to QC collection
        qc_overlays.append(overlay)
        qc_titles.append(f"{stem[:14]} (n={meta['num_egg_instances']})")

        manifest_records.append({
            "image_id": img_id,
            "mask_path": str(mask_path),
            "num_proposals": meta["num_proposals"],
            "num_egg_instances": meta["num_egg_instances"],
            "foreground_pixels": int(np.count_nonzero(binary_mask)),
            "instance_mask_paths": ";".join(instance_paths),
        })

    # Save manifest
    manifest_df = pd.DataFrame(manifest_records)
    manifest_df.to_csv(out_base / "sam_manifest.csv", index=False)
    logger.info(f"SAM mask generation complete. Manifest: {out_base / 'sam_manifest.csv'}")

    # Generate visual QC contact sheet
    if qc_overlays:
        logger.info("Generating visual Quality Control (QC) contact sheet...")
        qc_sheet = create_contact_sheet(
            qc_overlays,
            titles=qc_titles,
            cols=10,
            tile_size=(128, 128),
        )
        qc_sheet_path = qc_dir / "sam_qc_contact_sheet.jpg"
        save_image(qc_sheet, qc_sheet_path)
        logger.info(f"QC contact sheet saved to: {qc_sheet_path}")

    # Print summary statistics
    total_imgs = len(manifest_df)
    total_eggs = manifest_df["num_egg_instances"].sum()
    zero_eggs = (manifest_df["num_egg_instances"] == 0).sum()
    one_egg = (manifest_df["num_egg_instances"] == 1).sum()
    multi_eggs = (manifest_df["num_egg_instances"] > 1).sum()

    print("\n" + "=" * 65)
    print("TASK 1: SAM MASK GENERATION COMPLETE")
    print("=" * 65)
    print(f"Total images processed     : {total_imgs}")
    print(f"Total egg instances found  : {total_eggs}")
    print(f"Images with 1 egg          : {one_egg}")
    print(f"Images with >1 eggs        : {multi_eggs}")
    print(f"Images with 0 eggs         : {zero_eggs}")
    print(f"Binary masks directory     : {masks_dir}")
    print(f"Instance masks directory   : {instance_masks_dir}")
    print(f"Overlays directory         : {overlays_dir}")
    print(f"QC contact sheet           : {qc_dir / 'sam_qc_contact_sheet.jpg'}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
