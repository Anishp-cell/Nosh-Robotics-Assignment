"""
Dataset inspection and audit utility for Roboflow Egg Dataset V1.
Supports direct auditing of the ZIP archive or an extracted folder.
"""

import json
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd

from eggcv.common.io import compute_sha256, load_image, save_image
from eggcv.common.logging_utils import get_logger
from eggcv.common.visualization import create_contact_sheet
from eggcv.dataset.duplicate_detection import (
    build_duplicate_groups,
    compute_ahash,
    compute_dhash,
    compute_phash,
    find_duplicate_candidates,
    generate_pair_comparison_image,
)

logger = get_logger("eggcv.dataset.inspect")


class DatasetAuditor:
    """Audits the egg dataset directly from ZIP archive or extracted directory."""

    def __init__(self, source_path: Union[str, Path]):
        self.source_path = Path(source_path)
        if not self.source_path.exists():
            raise FileNotFoundError(f"Source dataset path does not exist: {self.source_path.resolve()}")
        self.is_zip = self.source_path.is_file() and self.source_path.suffix.lower() == ".zip"

    def run_audit(
        self,
        output_dir: Union[str, Path],
        hamming_threshold: int = 6,
        create_sheet: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        """
        Execute comprehensive dataset audit.

        Args:
            output_dir: Directory where audit CSVs, summary JSON, and contact sheets are written
            hamming_threshold: Hamming distance threshold for flagging perceptual duplicates
            create_sheet: Whether to generate the visual contact sheet

        Returns:
            Tuple of (audit_df, duplicate_candidates_df, summary_dict)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Starting dataset audit on source: {self.source_path.resolve()} (ZIP mode: {self.is_zip})")

        if self.is_zip:
            records, images_for_sheet = self._audit_from_zip()
        else:
            records, images_for_sheet = self._audit_from_dir()

        audit_df = pd.DataFrame(records)

        # Pairwise duplicate search
        logger.info("Computing exact and perceptual duplicate candidates...")
        duplicate_candidates_df = find_duplicate_candidates(
            audit_df=audit_df,
            hamming_threshold=hamming_threshold,
            hash_column="dhash",
        )

        # Build connected component group IDs to prevent leakage
        logger.info("Assigning zero-leakage duplicate group IDs...")
        all_ids = audit_df["image_id"].tolist()
        group_mapping = build_duplicate_groups(all_ids, duplicate_candidates_df)
        audit_df["duplicate_group_id"] = audit_df["image_id"].map(group_mapping)

        # Save duplicate visual comparisons if any candidates found
        if not duplicate_candidates_df.empty:
            pairs_dir = output_dir / "duplicate_pairs"
            pairs_dir.mkdir(parents=True, exist_ok=True)
            self._save_candidate_visuals(duplicate_candidates_df, pairs_dir)

        # Generate contact sheet
        contact_sheet_path = None
        if create_sheet and images_for_sheet:
            logger.info("Generating dataset contact sheet...")
            sheet_imgs = [item[1] for item in images_for_sheet]
            sheet_titles = [f"{item[0][:15]} (G{group_mapping.get(item[0], 0)})" for item in images_for_sheet]
            sheet = create_contact_sheet(sheet_imgs, titles=sheet_titles, cols=10, tile_size=(128, 128))
            contact_sheet_path = output_dir / "dataset_contact_sheet.jpg"
            save_image(sheet, contact_sheet_path)
            logger.info(f"Contact sheet saved to: {contact_sheet_path}")

        # Summary statistics
        total_images = len(audit_df)
        corrupted_count = int(audit_df["is_corrupted"].sum())
        exact_duplicate_pairs = int(duplicate_candidates_df["is_exact_duplicate"].sum()) if not duplicate_candidates_df.empty else 0
        near_duplicate_pairs = int(duplicate_candidates_df["is_near_duplicate"].sum()) if not duplicate_candidates_df.empty else 0
        total_groups = audit_df["duplicate_group_id"].nunique()

        folder_breakdown = audit_df.groupby("original_folder").size().to_dict()

        summary: Dict[str, Any] = {
            "source_path": str(self.source_path),
            "is_zip": self.is_zip,
            "total_images": total_images,
            "folder_counts": folder_breakdown,
            "corrupted_images_count": corrupted_count,
            "exact_duplicate_pairs_count": exact_duplicate_pairs,
            "near_duplicate_pairs_count": near_duplicate_pairs,
            "unique_duplicate_groups": total_groups,
            "hamming_distance_threshold": hamming_threshold,
            "contact_sheet_path": str(contact_sheet_path) if contact_sheet_path else None,
        }

        # Save files
        audit_csv_path = output_dir / "dataset_audit.csv"
        candidates_csv_path = output_dir / "duplicate_candidates.csv"
        summary_json_path = output_dir / "audit_summary.json"

        audit_df.to_csv(audit_csv_path, index=False)
        duplicate_candidates_df.to_csv(candidates_csv_path, index=False)
        with open(summary_json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Audit completed: {total_images} images audited. Breakdown: {folder_breakdown}")
        logger.info(f"Saved audit manifest: {audit_csv_path}")
        logger.info(f"Saved duplicate candidates: {candidates_csv_path}")

        return audit_df, duplicate_candidates_df, summary

    def _audit_from_zip(self) -> Tuple[List[Dict[str, Any]], List[Tuple[str, np.ndarray]]]:
        records: List[Dict[str, Any]] = []
        images_for_sheet: List[Tuple[str, np.ndarray]] = []

        with zipfile.ZipFile(self.source_path, "r") as z:
            namelist = sorted([n for n in z.namelist() if not n.endswith("/")])
            for name in namelist:
                ext = Path(name).suffix.lower()
                if ext not in [".jpg", ".jpeg", ".png"]:
                    # Non-image files (e.g. README.dataset.txt)
                    logger.debug(f"Skipping non-image file: {name}")
                    continue

                raw_bytes = z.read(name)
                record = self._process_image_bytes(name, raw_bytes)
                records.append(record)

                if not record["is_corrupted"] and record["_decoded_img"] is not None:
                    images_for_sheet.append((record["image_id"], record["_decoded_img"]))

        # Remove internal decoded image references before returning records
        for r in records:
            r.pop("_decoded_img", None)

        return records, images_for_sheet

    def _audit_from_dir(self) -> Tuple[List[Dict[str, Any]], List[Tuple[str, np.ndarray]]]:
        records: List[Dict[str, Any]] = []
        images_for_sheet: List[Tuple[str, np.ndarray]] = []

        image_files = sorted([
            p for p in self.source_path.rglob("*")
            if p.is_file() and p.suffix.lower() in [".jpg", ".jpeg", ".png"]
        ])

        for path in image_files:
            rel_name = path.relative_to(self.source_path).as_posix()
            with open(path, "rb") as f:
                raw_bytes = f.read()

            record = self._process_image_bytes(rel_name, raw_bytes, absolute_path=path)
            records.append(record)

            if not record["is_corrupted"] and record["_decoded_img"] is not None:
                images_for_sheet.append((record["image_id"], record["_decoded_img"]))

        for r in records:
            r.pop("_decoded_img", None)

        return records, images_for_sheet

    def _process_image_bytes(
        self,
        rel_path: str,
        raw_bytes: bytes,
        absolute_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Decode image bytes and extract checksums, hashes, and geometric attributes."""
        filename = Path(rel_path).name
        folder = Path(rel_path).parent.as_posix() or "root"
        file_size = len(raw_bytes)
        sha256 = compute_sha256(raw_bytes)

        is_corrupted = False
        width, height, channels = 0, 0, 0
        dhash_str, ahash_str, phash_str = "", "", ""
        mean_intensity, std_intensity = 0.0, 0.0
        decoded_img = None

        try:
            arr = np.frombuffer(raw_bytes, dtype=np.uint8)
            decoded_img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if decoded_img is None:
                is_corrupted = True
            else:
                height, width, channels = decoded_img.shape
                dhash_str = compute_dhash(decoded_img)
                ahash_str = compute_ahash(decoded_img)
                phash_str = compute_phash(decoded_img)
                mean_intensity = float(np.mean(decoded_img))
                std_intensity = float(np.std(decoded_img))
        except Exception as e:
            logger.warning(f"Error decoding image {rel_path}: {e}")
            is_corrupted = True

        return {
            "image_id": filename,
            "relative_path": rel_path,
            "original_folder": folder,
            "file_size_bytes": file_size,
            "sha256": sha256,
            "width": width,
            "height": height,
            "channels": channels,
            "is_corrupted": is_corrupted,
            "dhash": dhash_str,
            "ahash": ahash_str,
            "phash": phash_str,
            "mean_intensity": round(mean_intensity, 2),
            "std_intensity": round(std_intensity, 2),
            "absolute_path": str(absolute_path) if absolute_path else "",
            "_decoded_img": decoded_img,
        }

    def _save_candidate_visuals(self, candidates_df: pd.DataFrame, output_dir: Path):
        """Save visual side-by-side comparison images for all candidate pairs."""
        if candidates_df.empty:
            return

        # Cache decoded images
        cache: Dict[str, np.ndarray] = {}

        for idx, row in candidates_df.iterrows():
            id_a = str(row["image_a"])
            id_b = str(row["image_b"])

            for img_id in (id_a, id_b):
                if img_id not in cache:
                    img = self.get_image(img_id)
                    if img is not None:
                        cache[img_id] = img

            if id_a in cache and id_b in cache:
                kind = "EXACT_SHA" if row["is_exact_duplicate"] else f"NEAR_dHash_dist_{row['hamming_distance']}"
                comp = generate_pair_comparison_image(
                    cache[id_a],
                    cache[id_b],
                    f"{id_a} ({row['folder_a']})",
                    f"{id_b} ({row['folder_b']})",
                    f"Match: {kind}",
                )
                safe_name = f"pair_{idx:03d}_{id_a[:10]}_vs_{id_b[:10]}.jpg"
                save_image(comp, output_dir / safe_name)

    def get_image(self, image_id: str) -> Optional[np.ndarray]:
        """Fetch image array by image_id (filename)."""
        if self.is_zip:
            with zipfile.ZipFile(self.source_path, "r") as z:
                for name in z.namelist():
                    if Path(name).name == image_id:
                        raw = z.read(name)
                        arr = np.frombuffer(raw, dtype=np.uint8)
                        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
        else:
            for p in self.source_path.rglob(image_id):
                if p.is_file():
                    return load_image(p)
        return None
