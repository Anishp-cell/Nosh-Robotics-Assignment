"""
Duplicate and near-duplicate image detection using exact hashes (SHA-256)
and perceptual hashes (dHash, aHash, pHash) with Union-Find grouping.
"""

from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import cv2
import numpy as np
import pandas as pd


def compute_dhash(image: np.ndarray, hash_size: int = 8) -> str:
    """
    Compute Difference Hash (dHash) for an image using OpenCV and NumPy.
    Resizes image to (hash_size + 1, hash_size) and computes horizontal gradient.

    Args:
        image: BGR or grayscale image array
        hash_size: Grid size (default 8 produces 8x8 = 64-bit hash)

    Returns:
        str: 16-character hexadecimal hash string
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Resize to (width=hash_size + 1, height=hash_size)
    resized = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    # Compare adjacent horizontal pixels
    diff = resized[:, 1:] > resized[:, :-1]
    # Pack bits into integer
    bit_string = "".join("1" if b else "0" for b in diff.flatten())
    int_val = int(bit_string, 2)
    hex_len = (hash_size * hash_size) // 4
    return f"{int_val:0{hex_len}x}"


def compute_ahash(image: np.ndarray, hash_size: int = 8) -> str:
    """
    Compute Average Hash (aHash) for an image.
    Resizes image to (hash_size, hash_size) and compares pixels against the mean.

    Args:
        image: BGR or grayscale image array
        hash_size: Grid size (default 8 produces 64-bit hash)

    Returns:
        str: 16-character hexadecimal hash string
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    resized = cv2.resize(gray, (hash_size, hash_size), interpolation=cv2.INTER_AREA)
    avg = np.mean(resized)
    diff = resized > avg
    bit_string = "".join("1" if b else "0" for b in diff.flatten())
    int_val = int(bit_string, 2)
    hex_len = (hash_size * hash_size) // 4
    return f"{int_val:0{hex_len}x}"


def compute_phash(image: np.ndarray, hash_size: int = 8, highfreq_factor: int = 4) -> str:
    """
    Compute Perceptual Hash (pHash) using Discrete Cosine Transform (DCT).

    Args:
        image: BGR or grayscale image array
        hash_size: Grid size (default 8 produces 64-bit hash)
        highfreq_factor: Scaling factor for initial resize (default 4 -> 32x32)

    Returns:
        str: 16-character hexadecimal hash string
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    img_size = hash_size * highfreq_factor
    resized = cv2.resize(gray, (img_size, img_size), interpolation=cv2.INTER_AREA).astype(np.float32)
    # 2D DCT using OpenCV
    dct = cv2.dct(resized)
    # Extract low frequency 8x8 block (excluding DC coefficient at [0, 0])
    dct_low = dct[:hash_size, :hash_size]
    med = np.median(dct_low)
    diff = dct_low > med
    bit_string = "".join("1" if b else "0" for b in diff.flatten())
    int_val = int(bit_string, 2)
    hex_len = (hash_size * hash_size) // 4
    return f"{int_val:0{hex_len}x}"


def hamming_distance(hex_hash1: str, hex_hash2: str) -> int:
    """
    Compute Hamming distance (number of differing bits) between two hex hashes.
    """
    if len(hex_hash1) != len(hex_hash2):
        raise ValueError(f"Hash lengths do not match: {len(hex_hash1)} vs {len(hex_hash2)}")
    val1 = int(hex_hash1, 16)
    val2 = int(hex_hash2, 16)
    xor_val = val1 ^ val2
    return bin(xor_val).count("1")


class DisjointSetUnion:
    """Union-Find / Disjoint Set Union data structure for grouping connected images."""

    def __init__(self, elements: List[str]):
        self.parent = {elem: elem for elem in elements}
        self.rank = {elem: 0 for elem in elements}

    def find(self, item: str) -> str:
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, item1: str, item2: str):
        root1 = self.find(item1)
        root2 = self.find(item2)
        if root1 != root2:
            if self.rank[root1] < self.rank[root2]:
                self.parent[root1] = root2
            elif self.rank[root1] > self.rank[root2]:
                self.parent[root2] = root1
            else:
                self.parent[root2] = root1
                self.rank[root1] += 1


def find_duplicate_candidates(
    audit_df: pd.DataFrame,
    hamming_threshold: int = 6,
    hash_column: str = "dhash",
) -> pd.DataFrame:
    """
    Pairwise search for exact and near-duplicate candidates.

    Args:
        audit_df: DataFrame containing at least 'image_id', 'sha256', and the hash_column
        hamming_threshold: Maximum Hamming distance to consider as a near-duplicate candidate
        hash_column: Which perceptual hash column to compare (e.g. 'dhash')

    Returns:
        pd.DataFrame: Pairwise candidates with match classification
    """
    records = []
    n = len(audit_df)

    rows = audit_df.to_dict("records")
    for i in range(n):
        for j in range(i + 1, n):
            row_a = rows[i]
            row_b = rows[j]

            id_a = row_a["image_id"]
            id_b = row_b["image_id"]

            sha_match = row_a.get("sha256") == row_b.get("sha256")

            h_dist = -1
            hash_a = str(row_a.get(hash_column, ""))
            hash_b = str(row_b.get(hash_column, ""))
            if hash_a and hash_b and len(hash_a) == len(hash_b):
                h_dist = hamming_distance(hash_a, hash_b)

            # Exact duplicate or within perceptual threshold
            if sha_match or (0 <= h_dist <= hamming_threshold):
                records.append({
                    "image_a": id_a,
                    "image_b": id_b,
                    "folder_a": row_a.get("original_folder", ""),
                    "folder_b": row_b.get("original_folder", ""),
                    "sha256_match": sha_match,
                    "hamming_distance": h_dist,
                    "is_exact_duplicate": sha_match,
                    "is_near_duplicate": (not sha_match) and (h_dist <= hamming_threshold),
                })

    df = pd.DataFrame(records)
    if df.empty:
        df = pd.DataFrame(columns=[
            "image_a", "image_b", "folder_a", "folder_b",
            "sha256_match", "hamming_distance", "is_exact_duplicate", "is_near_duplicate"
        ])
    return df


def build_duplicate_groups(
    image_ids: List[str],
    candidate_pairs_df: pd.DataFrame,
) -> Dict[str, int]:
    """
    Group images into connected components where duplicates and near-duplicates
    share the same integer group ID. Singletons receive their own unique group IDs.

    Args:
        image_ids: Complete list of all image IDs in the dataset
        candidate_pairs_df: Pairwise candidate DataFrame

    Returns:
        Dict[str, int]: Mapping from image_id to integer group_id (1-indexed)
    """
    dsu = DisjointSetUnion(image_ids)

    # Union all pairs flagged as exact or near duplicates
    if not candidate_pairs_df.empty:
        for _, row in candidate_pairs_df.iterrows():
            dsu.union(str(row["image_a"]), str(row["image_b"]))

    # Map root IDs to compact consecutive integers 1, 2, ...
    root_to_group: Dict[str, int] = {}
    image_to_group: Dict[str, int] = {}

    current_group_id = 1
    for img_id in image_ids:
        root = dsu.find(img_id)
        if root not in root_to_group:
            root_to_group[root] = current_group_id
            current_group_id += 1
        image_to_group[img_id] = root_to_group[root]

    return image_to_group


def generate_pair_comparison_image(
    img_a: np.ndarray,
    img_b: np.ndarray,
    label_a: str,
    label_b: str,
    distance_info: str,
) -> np.ndarray:
    """
    Create a side-by-side visual comparison banner for two candidate duplicate images.
    """
    h_a, w_a = img_a.shape[:2]
    h_b, w_b = img_b.shape[:2]

    target_h = 320
    target_w = int(w_a * (target_h / h_a))
    res_a = cv2.resize(img_a, (target_w, target_h))
    res_b = cv2.resize(img_b, (target_w, target_h))

    banner_h = 40
    total_w = target_w * 2 + 10
    total_h = target_h + banner_h

    canvas = np.full((total_h, total_w, 3), (35, 35, 35), dtype=np.uint8)
    canvas[banner_h:, :target_w] = res_a
    canvas[banner_h:, target_w + 10 :] = res_b

    # Add header text
    cv2.putText(canvas, distance_info, (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
    # Add image labels
    cv2.putText(canvas, label_a[:35], (10, banner_h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(canvas, label_b[:35], (target_w + 15, banner_h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    return canvas
