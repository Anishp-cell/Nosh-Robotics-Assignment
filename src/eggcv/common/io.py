"""
I/O helpers for safe reading and writing of images, masks, and configurations.
"""

import hashlib
from pathlib import Path
from typing import Any, Dict, Union

import cv2
import numpy as np
import yaml


def load_image(source: Union[str, Path, bytes], bgr: bool = True) -> np.ndarray:
    """
    Safely load an image from a file path or raw bytes.

    Args:
        source: File path (str or Path) or raw image bytes
        bgr: If True, returns BGR (OpenCV format); if False, returns RGB

    Returns:
        np.ndarray: Image array of shape (H, W, 3) and dtype uint8

    Raises:
        FileNotFoundError: If file path does not exist
        ValueError: If file is corrupted or cannot be decoded
    """
    if isinstance(source, bytes):
        arr = np.frombuffer(source, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode image from raw bytes (corrupted or unsupported format).")
    else:
        path = Path(source)
        if not path.is_file():
            raise FileNotFoundError(f"Image file not found: {path.resolve()}")
        # cv2.imread cannot handle non-ASCII paths well on Windows, cv2.imdecode handles all paths safely:
        with open(path, "rb") as f:
            data = f.read()
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Failed to decode image: {path.resolve()} (corrupted or unsupported format).")

    if not bgr:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


def save_image(image: np.ndarray, path: Union[str, Path]) -> Path:
    """
    Save an image safely to disk, creating parent directories if needed.

    Args:
        image: Image array
        path: Target file path

    Returns:
        Path: Path to saved file
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    # Use cv2.imencode + open to avoid Windows Unicode path issues
    ext = target.suffix or ".jpg"
    success, encoded = cv2.imencode(ext, image)
    if not success:
        raise IOError(f"Failed to encode image to format {ext}")
    with open(target, "wb") as f:
        f.write(encoded.tobytes())
    return target


def save_mask(mask: np.ndarray, path: Union[str, Path]) -> Path:
    """
    Save a binary mask as a single-channel 8-bit PNG file (values 0 or 255).

    Args:
        mask: 2D array (bool, float, or uint8) where positive values represent the foreground
        path: Output file path (must end in .png)

    Returns:
        Path: Path to saved mask file
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    if mask.ndim != 2:
        raise ValueError(f"Mask must be 2-dimensional (H, W), but got shape {mask.shape}")

    # Standardize to 0 and 255 uint8
    if mask.dtype == bool:
        binary_mask = (mask.astype(np.uint8)) * 255
    else:
        binary_mask = np.where(mask > 0, 255, 0).astype(np.uint8)

    success, encoded = cv2.imencode(".png", binary_mask)
    if not success:
        raise IOError(f"Failed to encode mask to PNG for path {target}")
    with open(target, "wb") as f:
        f.write(encoded.tobytes())
    return target


def load_mask(path: Union[str, Path]) -> np.ndarray:
    """
    Load a single-channel binary mask from disk.

    Args:
        path: Path to PNG mask file

    Returns:
        np.ndarray: 2D uint8 array with values 0 or 255
    """
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"Mask file not found: {target.resolve()}")

    with open(target, "wb" if False else "rb") as f:
        data = f.read()
    arr = np.frombuffer(data, dtype=np.uint8)
    mask = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Failed to decode mask file: {target.resolve()}")

    # Threshold strictly to binary 0 or 255
    binary = np.where(mask > 127, 255, 0).astype(np.uint8)
    return binary


def compute_sha256(data_or_path: Union[str, Path, bytes]) -> str:
    """
    Compute SHA-256 hash from file path or bytes.
    """
    sha = hashlib.sha256()
    if isinstance(data_or_path, bytes):
        sha.update(data_or_path)
    else:
        path = Path(data_or_path)
        with open(path, "rb") as f:
            while chunk := f.read(65536):
                sha.update(chunk)
    return sha.hexdigest()


def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load a YAML configuration file safely.
    """
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"Configuration file not found: {target.resolve()}")
    with open(target, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def save_yaml(data: Dict[str, Any], path: Union[str, Path]) -> Path:
    """
    Save a dictionary as a YAML file safely.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)
    return target
