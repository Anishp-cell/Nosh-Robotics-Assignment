"""
Logging utilities for eggcv.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def get_logger(name: str = "eggcv", log_file: Optional[Path] = None, level: int = logging.INFO) -> logging.Logger:
    """
    Get or create a configured logger with standard formatting.

    Args:
        name: Name of the logger
        log_file: Optional file path to mirror logs to disk
        level: Logging level (default: logging.INFO)

    Returns:
        Configured logging.Logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding duplicate handlers if logger was already created
    if not logger.handlers:
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        if log_file:
            log_file = Path(log_file)
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(str(log_file), encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger
