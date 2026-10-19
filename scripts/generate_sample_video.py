#!/usr/bin/env python3
"""
Utility script: Generate a realistic synthetic conveyor egg video from dataset images
to validate and demonstrate Task 4 video tracking pipeline.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import cv2
import numpy as np
from eggcv.common.io import load_mask
from eggcv.dataset.inspect_dataset import DatasetAuditor

auditor = DatasetAuditor(source_path=PROJECT_ROOT / ".." / "egg.v1i.folder.zip")

output_dir = PROJECT_ROOT / "data" / "raw" / "video"
output_dir.mkdir(parents=True, exist_ok=True)
video_path = output_dir / "egg_conveyor_demo.mp4"

# Load a clean sample image with eggs
img_id = "not_damaged_101_jpg.rf.479d12a2b41a5ec6d718dcdabd14b0c9.jpg"
base_img = auditor.get_image(img_id)
ref_mask = load_mask(PROJECT_ROOT / "outputs" / "sam" / "binary_masks" / "not_damaged_101_jpg.rf.479d12a2b41a5ec6d718dcdabd14b0c9.png")

# Extract individual egg patches using connected components
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(ref_mask)

egg_patches = []
for label in range(1, num_labels):
    area = stats[label, cv2.CC_STAT_AREA]
    if area < 5000:
        continue
    x = stats[label, cv2.CC_STAT_LEFT]
    y = stats[label, cv2.CC_STAT_TOP]
    w = stats[label, cv2.CC_STAT_WIDTH]
    h = stats[label, cv2.CC_STAT_HEIGHT]
    
    egg_crop = base_img[y:y+h, x:x+w].copy()
    mask_crop = (labels[y:y+h, x:x+w] == label).astype(np.uint8) * 255
    egg_patches.append((egg_crop, mask_crop))

print(f"Extracted {len(egg_patches)} egg patches.")

# Create synthetic conveyor video (640x640, 120 frames at 30 fps)
fps = 30
width, height = 640, 640
total_frames = 120
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))

# Simulate 3 eggs moving from left to right at constant conveyor speed
conveyor_speed = 4.0  # pixels per frame

eggs = [
    {"patch_idx": 0 % len(egg_patches), "start_x": -100, "y": 180, "speed": conveyor_speed},
    {"patch_idx": 1 % len(egg_patches), "start_x": -300, "y": 360, "speed": conveyor_speed},
    {"patch_idx": 0 % len(egg_patches), "start_x": -520, "y": 250, "speed": conveyor_speed},
]

# Generate synthetic conveyor background
bg = np.full((height, width, 3), (120, 125, 128), dtype=np.uint8)
# Add conveyor belt horizontal texture lines
for y_line in range(0, height, 40):
    cv2.line(bg, (0, y_line), (width, y_line), (100, 105, 108), 2)

for f in range(total_frames):
    frame = bg.copy()
    # Scroll conveyor texture slightly
    shift = int((f * conveyor_speed) % 40)
    for y_line in range(shift, height, 40):
        cv2.line(frame, (0, y_line), (width, y_line), (95, 100, 103), 1)

    for egg in eggs:
        curr_x = int(egg["start_x"] + f * egg["speed"])
        curr_y = egg["y"]
        patch, mask = egg_patches[egg["patch_idx"]]
        ph, pw = patch.shape[:2]

        # Calculate overlap with frame canvas
        x1_dst = max(0, curr_x)
        y1_dst = max(0, curr_y)
        x2_dst = min(width, curr_x + pw)
        y2_dst = min(height, curr_y + ph)

        if x1_dst >= x2_dst or y1_dst >= y2_dst:
            continue

        x1_src = x1_dst - curr_x
        y1_src = y1_dst - curr_y
        x2_src = x1_src + (x2_dst - x1_dst)
        y2_src = y1_src + (y2_dst - y1_dst)

        sub_patch = patch[y1_src:y2_src, x1_src:x2_src]
        sub_mask = mask[y1_src:y2_src, x1_src:x2_src, None] / 255.0

        target_roi = frame[y1_dst:y2_dst, x1_dst:x2_dst]
        blended = (sub_patch * sub_mask + target_roi * (1.0 - sub_mask)).astype(np.uint8)
        frame[y1_dst:y2_dst, x1_dst:x2_dst] = blended

    writer.write(frame)

writer.release()
print(f"Generated synthetic conveyor video: {video_path} ({total_frames} frames, {width}x{height} @ {fps} fps)")
