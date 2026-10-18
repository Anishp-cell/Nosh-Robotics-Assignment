# egg-segmentation-tracking

> **Nosh Robotics Computer Vision Intern Assignment**  
> Complete, reproducible, leakage-aware computer vision pipeline for egg segmentation and video tracking.

---

## 1. Project Overview & Assignment Objectives

This repository contains the software and experimental infrastructure for the four tasks of the Nosh Robotics Computer Vision Intern assignment:

1. **Task 1 — Pretrained Segmentation (Meta SAM)**: Use Meta's Segment Anything Model (SAM ViT-B) to generate reference binary masks, instance segmentations, and overlays across the dataset.
2. **Task 2 — Classical Image Processing**: Develop a modular classical computer vision egg segmentation algorithm using OpenCV and NumPy, tuned strictly on the development split.
3. **Task 3 — Comparative Evaluation & Analysis**: Quantitatively evaluate classical segmentation against SAM reference masks using IoU, Dice, Precision, and Recall; generate failure case galleries and metric distributions.
4. **Task 4 — Video Tracking**: Apply the segmentation and tracking pipeline to the assigned video, maintaining persistent egg IDs across video frames.

---

## 2. Dataset V1 Provenance & Structure

- **Dataset**: Roboflow Universe Egg Dataset V1 (`egg.v1i.folder.zip`).
- **Source URL**: [https://universe.roboflow.com/fire-bf7qe/egg-vzhtr/dataset/1](https://universe.roboflow.com/fire-bf7qe/egg-vzhtr/dataset/1)
- **License**: Creative Commons Attribution 4.0 International ([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/))
- **Images**: 110 JPG images at $640 \times 640$ resolution.
- **Original Folder Breakdown**:
  - `train/egg`: 75 images
  - `train/Unlabeled`: 2 images (`not_damaged_135...jpg`, `not_damaged_145...jpg`)
  - `valid/egg`: 22 images
  - `test/egg`: 11 images
  - `README.dataset.txt`, `README.roboflow.txt`

> [!IMPORTANT]
> The original download is a **classification-folder export**, not an annotated segmentation dataset. Segmentation masks do not exist in the raw dataset. Ground truth reference masks are generated using Meta SAM in Task 1.

### Data Extraction Instructions
The project supports reading directly from `egg.v1i.folder.zip` without extracting, or from an extracted directory.
To extract locally into `data/raw/`:
```bash
python -c "import zipfile; zipfile.ZipFile('../egg.v1i.folder.zip').extractall('data/raw/egg-v1')"
```

---

## 3. Environment Setup & Hardware Configuration

### Verified Target Hardware
- **Operating System**: Windows 11 / Windows 10 (WSL2 Ubuntu supported)
- **GPU**: NVIDIA GeForce RTX 5050 Laptop GPU (8 GB VRAM)
- **RAM**: 16 GB System RAM
- **Python**: Python 3.10.11 (64-bit)
- **PyTorch**: PyTorch 2.12 with CUDA 12.8 acceleration

### Virtual Environment Creation
```bash
# Create dedicated virtual environment
python -m venv .venv

# Activate on Windows PowerShell
.venv\Scripts\Activate.ps1

# Activate on Linux / macOS / WSL
source .venv/bin/activate
```

### Dependency Installation
```bash
# 1. Install core dependencies
pip install -r requirements.txt

# 2. Install editable local package
pip install -e .

# 3. Install official Meta Segment Anything package
pip install git+https://github.com/facebookresearch/segment-anything.git
```

---

## 4. Meta SAM Checkpoint Setup (Task 1)

Download the official Meta SAM ViT-B checkpoint (`sam_vit_b_01ec64.pth`, ~375 MB) into `checkpoints/`:
```bash
python scripts/generate_sam_masks.py --download-checkpoint
```
*Or download directly using curl/wget:*
```bash
curl -o checkpoints/sam_vit_b_01ec64.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

---

## 5. Execution Workflow & Commands

### Step A: Dataset Audit & Duplicate Detection
Audits the raw dataset, checks for corruption, detects exact (SHA-256) and near (dHash) duplicates, and generates a visual contact sheet:
```bash
python scripts/audit_dataset.py
```
*Outputs: `outputs/dataset_audit/dataset_contact_sheet.jpg`, `data/manifests/dataset_audit.csv`, `data/manifests/duplicate_candidates.csv`.*

### Step B: Zero-Leakage Split Creation
Partitions the dataset into an 80:20 development/evaluation split while keeping all duplicate groups atomic:
```bash
python scripts/create_split.py --seed 42
```
*Outputs: `data/manifests/split.csv`, `data/manifests/split_summary.json`.*

### Task 1: Generate SAM Reference Masks
Runs Meta SAM to generate reference masks:
```bash
# Smoke test on 3 images first
python scripts/generate_sam_masks.py --max-images 3

# Full dataset reference mask generation
python scripts/generate_sam_masks.py
```
*Outputs: `outputs/sam/binary_masks/`, `outputs/sam/overlays/`, `outputs/sam/sam_manifest.csv`.*

### Task 2: Run Classical Segmentation Experiment
Runs the classical CV pipeline on the **development split** only:
```bash
python scripts/run_segmentation_experiment.py --split development
```
*Outputs: `outputs/classical/masks/`, `outputs/classical/overlays/`.*

### Task 3: Evaluate Segmentation Performance
Computes IoU, Dice, Precision, and Recall against SAM reference masks and generates error analysis:
```bash
# Evaluate on development split (for tuning)
python scripts/evaluate_segmentation.py --split development

# Benchmark on evaluation split (final validation ONLY)
python scripts/evaluate_segmentation.py --split evaluation
```
*Outputs: `outputs/evaluation/metrics_development.csv`, `outputs/evaluation/failure_cases/`, `outputs/figures/metric_distributions_development.png`.*

### Task 4: Video Tracking Pipeline
Runs segmentation and multi-object tracking on video:
```bash
# Smoke test on first 60 frames
python scripts/track_video.py --video-path data/raw/video/egg_video.mp4 --max-frames 60

# Full video tracking
python scripts/track_video.py --video-path data/raw/video/egg_video.mp4
```
*Outputs: `outputs/tracking/annotated_egg_video.mp4`, `outputs/tracking/tracking_trajectories.csv`.*

---

## 6. Directory Structure Overview

```
egg-segmentation-tracking/
├── configs/                   # Central YAML configurations (dataset, sam, seg, eval, track)
├── data/
│   ├── raw/                   # Raw immutable dataset
│   ├── manifests/             # Audit, duplicates, and split CSVs
│   ├── development/           # Development split references
│   └── evaluation/            # Evaluation split references
├── checkpoints/               # Pretrained weights (sam_vit_b_01ec64.pth)
├── src/eggcv/                 # Core modular Python package
│   ├── common/                # Safe I/O, overlays, error maps, logging
│   ├── dataset/               # Audit, duplicate detection (dHash/pHash/DSU), split
│   ├── segmentation/          # SAM segmenter, classical segmenter, morphology
│   ├── evaluation/            # IoU, Dice, precision, recall, failure galleries
│   └── tracking/              # Centroid distance, Hungarian association, tracker
├── scripts/                   # Runnable entry points with --help
├── experiments/               # Structured experiment log (experiment_log.csv)
├── tests/                     # Comprehensive pytest test suite
├── outputs/                   # Generated masks, overlays, evaluation, videos
├── docs/                      # Architecture, methodology, experiment protocol
├── report/                    # Final report markdown and figures
└── pyproject.toml             # Modern package definition
```

---

## 7. Testing & Verification

Run the full pytest suite:
```bash
pytest tests/ -v
```
Tests cover:
- Corrupted image handling and audit manifest consistency.
- Reproducibility and duplicate-group leakage prevention in splits.
- Analytical correctness of IoU, Dice, Precision, and Recall on toy masks.
- Morphological cleaning, hole filling, component priors, and watershed separation.
- Persistent multi-object tracking and track recovery on synthetic detections.

---

## 8. Known Limitations
- Classical color thresholding is sensitive to heavy cast shadows on conveyor belts.
- Touching/overlapping eggs require distance-transform watershed separation to prevent component merging.
- SAM inference on CPU is computationally intensive; CUDA acceleration on NVIDIA RTX GPU is strongly recommended.
- Roboflow Dataset V1 contains 2 unlabeled images (`train/Unlabeled/`), which have been audited and incorporated into the manifest rather than silently discarded.

---

## 9. AI Coding Tool Disclosure
This codebase was developed with the assistance of an AI coding agent (Google DeepMind Antigravity) under engineering supervision. All architectural decisions, mathematical metric formulations, leakage-prevention constraints, and test suites have been verified for correctness and maintainability.

---

## 10. Deliverable Links & Placeholders
- **Assigned Video URL**: [https://www.youtube.com/watch?v=Y4JyV94y2_A](https://www.youtube.com/watch?v=Y4JyV94y2_A)
- **Final Annotated Video (Google Drive Public Link)**: `[PLACEHOLDER: Insert public link after video upload]`
- **Technical Report**: [report/README.md](report/README.md)
