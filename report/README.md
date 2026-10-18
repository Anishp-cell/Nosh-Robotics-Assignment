# Technical Report & Deliverables

## Nosh Robotics Computer Vision Internship Assignment
**Candidate Project**: `egg-segmentation-tracking`  
**License**: MIT (Software) / CC BY 4.0 (Dataset)  

---

## 1. Executive Summary & Assigned Objectives
This project develops and evaluates an end-to-end egg segmentation and video tracking pipeline across four assigned tasks:
1. **Task 1: Pretrained Segmentation (Meta SAM)** — Reference mask generation and quality assurance.
2. **Task 2: Classical Image Processing** — Leakage-aware classical egg segmentation using OpenCV/NumPy.
3. **Task 3: Comparative Evaluation & Analysis** — Benchmark classical vs. SAM masks across development and evaluation splits.
4. **Task 4: Video Tracking** — Persistent ID multi-object tracking on the assigned video.

---

## 2. Dataset V1 Audit & Leakage-Aware Splitting
- **Dataset Origin**: Roboflow Universe (`fire-bf7qe/egg-vzhtr/dataset/1`).
- **Archive File**: `egg.v1i.folder.zip` (110 images, $640 \times 640$, JPG).
- **Split Distribution**:
  - Development Split: Target 80% (~88 images).
  - Evaluation Split: Target 20% (~22 images).
- **Leakage Prevention**: Perceptual difference hashing (dHash) and exact SHA-256 duplicate detection were performed prior to splitting. Images within duplicate groups are kept strictly atomic to avoid data contamination.

---

## 3. Experimental Benchmarks (Placeholders to be Populated Post-Inference)

| Method / Experiment ID | Split | Mean IoU | Median IoU | Mean Dice | Precision | Recall | Runtime (ms/img) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| *EXP-001 (Baseline Otsu)* | Dev | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* |
| *EXP-002 (Lab L\* + CLAHE)* | Dev | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* |
| *EXP-003 (Morph + Priors)* | Dev | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* |
| **Final Selected Model** | **Eval** | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* |

---

## 4. Video Tracking Deliverable (Task 4)
- **Assigned YouTube Source**: [https://www.youtube.com/watch?v=Y4JyV94y2_A](https://www.youtube.com/watch?v=Y4JyV94y2_A)
- **Annotated Video Deliverable (Google Drive Public URL)**:
  `[PLACEHOLDER: Insert public Google Drive video link here]`
- **Local Annotated Video Artifact**: `outputs/tracking/annotated_egg_video.mp4`
- **Trajectory CSV Log**: `outputs/tracking/tracking_trajectories.csv`

---

## 5. Dataset Attribution
The dataset utilized in this project was created and published on Roboflow Universe:
- **Title**: Egg Dataset V1 (Exported May 23, 2025)
- **URL**: [https://universe.roboflow.com/fire-bf7qe/egg-vzhtr/dataset/1](https://universe.roboflow.com/fire-bf7qe/egg-vzhtr/dataset/1)
- **License**: Creative Commons Attribution 4.0 International (CC BY 4.0)
- The original license terms permit sharing and adapting with proper attribution.
