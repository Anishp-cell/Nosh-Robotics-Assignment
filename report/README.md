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

## 3. Experimental Benchmarks

### Development Split Experiments (n=88 images)

| Experiment ID | Method | Mean IoU | Mean Dice | Precision | Recall | Runtime (s) | Decision |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| EXP-001 | Baseline Grayscale Otsu | 0.3675 | 0.5009 | 0.536 | 0.6234 | 6.1 | ITERATE |
| EXP-002 | Morph Glare Repair (Hole Fill) | 0.3669 | 0.4995 | 0.5342 | 0.6312 | 5.44 | ADOPT |
| EXP-003A | HSV Saturation Channel | 0.1773 | 0.2655 | 0.3113 | 0.3402 | 5.26 | REJECT |
| EXP-003B | Lab L* + CLAHE | 0.3445 | 0.4798 | 0.4323 | 0.7157 | 6.5 | ITERATE |
| EXP-004 | Adaptive Gaussian Threshold | 0.3440 | 0.4748 | 0.3442 | 0.9986 | 8.31 | REJECT |
| EXP-005 | Canny Edge + Closing | 0.1439 | 0.2157 | 0.3835 | 0.2780 | 4.37 | REJECT |
| EXP-006 | Strict Shape Priors | 0.2474 | 0.3279 | 0.4562 | 0.3321 | 3.73 | REJECT |
| EXP-007 | Watershed Instance Separation | 0.3540 | 0.4841 | 0.5475 | 0.5900 | 7.06 | ADOPT |
| **EXP-008** | **Final Optimized Classical** | **0.3718** | **0.5047** | **0.5476** | **0.6230** | **5.41** | **ADOPT** |

### Evaluation Split Benchmark (n=22 held-out images, Task 3)

| Metric | Classical CV (Adopted) | SAM ViT-B (Reference) |
| :--- | :---: | :---: |
| Mean IoU | 0.3897 | 1.000 (reference) |
| Median IoU | 0.2967 | 1.000 (reference) |
| Mean Dice (F1) | 0.5207 | 1.000 (reference) |
| Mean Precision | 0.5396 | 1.000 (reference) |
| Mean Recall | 0.6489 | 1.000 (reference) |
| Inference (ms/img) | **8.1 ms (CPU)** | ~180 ms (CUDA GPU) |
| Throughput | **123.5 FPS** | 5.6 FPS |
| **Speed-up** | **22.2×** | 1.0× |

---

## 4. Video Tracking Deliverable (Task 4)
- **Assigned YouTube Source**: [https://www.youtube.com/watch?v=Y4JyV94y2_A](https://www.youtube.com/watch?v=Y4JyV94y2_A)
- **Annotated Video Deliverable (Google Drive Public URL)**:
  `[PLACEHOLDER: Upload annotated_egg_video.mp4 to Google Drive and insert public link here]`
- **Local Annotated Video Artifact**: `outputs/tracking/annotated_egg_video.mp4`
- **Trajectory CSV Log**: `outputs/tracking/tracking_trajectories.csv`
- **Tracking Metrics Summary**: `outputs/tracking/tracking_metrics.json`

### Task 4 Results

| Metric | Value |
| :--- | :--- |
| Video Duration | 60s (1440 frames @ 24 FPS, 640×360) |
| Unique Confirmed Tracks | 51 |
| Total Trajectory Observations | 5,204 |
| Avg Detections Per Frame | 3.6 (range: 1–9) |
| Mean Track Lifespan | 102 frames (~4.3 sec) |
| Longest Track | 1,438 frames (Track #1, ~59.9 sec) |
| Processing Speed | **276 FPS** on CPU |
| Association Method | Centroid Euclidean Distance + Hungarian Assignment |

---

## 5. Dataset Attribution
The dataset utilized in this project was created and published on Roboflow Universe:
- **Title**: Egg Dataset V1 (Exported May 23, 2025)
- **URL**: [https://universe.roboflow.com/fire-bf7qe/egg-vzhtr/dataset/1](https://universe.roboflow.com/fire-bf7qe/egg-vzhtr/dataset/1)
- **License**: Creative Commons Attribution 4.0 International (CC BY 4.0)
- The original license terms permit sharing and adapting with proper attribution.
