# Experiment Protocol & Leakage Prevention Standard

## 1. Scientific Discipline
In applied computer vision, an algorithm that appears effective due to data leakage or uncontrolled evaluation snooping will fail in real-world deployment. This project adheres strictly to reproducible, leakage-aware experimental methodologies.

---

## 2. Split Protocol: Development vs. Evaluation Sets

The dataset is partitioned into an **80:20 split**:
- **Development Split (~80%, 88 images)**:
  - Used for visual exploratory data analysis (EDA).
  - Used for color-space channel analysis.
  - Used for morphological kernel sizing and threshold parameter tuning.
  - Evaluated repeatedly during iterative engineering.
- **Evaluation Split (~20%, 22 images)**:
  - **RESERVED EXCLUSIVELY FOR FINAL VALIDATION.**
  - **No parameter may be tuned using the evaluation split.**
  - **No algorithm choice may be made based on evaluation-split performance.**
  - Evaluated only once at the conclusion of development to report unbiased generalization performance.

---

## 3. Leakage Prevention via Perceptual Hash Grouping
Standard random splitting is vulnerable to near-duplicate image leakage (e.g. adjacent video frames or repeated shots in Roboflow classification exports). If an image identical or near-identical to an evaluation image is present in the development set, the evaluation score is artificially inflated.

**Our Mitigation**:
1. **SHA-256**: Identifies exact byte-level duplicate pairs.
2. **Difference Hashing (dHash)**: Computes a 64-bit gradient signature invariant to minor illumination shifts.
3. **Disjoint Set Union (DSU)**: Images with Hamming distance $\le 6$ or identical SHA-256 are clustered into atomic groups.
4. **Group Partitioning**: Entire groups are placed on one side of the split boundary. No group is ever split between development and evaluation.

---

## 4. Experiment Logging (`experiments/experiment_log.csv`)
Every algorithmic hypothesis must be recorded before or immediately upon execution:
1. `experiment_id`: Consecutive alphanumeric identifier (`EXP-001-baseline-otsu`, `EXP-002-lab-clahe`, etc.).
2. `hypothesis`: Engineering rationale explaining *why* the modification is expected to improve performance.
3. `parameters`: Exact dictionary of YAML parameters used.
4. `mean_iou`, `mean_dice`, `precision`, `recall`: Quantitatively measured metrics on the development set.
5. `decision`: `ADOPT` (promoted to default), `REJECT` (discarded), or `ITERATE` (leads to follow-up experiment).
6. `failure_cases`: Specific image IDs where the algorithm broke down, enabling targeted error analysis.

Unsuccessful experiments are valuable engineering data and must never be deleted or concealed.
