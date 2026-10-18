# Technical Methodology & Computer Vision Formulation

## 1. Dataset Provenance & Nature
- **Source**: Roboflow Universe (`fire-bf7qe/egg-vzhtr/dataset/1`).
- **Export Type**: Classification-folder export (`train/egg`, `train/Unlabeled`, `valid/egg`, `test/egg`).
- **Original Content**: 110 JPG images at $640 \times 640$ pixels.
- **License**: Creative Commons Attribution 4.0 International (CC BY 4.0).
- **Key Realization**: This is **not** an annotated segmentation dataset. Segmentation masks do not exist in the raw export. Ground truth reference masks for evaluation must be generated using Meta's Segment Anything Model (SAM) as specified in Task 1.
- **Unlabeled Images**: The dataset contains 2 images in `train/Unlabeled/` (`not_damaged_135` and `not_damaged_145`). Rather than discarding them blindly, they are audited and incorporated into the split manifest.

---

## 2. Classical Segmentation Algorithm Formulation (Task 2)

Egg segmentation presents distinctive computer vision challenges:
- Specular reflection (glare on smooth curved shell surfaces creating internal voids).
- Shadows cast by eggs onto conveyor belts or trays.
- Natural texture variations (brown eggs, white eggs, speckled shells).
- Low contrast between egg boundaries and light-colored backgrounds.

### Candidate Technical Pipeline
1. **Pre-filtering**: Gaussian smoothing ($5 \times 5, \sigma=1.0$) suppresses high-frequency camera sensor noise.
2. **Color Space Transformation**:
   - RGB is sensitive to illumination variations.
   - **CIE $L^*a^*b^*$**: Channel $L^*$ represents perceptual lightness. $b^*$ separates warm egg tones from cool industrial backgrounds.
   - **HSV**: Value ($V$) and Saturation ($S$) isolate bright egg surfaces.
3. **Contrast Equalization**: Optional CLAHE (Contrast Limited Adaptive Histogram Equalization) balances local illumination across images.
4. **Thresholding**:
   - **Otsu's Global Thresholding**: Minimizes intra-class variance of foreground and background pixel intensities.
   - **Adaptive Gaussian Thresholding**: Computes localized thresholds for non-uniform lighting conditions.
5. **Morphological Post-processing**:
   - **Morphological Opening**: Eliminates isolated speckle noise.
   - **Morphological Closing**: Bridges gaps in outer egg contours.
   - **Boundary Flood-Fill Hole Filling**: Reclaims interior pixels lost due to bright specular reflection.
6. **Geometric Shape Prior Filtering**:
   - Eggs are smooth, convex, roughly elliptical bodies.
   - Filter criteria: Minimum area, maximum area, circularity ($\frac{4\pi \cdot \text{Area}}{\text{Perimeter}^2} \ge 0.25$), and aspect ratio ($0.35 \le \frac{w}{h} \le 2.80$).
7. **Watershed Separation**: Optional Euclidean Distance Transform + Watershed marker segmentation to resolve touching egg instances.

---

## 3. Evaluation Metric Definitions (Task 3)

Let $P$ denote the binary prediction mask and $R$ denote the binary reference mask.
$TP = |P \cap R|$, $FP = |P \cap \bar{R}|$, $FN = |\bar{P} \cap R|$, $TN = |\bar{P} \cap \bar{R}|$.

### Metric Equations
- **Intersection over Union (IoU / Jaccard Index)**:
  $$\text{IoU} = \frac{TP}{TP + FP + FN}$$
- **Dice Coefficient (F1-Score)**:
  $$\text{Dice} = \frac{2 \cdot TP}{2 \cdot TP + FP + FN}$$
- **Pixel Precision**:
  $$\text{Precision} = \frac{TP}{TP + FP}$$
- **Pixel Recall**:
  $$\text{Recall} = \frac{TP}{TP + FN}$$

### Empty Mask Conventions
- When both $P$ and $R$ are empty ($TP=FP=FN=0$): $\text{IoU}=1.0$, $\text{Dice}=1.0$, $\text{Precision}=1.0$, $\text{Recall}=1.0$.
- When $R$ is empty and $P$ has foreground: $\text{IoU}=0.0$, $\text{Dice}=0.0$, $\text{Precision}=0.0$, $\text{Recall}=0.0$.
- When $R$ has foreground and $P$ is empty: $\text{IoU}=0.0$, $\text{Dice}=0.0$, $\text{Precision}=0.0$, $\text{Recall}=0.0$.

---

## 4. Multi-Object Video Tracking (Task 4)
- **Centroid State**: Each active egg track maintains an $(x, y)$ centroid trajectory history.
- **Cost Formulation**: Pairwise Euclidean distance matrix $C_{ij} = \|c_i^{\text{track}} - c_j^{\text{detection}}\|_2$.
- **Assignment**: Hungarian algorithm (`scipy.optimize.linear_sum_assignment`) finds optimal global bipartite matching subject to maximum distance threshold $d_{\text{max}} = 75$ px.
- **Track Lifecycle**:
  - Unmatched detections initialize new candidate tracks.
  - Candidate tracks require $\ge 2$ consecutive confirmations to receive a persistent track ID.
  - Missing tracks coast for up to 20 frames before termination.
