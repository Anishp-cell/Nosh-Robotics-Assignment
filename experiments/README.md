# Experiment Tracking Protocol

## Purpose
This directory tracks every classical segmentation and tracking experiment.
To maintain research integrity and prevent data snooping:
1. **Zero Evaluation Set Tuning**: All algorithmic variations and hyperparameter tuning must be evaluated strictly on the **development split**.
2. **Evaluation Split Benchmark**: The evaluation split is reserved exclusively for the final selected model to assess true generalization.
3. **Record All Experiments**: Unsuccessful attempts, edge case failures, and negative findings must be documented alongside successes.

## Experiment Log Columns (`experiment_log.csv`)
- `experiment_id`: Unique identifier (e.g. `EXP-001-baseline-otsu`)
- `timestamp`: Execution timestamp (ISO 8601)
- `hypothesis`: Engineering hypothesis motivating the experiment (e.g. "Otsu thresholding on Lab L* channel separates egg from dark conveyor background")
- `method`: Algorithm components used (e.g. `Gaussian Blur + Lab L* + Otsu + Morph Close`)
- `parameters`: Key hyperparameters (e.g. `blur_ksize=(5,5), close_ksize=(7,7), min_area=2000`)
- `development_split`: Name of development split used (e.g. `development-80`)
- `mean_iou`: Measured mean Intersection over Union on development set
- `mean_dice`: Measured mean Dice coefficient (F1 score)
- `precision`: Measured pixel precision
- `recall`: Measured pixel recall
- `runtime_seconds`: Execution time across development split
- `decision`: Engineering decision (`ADOPT`, `REJECT`, or `ITERATE`)
- `observations`: Key qualitative observations (e.g. "Struggles with shadow boundary on right side")
- `failure_cases`: Representative failure image IDs for error analysis
