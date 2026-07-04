# Robustness Report

Aggregate statistics for each model across all walk-forward windows.

## xgboost

- Robustness score:   0.6022
- Best window:        4  (score=0.5463)
- Worst window:       1 (score=0.4574)

### F1 Statistics

| Stat   | Value  |
|--------|--------|
| Mean   | 0.5055 |
| Median | 0.5025 |
| Min    | 0.4574 |
| Max    | 0.5463 |
| Std    | 0.0274 |
| CV     | 0.0542 |

### Additional Metrics (Mean ± Std)

- accuracy: 0.6590 ± 0.0685
- roc_auc: 0.7575 ± 0.0230
- directional_accuracy: 0.6590 ± 0.0685

## lightgbm

- Robustness score:   0.6005
- Best window:        4  (score=0.5284)
- Worst window:       1 (score=0.4289)

### F1 Statistics

| Stat   | Value  |
|--------|--------|
| Mean   | 0.4897 |
| Median | 0.4891 |
| Min    | 0.4289 |
| Max    | 0.5284 |
| Std    | 0.0329 |
| CV     | 0.0673 |

### Additional Metrics (Mean ± Std)

- accuracy: 0.6629 ± 0.0664
- roc_auc: 0.7572 ± 0.0195
- directional_accuracy: 0.6629 ± 0.0664

## random_forest

- Robustness score:   0.5604
- Best window:        4  (score=0.4406)
- Worst window:       0 (score=0.2990)

### F1 Statistics

| Stat   | Value  |
|--------|--------|
| Mean   | 0.3978 |
| Median | 0.4111 |
| Min    | 0.2990 |
| Max    | 0.4406 |
| Std    | 0.0472 |
| CV     | 0.1186 |

### Additional Metrics (Mean ± Std)

- accuracy: 0.6572 ± 0.0753
- roc_auc: 0.7314 ± 0.0254
- directional_accuracy: 0.6572 ± 0.0753

## extra_trees

- Robustness score:   0.5464
- Best window:        0  (score=0.3832)
- Worst window:       3 (score=0.3116)

### F1 Statistics

| Stat   | Value  |
|--------|--------|
| Mean   | 0.3559 |
| Median | 0.3635 |
| Min    | 0.3116 |
| Max    | 0.3832 |
| Std    | 0.0255 |
| CV     | 0.0716 |

### Additional Metrics (Mean ± Std)

- accuracy: 0.6509 ± 0.0839
- roc_auc: 0.7274 ± 0.0281
- directional_accuracy: 0.6509 ± 0.0839
