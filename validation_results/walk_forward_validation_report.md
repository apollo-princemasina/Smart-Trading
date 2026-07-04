# Walk-Forward Validation Report

## Executive Summary

- Task type:          classification
- Windows evaluated:  6
- Models evaluated:   4
- Total time:         55.6s
- Best model:         **xgboost**

## Model Rankings

| Rank | Model | Status | Ranking Score | Stability | Robustness | F1 Mean |
|---|---|---|---|---|---|---|
| 1 | xgboost | ⚠ Improve | 0.7802 | 0.9369 | 0.6022 | 0.5055 |
| 2 | lightgbm | ⚠ Improve | 0.7687 | 0.9360 | 0.6005 | 0.4897 |
| 3 | random_forest | ⚠ Improve | 0.7366 | 0.9144 | 0.5604 | 0.3978 |
| 4 | extra_trees | ⚠ Improve | 0.7330 | 0.9309 | 0.5464 | 0.3559 |

## Acceptance Summary

### ⚠ xgboost — Needs Improvement
  - Overfitting detected (train-test gap=0.2136)

### ⚠ lightgbm — Needs Improvement
  - Overfitting detected (train-test gap=0.3383)

### ⚠ random_forest — Needs Improvement
  - Overfitting detected (train-test gap=0.3982)

### ⚠ extra_trees — Needs Improvement
  - Overfitting detected (train-test gap=0.3565)
