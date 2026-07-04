# Hyperparameter Optimization Report

## Summary

| Field | Value |
|---|---|
| Best model | `xgboost` |
| Best window | 0 |
| Composite score | 0.6734 |
| Optimization metric | f1 |
| Val score (f1) | 0.5310 |

## Per-Window Results

| Window | Model | Trials | Best Val Score | Opt. Time (s) | Improved? |
|---|---|---|---|---|---|
| 000 | xgboost | 25 | 0.5310 | 41963.8 |  |
| 000 | lightgbm | 25 | 0.5295 | 2270.1 |  |
| 000 | random_forest | 16 | 0.4759 | 2435.7 |  |
| 000 | extra_trees | 25 | 0.4630 | 3007.9 |  |
| 001 | xgboost | 15 | 0.4779 | 6070.9 |  |
| 001 | lightgbm | 22 | 0.4528 | 1016.4 |  |
| 001 | random_forest | 25 | 0.4215 | 11212.2 |  |
| 001 | extra_trees | 25 | 0.4106 | 2156.2 |  |
| 002 | xgboost | 24 | 0.4991 | 1388.4 |  |
| 002 | lightgbm | 17 | 0.4788 | 878.1 |  |
| 002 | random_forest | 25 | 0.4453 | 12255.6 |  |
| 002 | extra_trees | 25 | 0.4179 | 2153.8 |  |
| 003 | xgboost | 23 | 0.5037 | 1173.8 |  |
| 003 | lightgbm | 25 | 0.5009 | 1069.6 |  |
| 003 | random_forest | 25 | 0.4568 | 11057.8 |  |
| 003 | extra_trees | 25 | 0.4337 | 2236.3 |  |
| 004 | xgboost | 25 | 0.5379 | 24200.6 |  |
| 004 | lightgbm | 22 | 0.5374 | 992.9 |  |
| 004 | random_forest | 25 | 0.4938 | 15318.3 |  |
| 004 | extra_trees | 25 | 0.4579 | 3473.6 |  |
| 005 | xgboost | 15 | 0.5306 | 926.2 |  |
| 005 | lightgbm | 22 | 0.5304 | 1010.2 |  |
| 005 | random_forest | 25 | 0.5045 | 11228.8 |  |
| 005 | extra_trees | 25 | 0.4609 | 3278.3 |  |

## Aggregate Statistics

- Windows optimized:   24
- Total trials:        551
- Total opt. time:     162775.3s
- Mean val score:      0.4813
- Min  val score:      0.4106
- Max  val score:      0.5379
