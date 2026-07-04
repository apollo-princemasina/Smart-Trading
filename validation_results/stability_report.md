# Stability Report

Measures window-to-window consistency of model performance.

## xgboost

- Stability score:   0.9369  (STABLE)
- Most variable:     short_accuracy
- Least variable:    n_classes
- Confidence CV:     0.0630
- Prediction std:    0.0428

### Coefficient of Variation (top variable metrics)

| Metric | CV |
|--------|----|
| short_accuracy | 0.2664 |
| risk_reward_accuracy | 0.1961 |
| log_loss | 0.1451 |
| expected_return | 0.1418 |
| cohen_kappa | 0.1189 |
| tp_prediction_accuracy | 0.1116 |
| sl_prediction_accuracy | 0.1082 |
| directional_accuracy | 0.1039 |

### Per-Window Scores (primary metric)

| Window | Score |
|--------|-------|
| 001 | 0.5266 |
| 002 | 0.4574 |
| 003 | 0.4978 |
| 004 | 0.5030 |
| 005 | 0.5463 |
| 006 | 0.5020 |

## lightgbm

- Stability score:   0.9360  (STABLE)
- Most variable:     log_loss
- Least variable:    n_classes
- Confidence CV:     0.1302
- Prediction std:    0.0953

### Coefficient of Variation (top variable metrics)

| Metric | CV |
|--------|----|
| log_loss | 0.2351 |
| short_accuracy | 0.2154 |
| risk_reward_accuracy | 0.1829 |
| expected_return | 0.1791 |
| cohen_kappa | 0.1563 |
| avg_confidence | 0.1302 |
| tp_prediction_accuracy | 0.1219 |
| directional_accuracy | 0.1001 |

### Per-Window Scores (primary metric)

| Window | Score |
|--------|-------|
| 001 | 0.5225 |
| 002 | 0.4289 |
| 003 | 0.4802 |
| 004 | 0.4805 |
| 005 | 0.5284 |
| 006 | 0.4977 |

## random_forest

- Stability score:   0.9144  (STABLE)
- Most variable:     short_accuracy
- Least variable:    n_classes
- Confidence CV:     0.1462
- Prediction std:    0.0791

### Coefficient of Variation (top variable metrics)

| Metric | CV |
|--------|----|
| short_accuracy | 0.4472 |
| cohen_kappa | 0.4208 |
| mcc | 0.2831 |
| expected_risk | 0.2784 |
| expected_return | 0.2490 |
| precision | 0.1649 |
| avg_confidence | 0.1462 |
| sl_prediction_accuracy | 0.1339 |

### Per-Window Scores (primary metric)

| Window | Score |
|--------|-------|
| 001 | 0.2990 |
| 002 | 0.3956 |
| 003 | 0.3969 |
| 004 | 0.4254 |
| 005 | 0.4406 |
| 006 | 0.4293 |

## extra_trees

- Stability score:   0.9309  (STABLE)
- Most variable:     risk_reward_accuracy
- Least variable:    n_classes
- Confidence CV:     0.0578
- Prediction std:    0.0344

### Coefficient of Variation (top variable metrics)

| Metric | CV |
|--------|----|
| risk_reward_accuracy | 0.2033 |
| short_accuracy | 0.1919 |
| expected_return | 0.1733 |
| tp_prediction_accuracy | 0.1380 |
| cohen_kappa | 0.1372 |
| expected_risk | 0.1352 |
| directional_accuracy | 0.1290 |
| accuracy | 0.1290 |

### Per-Window Scores (primary metric)

| Window | Score |
|--------|-------|
| 001 | 0.3832 |
| 002 | 0.3652 |
| 003 | 0.3336 |
| 004 | 0.3116 |
| 005 | 0.3619 |
| 006 | 0.3798 |
