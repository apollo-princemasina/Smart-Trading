# Label Generation Report

**Symbol**: EURUSD  
**Timeframe**: M15  
**Version**: v1  
**Generated**: 2026-07-01T20:51:46.849781+00:00  
**Generator**: LabelPipeline v1  

---

## Summary

| Metric | Value |
|---|---|
| Total rows | 87,503 |
| Valid rows | 87,502 (100.0%) |
| Label columns | 53 |
| Validation | ✓ PASSED |

---

## Label Columns

| Column | Dtype | NaN Rate | Valid Rows | Range / Classes |
|---|---|---|---|---|
| fwd_return_1b | float64 | 0.0% | 87,502 | [-0.01147, 0.01602] |
| direction_1b | float64 | 0.0% | 87,502 | 0.0, 1.0, 2.0 |
| bias_1b | float64 | 0.0% | 87,502 | 0.0, 1.0 |
| confidence_1b | float64 | 0.0% | 87,483 | [0, 1] |
| probability_1b | float64 | 0.0% | 87,483 | [4.54e-05, 1] |
| fwd_return_3b | float64 | 0.0% | 87,500 | [-0.0155, 0.01771] |
| direction_3b | float64 | 0.0% | 87,500 | 0.0, 1.0, 2.0 |
| bias_3b | float64 | 0.0% | 87,500 | 0.0, 1.0 |
| confidence_3b | float64 | 0.0% | 87,481 | [0, 1] |
| probability_3b | float64 | 0.0% | 87,481 | [4.54e-05, 1] |
| fwd_return_5b | float64 | 0.0% | 87,498 | [-0.01653, 0.0189] |
| direction_5b | float64 | 0.0% | 87,498 | 0.0, 1.0, 2.0 |
| bias_5b | float64 | 0.0% | 87,498 | 0.0, 1.0 |
| confidence_5b | float64 | 0.0% | 87,479 | [0, 1] |
| probability_5b | float64 | 0.0% | 87,479 | [4.54e-05, 0.9999] |
| fwd_return_10b | float64 | 0.0% | 87,493 | [-0.01608, 0.0223] |
| direction_10b | float64 | 0.0% | 87,493 | 0.0, 1.0, 2.0 |
| bias_10b | float64 | 0.0% | 87,493 | 0.0, 1.0 |
| confidence_10b | float64 | 0.0% | 87,474 | [0, 1] |
| probability_10b | float64 | 0.0% | 87,474 | [0.0002302, 0.9997] |
| long_outcome | float64 | 0.1% | 87,447 | 0.0, 1.0, 2.0 |
| long_outcome_bars | float64 | 0.1% | 87,447 | [1, 50] |
| long_mfe_pct | float64 | 0.1% | 87,447 | [0, 0.01795] |
| long_mae_pct | float64 | 0.1% | 87,447 | [0, 0.0127] |
| long_rr | float64 | 0.1% | 87,447 | [2, 2] |
| short_outcome | float64 | 0.1% | 87,447 | 0.0, 1.0, 2.0 |
| short_outcome_bars | float64 | 0.1% | 87,447 | [1, 50] |
| short_mfe_pct | float64 | 0.1% | 87,447 | [0, 0.01394] |
| short_mae_pct | float64 | 0.1% | 87,447 | [0, 0.01684] |
| short_rr | float64 | 0.1% | 87,447 | [2, 2] |
| outcome | float64 | 0.1% | 87,447 | 0.0, 1.0, 2.0 |
| outcome_bars | float64 | 0.1% | 87,447 | [1, 50] |
| mfe_pct | float64 | 0.1% | 87,447 | [0, 0.01795] |
| mae_pct | float64 | 0.1% | 87,447 | [0, 0.01684] |
| realized_rr | float64 | 0.1% | 87,447 | [2, 2] |
| expected_reward_pct | float64 | 0.1% | 87,447 | [0.0002326, 0.007444] |
| expected_risk_pct | float64 | 0.1% | 87,447 | [0.0001163, 0.003722] |
| trade_duration_bars | float64 | 0.1% | 87,447 | [1, 50] |
| setup_quality | float64 | 0.1% | 87,447 | 0.0, 1.0, 2.0, 3.0 |
| setup_score | float64 | 0.1% | 87,447 | [0, 100] |
| setup_mfe_mae_ratio | float64 | 0.0% | 87,496 | [0, 1.218e+06] |
| setup_achievable_rr | float64 | 0.0% | 87,496 | [0, 9.617] |
| entry_signal | float64 | 0.1% | 87,447 | 0.0, 1.0, 2.0 |
| optimal_entry_offset | float64 | 0.1% | 87,447 | [-1, 9] |
| time_to_entry | float64 | 1.7% | 86,015 | 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0 |
| is_optimal_entry | float64 | 0.1% | 87,453 | 0.0, 1.0 |
| mgmt_strategy | float64 | 0.1% | 87,447 | 0.0, 1.0, 3.0 |
| mgmt_optimal_exit_bar | float64 | 0.1% | 87,447 | [0, 49] |
| mgmt_max_r_multiple | float64 | 0.1% | 87,447 | [-3.79, 12.99] |
| mgmt_breakeven_bar | float64 | 0.1% | 87,447 | [-1, 30] |
| mgmt_trail_bar | float64 | 0.1% | 87,447 | [-1, 48] |
| mgmt_partial_exit_bar | float64 | 0.1% | 87,447 | [-1, 44] |
| mgmt_exit_type | float64 | 0.1% | 87,447 | 0.0, 1.0, 2.0, 3.0 |

---

## Class Distributions

### direction_1b

| Class | Count | Proportion |
|---|---|---|
| 0 | 16,317 | 18.65% |
| 1 | 54,777 | 62.60% |
| 2 | 16,408 | 18.75% |

### bias_1b

| Class | Count | Proportion |
|---|---|---|
| 0 | 43,968 | 50.25% |
| 1 | 43,534 | 49.75% |

### direction_3b

| Class | Count | Proportion |
|---|---|---|
| 0 | 24,672 | 28.20% |
| 1 | 37,721 | 43.11% |
| 2 | 25,107 | 28.69% |

### bias_3b

| Class | Count | Proportion |
|---|---|---|
| 0 | 43,499 | 49.71% |
| 1 | 44,001 | 50.29% |

### direction_5b

| Class | Count | Proportion |
|---|---|---|
| 0 | 28,164 | 32.19% |
| 1 | 30,458 | 34.81% |
| 2 | 28,876 | 33.00% |

### bias_5b

| Class | Count | Proportion |
|---|---|---|
| 0 | 43,337 | 49.53% |
| 1 | 44,161 | 50.47% |

### direction_10b

| Class | Count | Proportion |
|---|---|---|
| 0 | 32,035 | 36.61% |
| 1 | 22,015 | 25.16% |
| 2 | 33,443 | 38.22% |

### bias_10b

| Class | Count | Proportion |
|---|---|---|
| 0 | 43,165 | 49.34% |
| 1 | 44,328 | 50.66% |

### long_outcome

| Class | Count | Proportion |
|---|---|---|
| 0 | 1,488 | 1.70% |
| 1 | 27,708 | 31.69% |
| 2 | 58,251 | 66.61% |

### short_outcome

| Class | Count | Proportion |
|---|---|---|
| 0 | 1,478 | 1.69% |
| 1 | 27,881 | 31.88% |
| 2 | 58,088 | 66.43% |

### outcome

| Class | Count | Proportion |
|---|---|---|
| 0 | 1,829 | 2.09% |
| 1 | 41,335 | 47.27% |
| 2 | 44,283 | 50.64% |

### setup_quality

| Class | Count | Proportion |
|---|---|---|
| 0 | 34,814 | 39.81% |
| 1 | 11,249 | 12.86% |
| 2 | 49 | 0.06% |
| 3 | 41,335 | 47.27% |

### entry_signal

| Class | Count | Proportion |
|---|---|---|
| 0 | 1,432 | 1.64% |
| 1 | 49,875 | 57.03% |
| 2 | 36,140 | 41.33% |

### time_to_entry

| Class | Count | Proportion |
|---|---|---|
| 0 | 36,140 | 42.02% |
| 1 | 4,370 | 5.08% |
| 2 | 4,095 | 4.76% |
| 3 | 4,190 | 4.87% |
| 4 | 4,376 | 5.09% |
| 5 | 4,567 | 5.31% |
| 6 | 4,989 | 5.80% |
| 7 | 5,838 | 6.79% |
| 8 | 7,367 | 8.56% |
| 9 | 10,083 | 11.72% |

### is_optimal_entry

| Class | Count | Proportion |
|---|---|---|
| 0 | 49,935 | 57.10% |
| 1 | 37,518 | 42.90% |

### mgmt_strategy

| Class | Count | Proportion |
|---|---|---|
| 0 | 51 | 0.06% |
| 1 | 43,113 | 49.30% |
| 3 | 44,283 | 50.64% |

### mgmt_exit_type

| Class | Count | Proportion |
|---|---|---|
| 0 | 41,335 | 47.27% |
| 1 | 44,177 | 50.52% |
| 2 | 106 | 0.12% |
| 3 | 1,829 | 2.09% |

---

## Model Targets

| Model | Recommended Target Columns |
|---|---|
| Market Bias Classifier | direction_1b, direction_3b, direction_5b, direction_10b |
| Market Bias Regressor | fwd_return_1b, fwd_return_3b, fwd_return_5b |
| Setup Quality | setup_quality, setup_score |
| Entry Timing | entry_signal, is_optimal_entry |
| Trade Outcome Classifier | long_outcome, short_outcome, outcome |
| Trade Outcome Regressor | long_mfe_pct, long_mae_pct, realized_rr |
| Trade Management | mgmt_strategy, mgmt_optimal_exit_bar |

---

## Validation

```
LabelValidation: PASSED
  [PASS] empty: 87503 rows present.
  [PASS] nan_rate: 'fwd_return_1b': 0.0% NaN — OK.
  [PASS] nan_rate: 'direction_1b': 0.0% NaN — OK.
  [PASS] nan_rate: 'bias_1b': 0.0% NaN — OK.
  [PASS] nan_rate: 'confidence_1b': 0.0% NaN — OK.
  [PASS] nan_rate: 'probability_1b': 0.0% NaN — OK.
  [PASS] nan_rate: 'fwd_return_3b': 0.0% NaN — OK.
  [PASS] nan_rate: 'direction_3b': 0.0% NaN — OK.
  [PASS] nan_rate: 'bias_3b': 0.0% NaN — OK.
  [PASS] nan_rate: 'confidence_3b': 0.0% NaN — OK.
  [PASS] nan_rate: 'probability_3b': 0.0% NaN — OK.
  [PASS] nan_rate: 'fwd_return_5b': 0.0% NaN — OK.
  [PASS] nan_rate: 'direction_5b': 0.0% NaN — OK.
  [PASS] nan_rate: 'bias_5b': 0.0% NaN — OK.
  [PASS] nan_rate: 'confidence_5b': 0.0% NaN — OK.
  [PASS] nan_rate: 'probability_5b': 0.0% NaN — OK.
  [PASS] nan_rate: 'fwd_return_10b': 0.0% NaN — OK.
  [PASS] nan_rate: 'direction_10b': 0.0% NaN — OK.
  [PASS] nan_rate: 'bias_10b': 0.0% NaN — OK.
  [PASS] nan_rate: 'confidence_10b': 0.0% NaN — OK.
  [PASS] nan_rate: 'probability_10b': 0.0% NaN — OK.
  [PASS] nan_rate: 'long_outcome': 0.1% NaN — OK.
  [PASS] nan_rate: 'long_outcome_bars': 0.1% NaN — OK.
  [PASS] nan_rate: 'long_mfe_pct': 0.1% NaN — OK.
  [PASS] nan_rate: 'long_mae_pct': 0.1% NaN — OK.
  [PASS] nan_rate: 'long_rr': 0.1% NaN — OK.
  [PASS] nan_rate: 'short_outcome': 0.1% NaN — OK.
  [PASS] nan_rate: 'short_outcome_bars': 0.1% NaN — OK.
  [PASS] nan_rate: 'short_mfe_pct': 0.1% NaN — OK.
  [PASS] nan_rate: 'short_mae_pct': 0.1% NaN — OK.
  [PASS] nan_rate: 'short_rr': 0.1% NaN — OK.
  [PASS] nan_rate: 'outcome': 0.1% NaN — OK.
  [PASS] nan_rate: 'outcome_bars': 0.1% NaN — OK.
  [PASS] nan_rate: 'mfe_pct': 0.1% NaN — OK.
  [PASS] nan_rate: 'mae_pct': 0.1% NaN — OK.
  [PASS] nan_rate: 'realized_rr': 0.1% NaN — OK.
  [PASS] nan_rate: 'expected_reward_pct': 0.1% NaN — OK.
  [PASS] nan_rate: 'expected_risk_pct': 0.1% NaN — OK.
  [PASS] nan_rate: 'trade_duration_bars': 0.1% NaN — OK.
  [PASS] nan_rate: 'setup_quality': 0.1% NaN — OK.
  [PASS] nan_rate: 'setup_score': 0.1% NaN — OK.
  [PASS] nan_rate: 'setup_mfe_mae_ratio': 0.0% NaN — OK.
  [PASS] nan_rate: 'setup_achievable_rr': 0.0% NaN — OK.
  [PASS] nan_rate: 'entry_signal': 0.1% NaN — OK.
  [PASS] nan_rate: 'optimal_entry_offset': 0.1% NaN — OK.
  [PASS] nan_rate: 'time_to_entry': 1.7% NaN — OK.
  [PASS] nan_rate: 'is_optimal_entry': 0.1% NaN — OK.
  [PASS] nan_rate: 'mgmt_strategy': 0.1% NaN — OK.
  [PASS] nan_rate: 'mgmt_optimal_exit_bar': 0.1% NaN — OK.
  [PASS] nan_rate: 'mgmt_max_r_multiple': 0.1% NaN — OK.
  [PASS] nan_rate: 'mgmt_breakeven_bar': 0.1% NaN — OK.
  [PASS] nan_rate: 'mgmt_trail_bar': 0.1% NaN — OK.
  [PASS] nan_rate: 'mgmt_partial_exit_bar': 0.1% NaN — OK.
  [PASS] nan_rate: 'mgmt_exit_type': 0.1% NaN — OK.
  [PASS] class_balance: 'direction_1b': balanced (min class 18.65%).
  [PASS] class_balance: 'bias_1b': balanced (min class 49.75%).
  [PASS] class_balance: 'direction_3b': balanced (min class 28.20%).
  [PASS] class_balance: 'bias_3b': balanced (min class 49.71%).
  [PASS] class_balance: 'direction_5b': balanced (min class 32.19%).
  [PASS] class_balance: 'bias_5b': balanced (min class 49.53%).
  [PASS] class_balance: 'direction_10b': balanced (min class 25.16%).
  [PASS] class_balance: 'bias_10b': balanced (min class 49.34%).
  [WARNING] class_balance: 'long_outcome': minority class 1.70% < floor 2.00%.
  [WARNING] class_balance: 'short_outcome': minority class 1.69% < floor 2.00%.
  [PASS] class_balance: 'outcome': balanced (min class 2.09%).
  [WARNING] class_balance: 'setup_quality': minority class 0.06% < floor 2.00%.
  [WARNING] class_balance: 'entry_signal': minority class 1.64% < floor 2.00%.
  [PASS] class_balance: 'time_to_entry': balanced (min class 4.76%).
  [PASS] class_balance: 'is_optimal_entry': balanced (min class 42.90%).
  [WARNING] class_balance: 'mgmt_strategy': minority class 0.06% < floor 2.00%.
  [WARNING] class_balance: 'mgmt_exit_type': minority class 0.12% < floor 2.00%.
  [WARNING] value_range: 'setup_score' expected [0,1], range [0.000, 100.000].
```

---

## Notes

_No notes._

> Labels are strictly forward-looking. They must NEVER be used as input features.
> Always drop NaN rows before training. Last N rows are NaN by design.