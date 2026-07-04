# Generalization Report

Evaluates overfitting, underfitting, performance degradation, and regime sensitivity.

## xgboost

| Dimension               | Value |
|-------------------------|-------|
| Generalization score    | 0.9220 |
| Overfitting detected    | Yes ⚠ |
| Underfitting detected   | No |
| Performance degradation | No |
| Regime sensitivity      | 0.0204 |
| Train-test gap          | 0.2136 |
| Degradation slope       | 0.0043 |

**Market Regime Performance**

| Regime   | Mean Score |
|----------|------------|
| High Vol | 0.5157 |
| Low Vol  | 0.4953 |
| Trending | N/A |
| Ranging  | 0.5055 |

## lightgbm

| Dimension               | Value |
|-------------------------|-------|
| Generalization score    | 0.8828 |
| Overfitting detected    | Yes ⚠ |
| Underfitting detected   | No |
| Performance degradation | No |
| Regime sensitivity      | 0.0133 |
| Train-test gap          | 0.3383 |
| Degradation slope       | 0.0050 |

**Market Regime Performance**

| Regime   | Mean Score |
|----------|------------|
| High Vol | 0.4964 |
| Low Vol  | 0.4830 |
| Trending | N/A |
| Ranging  | 0.4897 |

## random_forest

| Dimension               | Value |
|-------------------------|-------|
| Generalization score    | 0.8518 |
| Overfitting detected    | Yes ⚠ |
| Underfitting detected   | No |
| Performance degradation | No |
| Regime sensitivity      | 0.0463 |
| Train-test gap          | 0.3982 |
| Degradation slope       | 0.0233 |

**Market Regime Performance**

| Regime   | Mean Score |
|----------|------------|
| High Vol | 0.4210 |
| Low Vol  | 0.3746 |
| Trending | N/A |
| Ranging  | 0.3978 |

## extra_trees

| Dimension               | Value |
|-------------------------|-------|
| Generalization score    | 0.8677 |
| Overfitting detected    | Yes ⚠ |
| Underfitting detected   | No |
| Performance degradation | No |
| Regime sensitivity      | 0.0404 |
| Train-test gap          | 0.3565 |
| Degradation slope       | -0.0014 |

**Market Regime Performance**

| Regime   | Mean Score |
|----------|------------|
| High Vol | 0.3357 |
| Low Vol  | 0.3761 |
| Trending | N/A |
| Ranging  | 0.3559 |
