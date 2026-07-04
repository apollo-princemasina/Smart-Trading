# Training Dataset Report

**Dataset**: training_dataset v3  
**Symbol**: EURUSD  
**Timeframe**: M15  
**Generated**: 2026-07-01T20:53:43.813571+00:00  
**Pipeline**: 1.0.0  

---

## Dataset Summary

| Metric | Value |
|---|---|
| Total rows | 87,502 |
| Feature columns | 252 |
| Label columns | 53 |
| Total columns | 305 |
| Start date | 2022-06-21 18:30:00+00:00 |
| End date | 2025-12-30 21:45:00+00:00 |
| Missing values | 1.70% |
| Feature set | all |
| Label version | v1 |
| Schema version | 1.0.0 |
| Validation | ✓ PASSED |

---

## Feature Summary

| Prefix | Feature Count |
|---|---|
| `other_` | 30 |
| `rolling_` | 15 |
| `fvg_` | 10 |
| `swing_` | 10 |
| `price_` | 9 |
| `ob_` | 8 |
| `nearest_` | 7 |
| `w1_` | 6 |
| `d1_` | 6 |
| `h4_` | 6 |
| `h1_` | 6 |
| `lower_` | 6 |
| `trend_` | 5 |
| `distance_` | 5 |
| `liquidity_` | 5 |
| `higher_` | 4 |
| `last_` | 4 |
| `bb_` | 4 |
| `pd_` | 4 |
| `body_` | 3 |
| `aroon_` | 3 |
| `volatility_` | 3 |
| `sweep_` | 3 |
| `magnet_` | 3 |
| `upper_` | 2 |
| `is_` | 2 |
| `consecutive_` | 2 |
| `pivot_` | 2 |
| `major_` | 2 |
| `minor_` | 2 |
| `stochastic_` | 2 |
| `macd_` | 2 |
| `ema_` | 2 |
| `normalized_` | 2 |
| `kc_` | 2 |
| `dc_` | 2 |
| `ibos_` | 2 |
| `ichoch_` | 2 |
| `bos_` | 2 |
| `choch_` | 2 |
| `eqh_` | 2 |
| `eql_` | 2 |
| `momentum_` | 2 |
| `touch_` | 2 |
| `liq_` | 2 |
| `target_` | 2 |
| `tick_` | 1 |
| `real_` | 1 |
| `total_` | 1 |
| `true_` | 1 |
| `doji_` | 1 |
| `marubozu_` | 1 |
| `inside_` | 1 |
| `outside_` | 1 |
| `williams_` | 1 |
| `log_` | 1 |
| `simple_` | 1 |
| `fwd_` | 1 |
| `plus_` | 1 |
| `minus_` | 1 |
| `parabolic_` | 1 |
| `chaikin_` | 1 |
| `structure_` | 1 |
| `bars_` | 1 |
| `percentile_` | 1 |
| `approximate_` | 1 |
| `efficiency_` | 1 |
| `fractal_` | 1 |
| `market_` | 1 |
| `directional_` | 1 |
| `mean_` | 1 |
| `realized_` | 1 |
| `historical_` | 1 |
| `atr_` | 1 |
| `bullish_` | 1 |
| `bearish_` | 1 |
| `strong_` | 1 |
| `weak_` | 1 |
| `confirmed_` | 1 |
| `num_` | 1 |
| `return_` | 1 |
| `noise_` | 1 |
| `regime_` | 1 |
| `buy_` | 1 |
| `sell_` | 1 |
| `cluster_` | 1 |
| `proximity_` | 1 |
| `age_` | 1 |
| `ranking_` | 1 |

### Cleanest Features (by NaN rate)

| Feature | Dtype | NaN Rate | Valid Rows |
|---|---|---|---|
| `open` | float64 | 0.0% | 87,502 |
| `high` | float64 | 0.0% | 87,502 |
| `low` | float64 | 0.0% | 87,502 |
| `close` | float64 | 0.0% | 87,502 |
| `tick_volume` | int64 | 0.0% | 87,502 |
| `spread` | int64 | 0.0% | 87,502 |
| `real_volume` | int64 | 0.0% | 87,502 |
| `w1_timestamp` | datetime64[ns, UTC] | 0.0% | 87,502 |
| `w1_open` | float64 | 0.0% | 87,502 |
| `w1_high` | float64 | 0.0% | 87,502 |
| `w1_low` | float64 | 0.0% | 87,502 |
| `w1_close` | float64 | 0.0% | 87,502 |
| `w1_tick_volume` | int64 | 0.0% | 87,502 |
| `d1_timestamp` | datetime64[ns, UTC] | 0.0% | 87,502 |
| `d1_open` | float64 | 0.0% | 87,502 |
| `d1_high` | float64 | 0.0% | 87,502 |
| `d1_low` | float64 | 0.0% | 87,502 |
| `d1_close` | float64 | 0.0% | 87,502 |
| `d1_tick_volume` | int64 | 0.0% | 87,502 |
| `h4_timestamp` | datetime64[ns, UTC] | 0.0% | 87,502 |

---

## Label Summary

| Label Column | Dtype | NaN Rate | Valid Rows | Value Range |
|---|---|---|---|---|
| `fwd_return_1b` | float64 | 0.0% | 87,502 | [-0.01147, 0.01602] |
| `direction_1b` | float64 | 0.0% | 87,502 | Classes: 0.0, 1.0, 2.0 |
| `bias_1b` | float64 | 0.0% | 87,502 | Classes: 0.0, 1.0 |
| `confidence_1b` | float64 | 0.0% | 87,483 | [0, 1] |
| `probability_1b` | float64 | 0.0% | 87,483 | [4.54e-05, 1] |
| `fwd_return_3b` | float64 | 0.0% | 87,500 | [-0.0155, 0.01771] |
| `direction_3b` | float64 | 0.0% | 87,500 | Classes: 0.0, 1.0, 2.0 |
| `bias_3b` | float64 | 0.0% | 87,500 | Classes: 0.0, 1.0 |
| `confidence_3b` | float64 | 0.0% | 87,481 | [0, 1] |
| `probability_3b` | float64 | 0.0% | 87,481 | [4.54e-05, 1] |
| `fwd_return_5b` | float64 | 0.0% | 87,498 | [-0.01653, 0.0189] |
| `direction_5b` | float64 | 0.0% | 87,498 | Classes: 0.0, 1.0, 2.0 |
| `bias_5b` | float64 | 0.0% | 87,498 | Classes: 0.0, 1.0 |
| `confidence_5b` | float64 | 0.0% | 87,479 | [0, 1] |
| `probability_5b` | float64 | 0.0% | 87,479 | [4.54e-05, 0.9999] |
| `fwd_return_10b` | float64 | 0.0% | 87,493 | [-0.01608, 0.0223] |
| `direction_10b` | float64 | 0.0% | 87,493 | Classes: 0.0, 1.0, 2.0 |
| `bias_10b` | float64 | 0.0% | 87,493 | Classes: 0.0, 1.0 |
| `confidence_10b` | float64 | 0.0% | 87,474 | [0, 1] |
| `probability_10b` | float64 | 0.0% | 87,474 | [0.0002302, 0.9997] |
| `long_outcome` | float64 | 0.1% | 87,447 | Classes: 0.0, 1.0, 2.0 |
| `long_outcome_bars` | float64 | 0.1% | 87,447 | [1, 50] |
| `long_mfe_pct` | float64 | 0.1% | 87,447 | [0, 0.01795] |
| `long_mae_pct` | float64 | 0.1% | 87,447 | [0, 0.0127] |
| `long_rr` | float64 | 0.1% | 87,447 | [2, 2] |
| `short_outcome` | float64 | 0.1% | 87,447 | Classes: 0.0, 1.0, 2.0 |
| `short_outcome_bars` | float64 | 0.1% | 87,447 | [1, 50] |
| `short_mfe_pct` | float64 | 0.1% | 87,447 | [0, 0.01394] |
| `short_mae_pct` | float64 | 0.1% | 87,447 | [0, 0.01684] |
| `short_rr` | float64 | 0.1% | 87,447 | [2, 2] |
| `outcome` | float64 | 0.1% | 87,447 | Classes: 0.0, 1.0, 2.0 |
| `outcome_bars` | float64 | 0.1% | 87,447 | [1, 50] |
| `mfe_pct` | float64 | 0.1% | 87,447 | [0, 0.01795] |
| `mae_pct` | float64 | 0.1% | 87,447 | [0, 0.01684] |
| `realized_rr` | float64 | 0.1% | 87,447 | [2, 2] |
| `expected_reward_pct` | float64 | 0.1% | 87,447 | [0.0002326, 0.007444] |
| `expected_risk_pct` | float64 | 0.1% | 87,447 | [0.0001163, 0.003722] |
| `trade_duration_bars` | float64 | 0.1% | 87,447 | [1, 50] |
| `setup_quality` | float64 | 0.1% | 87,447 | Classes: 0.0, 1.0, 2.0, 3.0 |
| `setup_score` | float64 | 0.1% | 87,447 | [0, 100] |
| `setup_mfe_mae_ratio` | float64 | 0.0% | 87,496 | [0, 1.218e+06] |
| `setup_achievable_rr` | float64 | 0.0% | 87,496 | [0, 9.617] |
| `entry_signal` | float64 | 0.1% | 87,447 | Classes: 0.0, 1.0, 2.0 |
| `optimal_entry_offset` | float64 | 0.1% | 87,447 | Classes: -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0 |
| `time_to_entry` | float64 | 1.7% | 86,015 | Classes: 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0 |
| `is_optimal_entry` | float64 | 0.1% | 87,453 | Classes: 0.0, 1.0 |
| `mgmt_strategy` | float64 | 0.1% | 87,447 | Classes: 0.0, 1.0, 3.0 |
| `mgmt_optimal_exit_bar` | float64 | 0.1% | 87,447 | [0, 49] |
| `mgmt_max_r_multiple` | float64 | 0.1% | 87,447 | [-3.79, 12.99] |
| `mgmt_breakeven_bar` | float64 | 0.1% | 87,447 | [-1, 30] |
| `mgmt_trail_bar` | float64 | 0.1% | 87,447 | [-1, 48] |
| `mgmt_partial_exit_bar` | float64 | 0.1% | 87,447 | [-1, 44] |
| `mgmt_exit_type` | float64 | 0.1% | 87,447 | Classes: 0.0, 1.0, 2.0, 3.0 |

---

## Class Distributions

### `direction_1b`

| Class | Count | Proportion |
|---|---|---|
| 0 | 16,317 | 18.65% |
| 1 | 54,777 | 62.60% |
| 2 | 16,408 | 18.75% |

### `bias_1b`

| Class | Count | Proportion |
|---|---|---|
| 0 | 43,968 | 50.25% |
| 1 | 43,534 | 49.75% |

### `direction_3b`

| Class | Count | Proportion |
|---|---|---|
| 0 | 24,672 | 28.20% |
| 1 | 37,721 | 43.11% |
| 2 | 25,107 | 28.69% |

### `bias_3b`

| Class | Count | Proportion |
|---|---|---|
| 0 | 43,499 | 49.71% |
| 1 | 44,001 | 50.29% |

### `direction_5b`

| Class | Count | Proportion |
|---|---|---|
| 0 | 28,164 | 32.19% |
| 1 | 30,458 | 34.81% |
| 2 | 28,876 | 33.00% |

### `bias_5b`

| Class | Count | Proportion |
|---|---|---|
| 0 | 43,337 | 49.53% |
| 1 | 44,161 | 50.47% |

### `direction_10b`

| Class | Count | Proportion |
|---|---|---|
| 0 | 32,035 | 36.61% |
| 1 | 22,015 | 25.16% |
| 2 | 33,443 | 38.22% |

### `bias_10b`

| Class | Count | Proportion |
|---|---|---|
| 0 | 43,165 | 49.34% |
| 1 | 44,328 | 50.66% |

### `long_outcome`

| Class | Count | Proportion |
|---|---|---|
| 0 | 1,488 | 1.70% |
| 1 | 27,708 | 31.69% |
| 2 | 58,251 | 66.61% |

### `short_outcome`

| Class | Count | Proportion |
|---|---|---|
| 0 | 1,478 | 1.69% |
| 1 | 27,881 | 31.88% |
| 2 | 58,088 | 66.43% |

### `outcome`

| Class | Count | Proportion |
|---|---|---|
| 0 | 1,829 | 2.09% |
| 1 | 41,335 | 47.27% |
| 2 | 44,283 | 50.64% |

### `setup_quality`

| Class | Count | Proportion |
|---|---|---|
| 0 | 34,814 | 39.81% |
| 1 | 11,249 | 12.86% |
| 2 | 49 | 0.06% |
| 3 | 41,335 | 47.27% |

### `entry_signal`

| Class | Count | Proportion |
|---|---|---|
| 0 | 1,432 | 1.64% |
| 1 | 49,875 | 57.03% |
| 2 | 36,140 | 41.33% |

### `optimal_entry_offset`

| Class | Count | Proportion |
|---|---|---|
| -1 | 1,432 | 1.64% |
| 0 | 36,140 | 41.33% |
| 1 | 4,370 | 5.00% |
| 2 | 4,095 | 4.68% |
| 3 | 4,190 | 4.79% |
| 4 | 4,376 | 5.00% |
| 5 | 4,567 | 5.22% |
| 6 | 4,989 | 5.71% |
| 7 | 5,838 | 6.68% |
| 8 | 7,367 | 8.42% |
| 9 | 10,083 | 11.53% |

### `time_to_entry`

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

### `is_optimal_entry`

| Class | Count | Proportion |
|---|---|---|
| 0 | 49,935 | 57.10% |
| 1 | 37,518 | 42.90% |

### `mgmt_strategy`

| Class | Count | Proportion |
|---|---|---|
| 0 | 51 | 0.06% |
| 1 | 43,113 | 49.30% |
| 3 | 44,283 | 50.64% |

### `mgmt_exit_type`

| Class | Count | Proportion |
|---|---|---|
| 0 | 41,335 | 47.27% |
| 1 | 44,177 | 50.52% |
| 2 | 106 | 0.12% |
| 3 | 1,829 | 2.09% |

---

## Missing Values

Columns with > 1 % missing values:

| Column | NaN Count | NaN Rate |
|---|---|---|
| `fvg_bearish_top` | 61,555 | 70.35% |
| `fvg_bearish_bottom` | 61,555 | 70.35% |
| `fvg_bullish_top` | 60,176 | 68.77% |
| `fvg_bullish_bottom` | 60,176 | 68.77% |
| `ob_bullish_top` | 51,682 | 59.06% |
| `ob_bullish_bottom` | 51,682 | 59.06% |
| `ob_bearish_top` | 50,815 | 58.07% |
| `ob_bearish_bottom` | 50,815 | 58.07% |
| `time_to_entry` | 1,487 | 1.70% |

---

## Validation Results

```
DatasetValidation: PASSED — 6 passed, 9 warnings, 0 failures
  [WARNING] feature_nan_rate: Feature 'fvg_bullish_top': 68.8% NaN > 50.0%.
  [WARNING] feature_nan_rate: Feature 'fvg_bullish_bottom': 68.8% NaN > 50.0%.
  [WARNING] feature_nan_rate: Feature 'fvg_bearish_top': 70.3% NaN > 50.0%.
  [WARNING] feature_nan_rate: Feature 'fvg_bearish_bottom': 70.3% NaN > 50.0%.
  [WARNING] feature_nan_rate: Feature 'ob_bullish_top': 59.1% NaN > 50.0%.
  [WARNING] feature_nan_rate: Feature 'ob_bullish_bottom': 59.1% NaN > 50.0%.
  [WARNING] feature_nan_rate: Feature 'ob_bearish_top': 58.1% NaN > 50.0%.
  [WARNING] feature_nan_rate: Feature 'ob_bearish_bottom': 58.1% NaN > 50.0%.
  [WARNING] dtype_consistency: Non-numeric features: ['w1_timestamp', 'd1_timestamp', 'h4_timestamp', 'h1_timestamp']
```

---

## Selected Features

Total: **252** features

```
open
high
low
close
tick_volume
spread
real_volume
w1_timestamp
w1_open
w1_high
w1_low
w1_close
w1_tick_volume
d1_timestamp
d1_open
d1_high
d1_low
d1_close
d1_tick_volume
h4_timestamp
h4_open
h4_high
h4_low
h4_close
h4_tick_volume
h1_timestamp
h1_open
h1_high
h1_low
h1_close
h1_tick_volume
body_size
body_ratio
upper_wick
lower_wick
upper_wick_ratio
lower_wick_ratio
total_range
true_range
body_to_range_ratio
is_bullish
is_bearish
doji_score
marubozu_score
inside_bar
outside_bar
consecutive_bulls
consecutive_bears
higher_close_count
lower_close_count
higher_high_count
lower_low_count
fvg_bullish
fvg_bearish
fvg_bullish_top
fvg_bullish_bottom
fvg_bearish_top
fvg_bearish_bottom
fvg_bullish_active
fvg_bearish_active
fvg_bullish_age
fvg_bearish_age
pivot_high
pivot_low
major_pivot_high
major_pivot_low
minor_pivot_high
minor_pivot_low
higher_high
lower_high
higher_low
lower_low
swing_high_id
swing_low_id
swing_high_price
swing_low_price
swing_high_duration
swing_low_duration
swing_high_range
swing_low_range
swing_high_strength
swing_low_strength
trend
trend_duration
trend_strength
last_major_high
last_major_low
last_internal_high
last_internal_low
distance_to_last_major_high
distance_to_last_major_low
distance_to_last_internal_high
distance_to_last_internal_low
rsi
stochastic_k
stochastic_d
macd
macd_signal
macd_histogram
cci
williams_r
roc
price_momentum
tsi
ema9
ema20
ema50
ema100
ema200
sma20
sma50
sma100
wma20
hma20
ema_slope
ema_cross
log_return
simple_return
rolling_return_5
rolling_return_20
fwd_return_1
rolling_mean
rolling_median
rolling_var
rolling_std
rolling_min
rolling_max
rolling_q25
rolling_q75
rolling_mad
adx
plus_di
minus_di
aroon_up
aroon_down
aroon_oscillator
parabolic_sar
atr
normalized_atr
bb_upper
bb_lower
bb_width
bb_percent_b
kc_upper
kc_lower
dc_upper
dc_lower
chaikin_volatility
ibos_bullish
ibos_bearish
ichoch_bullish
ichoch_bearish
bos_bullish
bos_bearish
choch_bullish
choch_bearish
structure_bias
bars_since_structure_break
eqh
eqh_price
eql
eql_price
eqh_age
eql_age
pd_ratio
pd_equilibrium
pd_distance_from_eq
pd_zone
skewness
kurtosis
zscore
percentile_rank
normalized_price
price_rank
entropy
rolling_entropy_5
approximate_entropy
efficiency_ratio
hurst
fractal_dimension
market_noise
directional_efficiency
price_smoothness
mean_reversion_score
trend_score
price_velocity
price_acceleration
price_deceleration
rolling_momentum_5
rolling_momentum_20
momentum_persistence
trend_persistence
realized_volatility
historical_volatility
volatility_expansion
volatility_compression
atr_ratio
rolling_atr_20
volatility_regime
ob_bullish
ob_bearish
ob_bullish_top
ob_bullish_bottom
ob_bearish_top
ob_bearish_bottom
ob_bullish_active
ob_bearish_active
price_in_bullish_ob
price_in_bearish_ob
bullish_liquidity_sweep
bearish_liquidity_sweep
liquidity_score
nearest_liquidity_distance
nearest_buy_liquidity
nearest_sell_liquidity
liquidity_age
touch_count
strong_sweep
weak_sweep
confirmed_sweep
sweep_strength
liquidity_cluster_size
sweep_penetration
sweep_rejection
liq_zone_width
liq_zone_lifetime
num_nearby_liq_pools
return_vol_ratio
trend_quality
noise_ratio
price_efficiency
regime_consistency
nearest_buy_liquidity_distance
nearest_sell_liquidity_distance
nearest_liquidity_score
magnet_score
magnet_probability
liquidity_rank
target_liquidity
distance_to_target
buy_side_probability
sell_side_probability
liquidity_density
cluster_strength
magnet_strength
nearest_cluster_size
proximity_contribution
age_contribution
touch_contribution
momentum_contribution
ranking_position
target_direction
```

---

## Selected Labels

Total: **53** label columns  
Groups: all

```
fwd_return_1b
direction_1b
bias_1b
confidence_1b
probability_1b
fwd_return_3b
direction_3b
bias_3b
confidence_3b
probability_3b
fwd_return_5b
direction_5b
bias_5b
confidence_5b
probability_5b
fwd_return_10b
direction_10b
bias_10b
confidence_10b
probability_10b
long_outcome
long_outcome_bars
long_mfe_pct
long_mae_pct
long_rr
short_outcome
short_outcome_bars
short_mfe_pct
short_mae_pct
short_rr
outcome
outcome_bars
mfe_pct
mae_pct
realized_rr
expected_reward_pct
expected_risk_pct
trade_duration_bars
setup_quality
setup_score
setup_mfe_mae_ratio
setup_achievable_rr
entry_signal
optimal_entry_offset
time_to_entry
is_optimal_entry
mgmt_strategy
mgmt_optimal_exit_bar
mgmt_max_r_multiple
mgmt_breakeven_bar
mgmt_trail_bar
mgmt_partial_exit_bar
mgmt_exit_type
```

---

> **Important**: Labels are strictly forward-looking targets.
> Never use label columns as model input features.
> Always use time-series cross-validation (walk-forward) to avoid look-ahead bias.