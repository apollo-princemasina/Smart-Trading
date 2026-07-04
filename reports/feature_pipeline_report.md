# Feature Pipeline Report — EURUSD

Generated: 2026-07-01 20:48 UTC  
Total pipeline time: 150657 ms  
Python: 3.11.9

---

## 1. Registered Features

| Category | Feature Name | Dependencies | Status |
|---|---|---|---|
| labels | labels_placeholder | — | enabled |
| liquidity | equal_highs_lows | market_structure | enabled |
| liquidity | liquidity_magnet | market_structure, bos_choch, equal_highs_lows, liquidity_sweeps | enabled |
| liquidity | liquidity_placeholder | — | enabled |
| liquidity | liquidity_sweeps | market_structure, bos_choch, equal_highs_lows | enabled |
| market_structure | bos_choch | market_structure | enabled |
| market_structure | fair_value_gaps | — | enabled |
| market_structure | market_structure | — | enabled |
| market_structure | market_structure_placeholder | — | enabled |
| market_structure | order_blocks | bos_choch | enabled |
| market_structure | premium_discount | market_structure | enabled |
| momentum | momentum_placeholder | — | enabled |
| sessions | sessions | — | enabled |
| sessions | sessions_placeholder | — | enabled |
| statistics | candle_statistics | — | enabled |
| statistics | distribution | returns | enabled |
| statistics | entropy | returns | enabled |
| statistics | market_microstructure | returns | enabled |
| statistics | momentum_stats | returns | enabled |
| statistics | returns | — | enabled |
| statistics | rolling_statistics | — | enabled |
| statistics | statistics | returns, rolling_statistics, distribution, candle_statistics, momentum_stats, volatility_stats, entropy, market_microstructure | enabled |
| statistics | statistics_placeholder | — | enabled |
| statistics | volatility_stats | returns, volatility | enabled |
| technical | momentum | — | enabled |
| technical | moving_averages | — | enabled |
| technical | oscillators | — | enabled |
| technical | technical | moving_averages, momentum, trend, volatility, oscillators | enabled |
| technical | technical_placeholder | — | enabled |
| technical | trend | — | enabled |
| technical | volatility | — | enabled |
| trend | trend_placeholder | — | enabled |
| volatility | volatility_placeholder | — | enabled |
| volume | premium_discount_delta_volume | premium_discount | enabled |
| volume | volume_placeholder | — | enabled |

## 2. Execution Order (dependency-sorted)

1. **candle_statistics** `[statistics]`
2. **fair_value_gaps** `[market_structure]`
3. **labels_placeholder** `[labels]`
4. **liquidity_placeholder** `[liquidity]`
5. **market_structure** `[market_structure]`
6. **market_structure_placeholder** `[market_structure]`
7. **momentum** `[technical]`
8. **momentum_placeholder** `[momentum]`
9. **moving_averages** `[technical]`
10. **oscillators** `[technical]`
11. **returns** `[statistics]`
12. **rolling_statistics** `[statistics]`
13. **sessions** `[sessions]`
14. **sessions_placeholder** `[sessions]`
15. **statistics_placeholder** `[statistics]`
16. **technical_placeholder** `[technical]`
17. **trend** `[technical]`
18. **trend_placeholder** `[trend]`
19. **volatility** `[technical]`
20. **volatility_placeholder** `[volatility]`
21. **volume_placeholder** `[volume]`
22. **bos_choch** `[market_structure]`
23. **equal_highs_lows** `[liquidity]`
24. **premium_discount** `[market_structure]`
25. **distribution** `[statistics]`
26. **entropy** `[statistics]`
27. **market_microstructure** `[statistics]`
28. **momentum_stats** `[statistics]`
29. **technical** `[technical]`
30. **volatility_stats** `[statistics]`
31. **order_blocks** `[market_structure]`
32. **liquidity_sweeps** `[liquidity]`
33. **premium_discount_delta_volume** `[volume]`
34. **statistics** `[statistics]`
35. **liquidity_magnet** `[liquidity]`

## 3. Execution Summary

| # | Feature | Category | Status | Columns | NaN | Inf | Time (ms) |
|---|---|---|---|---|---|---|---|
| 1 | candle_statistics | statistics | ok | 21 | 0 | 0 | 113.41 |
| 2 | fair_value_gaps | market_structure | ok | 10 | 243464 | 0 | 490.3 |
| 3 | labels_placeholder | labels | ok | 0 | 0 | 0 | 0.0 |
| 4 | liquidity_placeholder | liquidity | ok | 0 | 0 | 0 | 0.0 |
| 5 | market_structure | market_structure | ok | 31 | 212 | 0 | 317.53 |
| 6 | market_structure_placeholder | market_structure | ok | 0 | 0 | 0 | 0.0 |
| 7 | momentum | technical | ok | 11 | 0 | 0 | 1635.52 |
| 8 | momentum_placeholder | momentum | ok | 0 | 0 | 0 | 0.0 |
| 9 | moving_averages | technical | ok | 12 | 0 | 0 | 54.05 |
| 10 | oscillators | — | skipped | — | — | — | 0 |
| 11 | returns | statistics | ok | 5 | 0 | 0 | 25.02 |
| 12 | rolling_statistics | statistics | ok | 9 | 0 | 0 | 2359.91 |
| 13 | sessions | — | skipped | — | — | — | 0 |
| 14 | sessions_placeholder | sessions | ok | 0 | 0 | 0 | 0.0 |
| 15 | statistics_placeholder | statistics | ok | 0 | 0 | 0 | 0.44 |
| 16 | technical_placeholder | technical | ok | 0 | 0 | 0 | 0.31 |
| 17 | trend | technical | ok | 7 | 0 | 0 | 965.23 |
| 18 | trend_placeholder | trend | ok | 0 | 0 | 0 | 0.0 |
| 19 | volatility | technical | ok | 11 | 0 | 0 | 105.64 |
| 20 | volatility_placeholder | volatility | ok | 0 | 0 | 0 | 0.0 |
| 21 | volume_placeholder | volume | ok | 0 | 0 | 0 | 0.0 |
| 22 | bos_choch | market_structure | ok | 10 | 0 | 0 | 84.72 |
| 23 | equal_highs_lows | liquidity | ok | 6 | 143 | 0 | 366.57 |
| 24 | premium_discount | market_structure | ok | 4 | 96 | 0 | 13.91 |
| 25 | distribution | statistics | ok | 6 | 0 | 0 | 9012.31 |
| 26 | entropy | statistics | ok | 3 | 0 | 0 | 43456.92 |
| 27 | market_microstructure | statistics | ok | 8 | 0 | 0 | 15349.79 |
| 28 | momentum_stats | statistics | ok | 7 | 0 | 0 | 5882.64 |
| 29 | technical | — | skipped | — | — | — | 0 |
| 30 | volatility_stats | statistics | ok | 7 | 0 | 0 | 29407.78 |
| 31 | order_blocks | market_structure | ok | 10 | 204998 | 0 | 673.91 |
| 32 | liquidity_sweeps | liquidity | ok | 18 | 40 | 0 | 8764.46 |
| 33 | premium_discount_delta_volume | — | skipped | — | — | — | 0 |
| 34 | statistics | statistics | ok | 5 | 0 | 0 | 12.24 |
| 35 | liquidity_magnet | liquidity | ok | 20 | 299 | 0 | 19922.98 |

## 4. Validation Summary

- Total features executed : 31
- Passed                  : 31
- Failed                  : 0
- Total NaN cells         : 449,252
- Total ±Inf cells        : 0
- Overall status          : **PASS**

## 5. Output Dataset

- Rows    : 87,503
- Columns : 253

**Output columns:**

- `timestamp`
- `open`
- `high`
- `low`
- `close`
- `tick_volume`
- `spread`
- `real_volume`
- `w1_timestamp`
- `w1_open`
- `w1_high`
- `w1_low`
- `w1_close`
- `w1_tick_volume`
- `d1_timestamp`
- `d1_open`
- `d1_high`
- `d1_low`
- `d1_close`
- `d1_tick_volume`
- `h4_timestamp`
- `h4_open`
- `h4_high`
- `h4_low`
- `h4_close`
- `h4_tick_volume`
- `h1_timestamp`
- `h1_open`
- `h1_high`
- `h1_low`
- `h1_close`
- `h1_tick_volume`
- `body_size`
- `body_ratio`
- `upper_wick`
- `lower_wick`
- `upper_wick_ratio`
- `lower_wick_ratio`
- `total_range`
- `true_range`
- `body_to_range_ratio`
- `is_bullish`
- `is_bearish`
- `doji_score`
- `marubozu_score`
- `inside_bar`
- `outside_bar`
- `consecutive_bulls`
- `consecutive_bears`
- `higher_close_count`
- `lower_close_count`
- `higher_high_count`
- `lower_low_count`
- `fvg_bullish`
- `fvg_bearish`
- `fvg_bullish_top`
- `fvg_bullish_bottom`
- `fvg_bearish_top`
- `fvg_bearish_bottom`
- `fvg_bullish_active`
- `fvg_bearish_active`
- `fvg_bullish_age`
- `fvg_bearish_age`
- `pivot_high`
- `pivot_low`
- `major_pivot_high`
- `major_pivot_low`
- `minor_pivot_high`
- `minor_pivot_low`
- `higher_high`
- `lower_high`
- `higher_low`
- `lower_low`
- `swing_high_id`
- `swing_low_id`
- `swing_high_price`
- `swing_low_price`
- `swing_high_duration`
- `swing_low_duration`
- `swing_high_range`
- `swing_low_range`
- `swing_high_strength`
- `swing_low_strength`
- `trend`
- `trend_duration`
- `trend_strength`
- `last_major_high`
- `last_major_low`
- `last_internal_high`
- `last_internal_low`
- `distance_to_last_major_high`
- `distance_to_last_major_low`
- `distance_to_last_internal_high`
- `distance_to_last_internal_low`
- `rsi`
- `stochastic_k`
- `stochastic_d`
- `macd`
- `macd_signal`
- `macd_histogram`
- `cci`
- `williams_r`
- `roc`
- `price_momentum`
- `tsi`
- `ema9`
- `ema20`
- `ema50`
- `ema100`
- `ema200`
- `sma20`
- `sma50`
- `sma100`
- `wma20`
- `hma20`
- `ema_slope`
- `ema_cross`
- `log_return`
- `simple_return`
- `rolling_return_5`
- `rolling_return_20`
- `fwd_return_1`
- `rolling_mean`
- `rolling_median`
- `rolling_var`
- `rolling_std`
- `rolling_min`
- `rolling_max`
- `rolling_q25`
- `rolling_q75`
- `rolling_mad`
- `adx`
- `plus_di`
- `minus_di`
- `aroon_up`
- `aroon_down`
- `aroon_oscillator`
- `parabolic_sar`
- `atr`
- `normalized_atr`
- `bb_upper`
- `bb_lower`
- `bb_width`
- `bb_percent_b`
- `kc_upper`
- `kc_lower`
- `dc_upper`
- `dc_lower`
- `chaikin_volatility`
- `ibos_bullish`
- `ibos_bearish`
- `ichoch_bullish`
- `ichoch_bearish`
- `bos_bullish`
- `bos_bearish`
- `choch_bullish`
- `choch_bearish`
- `structure_bias`
- `bars_since_structure_break`
- `eqh`
- `eqh_price`
- `eql`
- `eql_price`
- `eqh_age`
- `eql_age`
- `pd_ratio`
- `pd_equilibrium`
- `pd_distance_from_eq`
- `pd_zone`
- `skewness`
- `kurtosis`
- `zscore`
- `percentile_rank`
- `normalized_price`
- `price_rank`
- `entropy`
- `rolling_entropy_5`
- `approximate_entropy`
- `efficiency_ratio`
- `hurst`
- `fractal_dimension`
- `market_noise`
- `directional_efficiency`
- `price_smoothness`
- `mean_reversion_score`
- `trend_score`
- `price_velocity`
- `price_acceleration`
- `price_deceleration`
- `rolling_momentum_5`
- `rolling_momentum_20`
- `momentum_persistence`
- `trend_persistence`
- `realized_volatility`
- `historical_volatility`
- `volatility_expansion`
- `volatility_compression`
- `atr_ratio`
- `rolling_atr_20`
- `volatility_regime`
- `ob_bullish`
- `ob_bearish`
- `ob_bullish_top`
- `ob_bullish_bottom`
- `ob_bearish_top`
- `ob_bearish_bottom`
- `ob_bullish_active`
- `ob_bearish_active`
- `price_in_bullish_ob`
- `price_in_bearish_ob`
- `bullish_liquidity_sweep`
- `bearish_liquidity_sweep`
- `liquidity_score`
- `nearest_liquidity_distance`
- `nearest_buy_liquidity`
- `nearest_sell_liquidity`
- `liquidity_age`
- `touch_count`
- `strong_sweep`
- `weak_sweep`
- `confirmed_sweep`
- `sweep_strength`
- `liquidity_cluster_size`
- `sweep_penetration`
- `sweep_rejection`
- `liq_zone_width`
- `liq_zone_lifetime`
- `num_nearby_liq_pools`
- `return_vol_ratio`
- `trend_quality`
- `noise_ratio`
- `price_efficiency`
- `regime_consistency`
- `nearest_buy_liquidity_distance`
- `nearest_sell_liquidity_distance`
- `nearest_liquidity_score`
- `magnet_score`
- `magnet_probability`
- `liquidity_rank`
- `target_liquidity`
- `distance_to_target`
- `buy_side_probability`
- `sell_side_probability`
- `liquidity_density`
- `cluster_strength`
- `magnet_strength`
- `nearest_cluster_size`
- `proximity_contribution`
- `age_contribution`
- `touch_contribution`
- `momentum_contribution`
- `ranking_position`
- `target_direction`

## 6. Future Feature Modules

| Category | Planned Indicators |
|---|---|
| market_structure | BOS, CHoCH, MSS, Swing Highs/Lows, Order Blocks, FVGs |
| liquidity        | Liquidity Pools, Equal Highs/Lows, Stop Hunt Detection |
| sessions         | London/NY/Asia markers, Kill Zone flags, Session OHLC |
| trend            | EMA Stack, Trend Bias, Higher-TF Direction |
| volatility       | ATR, Bollinger Bands, Historical Volatility |
| momentum         | RSI, MACD, Stochastic, ADX, Z-Score |
| volume           | Delta Volume, Volume Profile, Cumulative Volume Delta |
| labels           | Triple Barrier Labels, Binary Direction, RR Labels |
