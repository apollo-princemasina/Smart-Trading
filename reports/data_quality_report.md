# Data Quality Report — EURUSD

Generated: 2026-07-01 06:13 UTC  
Source: MetaTrader 5 / MetaQuotes-Demo  
Pipeline: ICT + ML Smart Trading Preprocessing


## 1. Raw OHLCV Validation

| TF | Rows | Dupes | OHLC Err | Neg Price | Neg Vol | Large Spread | Const | Gaps | Status |
|---|---|---|---|---|---|---|---|---|---|
| W1 | 470 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **PASS** |
| D1 | 2,336 | 0 | 0 | 0 | 0 | 0 | 0 | 5 | **PASS** |
| H4 | 13,994 | 0 | 0 | 0 | 0 | 0 | 0 | 5 | **PASS** |
| H1 | 55,782 | 0 | 0 | 0 | 0 | 0 | 1 | 5 | **PASS** |
| M15 | 87,503 | 0 | 0 | 0 | 0 | 0 | 11 | 1 | **PASS** |

- [D1 WARN] 5 unexpected time gaps > 3 days 00:00:00 (excluding weekend windows).
- [H4 WARN] 5 unexpected time gaps > 0 days 12:00:00 (excluding weekend windows).
- [H1 WARN] 1 constant candles (O=H=L=C). May be synthetic filler bars.
- [H1 WARN] 5 unexpected time gaps > 0 days 03:00:00 (excluding weekend windows).
- [M15 WARN] 11 constant candles (O=H=L=C). May be synthetic filler bars.
- [M15 WARN] 1 unexpected time gaps > 0 days 00:45:00 (excluding weekend windows).

## 2. Cleaning Summary

| TF | Input | Output | Removed | Dupes | Bad OHLC | Zero Price | Sorted | UTC |
|---|---|---|---|---|---|---|---|---|
| W1 | 470 | 470 | 0 | 0 | 0 | 0 | no | already |
| D1 | 2,336 | 2,336 | 0 | 0 | 0 | 0 | no | already |
| H4 | 13,994 | 13,994 | 0 | 0 | 0 | 0 | no | already |
| H1 | 55,782 | 55,782 | 0 | 0 | 0 | 0 | no | already |
| M15 | 87,503 | 87,503 | 0 | 0 | 0 | 0 | no | already |


## 3. Market Calendar Validation

| TF | Weekend Candles | Expected Gaps | Unexpected Gaps | Thin-Market Rows |
|---|---|---|---|---|
| W1 | 470 | 0 | 0 | 4 |
| D1 | 0 | 0 | 0 | 9 |
| H4 | 0 | 472 | 5 | 39 |
| H1 | 0 | 472 | 12 | 144 |
| M15 | 0 | 187 | 8 | 357 |

- [W1] 0 Saturday + 470 early-Sunday candles found (market closed; likely synthetic filler data).
- [W1] 4 rows on known thin-market dates (Christmas / New Year). Spreads are typically wider.
- [D1] 9 rows on known thin-market dates (Christmas / New Year). Spreads are typically wider.
- [H4] 39 rows on known thin-market dates (Christmas / New Year). Spreads are typically wider.
- [H4] 5 unexpected intra-week gaps > 0 days 16:00:00. Check broker connectivity or missing data around these periods.
- [H1] 144 rows on known thin-market dates (Christmas / New Year). Spreads are typically wider.
- [H1] 12 unexpected intra-week gaps > 0 days 04:00:00. Check broker connectivity or missing data around these periods.
- [M15] 357 rows on known thin-market dates (Christmas / New Year). Spreads are typically wider.
- [M15] 8 unexpected intra-week gaps > 0 days 01:00:00. Check broker connectivity or missing data around these periods.

## 4. Cross-Timeframe Consistency

| Pair | Periods | Open Err | High Err | Low Err | Close Err | Vol Err | Incomplete | Status |
|---|---|---|---|---|---|---|---|---|
| M15->H1 | 21,888 | 1 | 4 | 1 | 3 | 4 | 23 | **FAIL** |
| H1->H4 | 13,969 | 5 | 5 | 3 | 6 | 9 | 63 | **FAIL** |
| H4->D1 | 2,336 | 0 | 0 | 0 | 0 | 0 | 11 | **PASS** |

- **[M15->H1 ERROR]** 1 open mismatches between aggregated M15 and H1. Sample timestamps: [Timestamp('2022-06-21 18:00:00+0000', tz='UTC')]
- **[M15->H1 ERROR]** 4 high mismatches between aggregated M15 and H1. Sample timestamps: [Timestamp('2022-06-21 18:00:00+0000', tz='UTC'), Timestamp('2022-12-30 22:00:00+0000', tz='UTC'), Timestamp('2024-12-30 22:00:00+0000', tz='UTC')]
- **[M15->H1 ERROR]** 1 low mismatches between aggregated M15 and H1. Sample timestamps: [Timestamp('2025-12-30 22:00:00+0000', tz='UTC')]
- **[M15->H1 ERROR]** 3 close mismatches between aggregated M15 and H1. Sample timestamps: [Timestamp('2022-12-30 22:00:00+0000', tz='UTC'), Timestamp('2024-12-30 22:00:00+0000', tz='UTC'), Timestamp('2025-12-30 22:00:00+0000', tz='UTC')]
- [M15->H1 WARN] 23 periods have fewer lower-TF candles than expected (4). Missing candles cause slight OHLC deviation.
- [M15->H1 WARN] 4 periods with tick_volume deviation > 10% between aggregated M15 and H1. Tick_volume sampling differs per timeframe in MT5.
- **[H1->H4 ERROR]** 5 open mismatches between aggregated H1 and H4. Sample timestamps: [Timestamp('2018-12-31 20:00:00+0000', tz='UTC'), Timestamp('2019-12-31 20:00:00+0000', tz='UTC'), Timestamp('2020-12-31 20:00:00+0000', tz='UTC')]
- **[H1->H4 ERROR]** 5 high mismatches between aggregated H1 and H4. Sample timestamps: [Timestamp('2019-12-31 20:00:00+0000', tz='UTC'), Timestamp('2020-12-30 20:00:00+0000', tz='UTC'), Timestamp('2020-12-31 20:00:00+0000', tz='UTC')]
- **[H1->H4 ERROR]** 3 low mismatches between aggregated H1 and H4. Sample timestamps: [Timestamp('2018-12-31 20:00:00+0000', tz='UTC'), Timestamp('2021-12-31 20:00:00+0000', tz='UTC'), Timestamp('2024-12-31 20:00:00+0000', tz='UTC')]
- **[H1->H4 ERROR]** 6 close mismatches between aggregated H1 and H4. Sample timestamps: [Timestamp('2019-12-30 20:00:00+0000', tz='UTC'), Timestamp('2020-12-30 20:00:00+0000', tz='UTC'), Timestamp('2021-12-30 20:00:00+0000', tz='UTC')]
- [H1->H4 WARN] 63 periods have fewer lower-TF candles than expected (4). Missing candles cause slight OHLC deviation.
- [H1->H4 WARN] 9 periods with tick_volume deviation > 10% between aggregated H1 and H4. Tick_volume sampling differs per timeframe in MT5.
- [H4->D1 WARN] 11 periods have fewer lower-TF candles than expected (6). Missing candles cause slight OHLC deviation.

## 5. Multi-Timeframe Merge

- Base timeframe: **M15**
- Higher TFs attached: **4**
- Base rows: 87,503
- Merged rows: 87,503

**Rows without a completed HTF candle (warm-up period):**

  - W1: 0 rows
  - D1: 0 rows
  - H4: 0 rows
  - H1: 0 rows

## 6. Overall Assessment

**Issues requiring attention before feature engineering:**

- Cross-TF consistency FAILED for M15->H1
- Cross-TF consistency FAILED for H1->H4

Total rows cleaned (removed): **0**