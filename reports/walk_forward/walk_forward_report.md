# Walk-Forward Validation Report
**Symbol:** EURUSD  
**Windows generated:** 6  
**Window type:** rolling  

## Configuration

| Parameter | Value |
|-----------|-------|
| window_type | rolling |
| train_period | 18m |
| val_period | 6m |
| test_period | 3m |
| step_period | 3m |
| anchor_date | None |
| gap_bars | 0 |
| min_train_samples | 5000 |
| min_val_samples | 2000 |
| min_test_samples | 1000 |
| max_windows | 0 |
| validate | True |

## Dataset Coverage

- **Full span:** 2022-06-21T18:30:00 → 2025-12-19T23:45:00
- **Total windows:** 6
- **Validation passed:** 6/6

## Window Summary

| # | Train Start | Train End | Val Start | Val End | Test Start | Test End | Train rows | Val rows | Test rows | Valid |
|---|------------|-----------|-----------|---------|------------|----------|-----------|----------|-----------|-------|
| 000 | 2022-06-21 | 2023-12-21 | 2023-12-21   | 2024-06-21 | 2024-06-21  | 2024-09-20 | 37,537 | 12,379 | 6,215 | ✓ |
| 001 | 2022-09-21 | 2024-03-21 | 2024-03-21   | 2024-09-20 | 2024-09-23  | 2024-12-20 | 37,252 | 12,543 | 6,236 | ✓ |
| 002 | 2022-12-21 | 2024-06-21 | 2024-06-21   | 2024-12-20 | 2024-12-23  | 2025-03-21 | 37,344 | 12,451 | 5,932 | ✓ |
| 003 | 2023-03-21 | 2024-09-20 | 2024-09-23   | 2025-03-21 | 2025-03-24  | 2025-06-23 | 37,490 | 12,168 | 6,325 | ✓ |
| 004 | 2023-06-21 | 2024-12-20 | 2024-12-23   | 2025-06-20 | 2025-06-23  | 2025-09-22 | 37,394 | 12,161 | 6,275 | ✓ |
| 005 | 2023-09-21 | 2025-03-21 | 2025-03-21   | 2025-09-19 | 2025-09-22  | 2025-12-19 | 36,980 | 12,427 | 6,226 | ✓ |

## Aggregate Statistics

### Train
- Min rows: 36,980
- Max rows: 37,537
- Mean rows: 37,333

### Validation
- Min rows: 12,161
- Max rows: 12,543
- Mean rows: 12,355

### Test
- Min rows: 5,932
- Max rows: 6,325
- Mean rows: 6,202

## Artefact Paths

**Window 000**
- `train`: `C:\Users\ndlov\Documents\Research and Innovation\Smart Trading\data\ml\windows\window_000\train.parquet`
- `validation`: `C:\Users\ndlov\Documents\Research and Innovation\Smart Trading\data\ml\windows\window_000\validation.parquet`
- `test`: `C:\Users\ndlov\Documents\Research and Innovation\Smart Trading\data\ml\windows\window_000\test.parquet`

**Window 001**
- `train`: `C:\Users\ndlov\Documents\Research and Innovation\Smart Trading\data\ml\windows\window_001\train.parquet`
- `validation`: `C:\Users\ndlov\Documents\Research and Innovation\Smart Trading\data\ml\windows\window_001\validation.parquet`
- `test`: `C:\Users\ndlov\Documents\Research and Innovation\Smart Trading\data\ml\windows\window_001\test.parquet`

**Window 002**
- `train`: `C:\Users\ndlov\Documents\Research and Innovation\Smart Trading\data\ml\windows\window_002\train.parquet`
- `validation`: `C:\Users\ndlov\Documents\Research and Innovation\Smart Trading\data\ml\windows\window_002\validation.parquet`
- `test`: `C:\Users\ndlov\Documents\Research and Innovation\Smart Trading\data\ml\windows\window_002\test.parquet`

**Window 003**
- `train`: `C:\Users\ndlov\Documents\Research and Innovation\Smart Trading\data\ml\windows\window_003\train.parquet`
- `validation`: `C:\Users\ndlov\Documents\Research and Innovation\Smart Trading\data\ml\windows\window_003\validation.parquet`
- `test`: `C:\Users\ndlov\Documents\Research and Innovation\Smart Trading\data\ml\windows\window_003\test.parquet`

**Window 004**
- `train`: `C:\Users\ndlov\Documents\Research and Innovation\Smart Trading\data\ml\windows\window_004\train.parquet`
- `validation`: `C:\Users\ndlov\Documents\Research and Innovation\Smart Trading\data\ml\windows\window_004\validation.parquet`
- `test`: `C:\Users\ndlov\Documents\Research and Innovation\Smart Trading\data\ml\windows\window_004\test.parquet`

**Window 005**
- `train`: `C:\Users\ndlov\Documents\Research and Innovation\Smart Trading\data\ml\windows\window_005\train.parquet`
- `validation`: `C:\Users\ndlov\Documents\Research and Innovation\Smart Trading\data\ml\windows\window_005\validation.parquet`
- `test`: `C:\Users\ndlov\Documents\Research and Innovation\Smart Trading\data\ml\windows\window_005\test.parquet`
