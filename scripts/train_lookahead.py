"""
Train direction_4b (1-hour) and direction_8b (2-hour) lookahead models.

Same 247 features and XGBoost hyperparameters as window_000/xgboost (best_model).
Labels: forward log-return on close with ±0.0003 neutral threshold.
Walk-forward window: identical date split to window_000.

Output:
  models/lookahead_4b/   (1-hour conviction model)
  models/lookahead_8b/   (2-hour conviction model)

Run:
  python scripts/train_lookahead.py
"""
import sys
import json
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from pathlib import Path
from datetime import datetime, timezone
from sklearn.metrics import accuracy_score, f1_score, classification_report

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FEATURE_DATASET = ROOT / "data/features/EURUSD/feature_dataset.parquet"
FEATURE_ORDER_F = ROOT / "models/best_model/feature_order.json"
BEST_META_F     = ROOT / "models/best_model/model_metadata.json"
BEST_PREPROC_F  = ROOT / "models/best_model/preprocessing.joblib"

# Same neutral threshold as src/labels/market_bias.py
NEUTRAL_THRESHOLD = 0.0003

# Same walk-forward window as window_000 (from model_metadata.json)
TRAIN_START = pd.Timestamp("2022-06-21", tz="UTC")
TRAIN_END   = pd.Timestamp("2023-12-21", tz="UTC")   # 18-month train
VAL_END     = pd.Timestamp("2024-06-21", tz="UTC")   # 6-month val
TEST_END    = pd.Timestamp("2024-09-21", tz="UTC")   # 3-month test


def compute_direction(close: pd.Series, horizon: int) -> pd.Series:
    """0=SELL, 1=HOLD, 2=BUY  (matches src/labels/market_bias.py)."""
    fwd_ret = np.log(close.shift(-horizon) / close)
    labels = np.where(fwd_ret > NEUTRAL_THRESHOLD, 2,
                      np.where(fwd_ret < -NEUTRAL_THRESHOLD, 0, 1)).astype(float)
    s = pd.Series(labels, index=close.index)
    s[fwd_ret.isna()] = np.nan
    return s


def save_bundle(model, preproc, feature_order: list, meta: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model,   out_dir / "model.joblib",         compress=3)
    joblib.dump(preproc, out_dir / "preprocessing.joblib", compress=3)
    (out_dir / "feature_order.json").write_text(json.dumps(feature_order, indent=2))
    (out_dir / "model_metadata.json").write_text(json.dumps(meta, indent=2))
    (out_dir / "label_version.json").write_text(json.dumps({
        "target_column": meta["target_column"],
        "task_type": "classification",
        "n_classes": 3,
        "class_names": {"0": "SELL", "1": "HOLD", "2": "BUY"},
    }, indent=2))
    # Minimal selected_features.json so PipelineManager doesn't fail
    (out_dir / "selected_features.json").write_text(json.dumps({
        "n_features": len(feature_order),
        "feature_names": feature_order,
        "schema_version": "1.0",
    }, indent=2))
    print(f"    Bundle written → {out_dir}")


def main():
    print("=" * 65)
    print("  MFIP — Lookahead Model Training")
    print(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 65)

    # ── Load config from best model ──────────────────────────────────
    best_meta    = json.loads(BEST_META_F.read_text())
    feature_order = json.loads(FEATURE_ORDER_F.read_text())  # 247 names in order
    model_params  = best_meta["model_params"]

    print(f"\nBase model  : {best_meta['model_name'].upper()}  window={best_meta['window_number']}")
    print(f"Features    : {len(feature_order)}")
    print(f"n_estimators: {model_params['n_estimators']}  max_depth={model_params['max_depth']}")
    print(f"lr          : {model_params['learning_rate']:.4f}")

    # ── Load feature dataset ─────────────────────────────────────────
    print(f"\nLoading feature dataset: {FEATURE_DATASET}")
    df = pd.read_parquet(FEATURE_DATASET)
    print(f"  Shape: {df.shape[0]:,} rows × {df.shape[1]} cols")

    # Parse timestamp column → DatetimeIndex for windowing
    ts = pd.to_datetime(df["timestamp"])
    df.index = ts

    close = df["close"].copy()
    print(f"  Close range: {close.index[0]} → {close.index[-1]}")

    # ── Load existing preprocessor (already fitted on same features) ─
    preproc = joblib.load(BEST_PREPROC_F)
    print(f"  Preprocessor: {type(preproc).__name__}")

    # ── Build imputed feature matrix ─────────────────────────────────
    # Validate all feature columns are present
    missing = [f for f in feature_order if f not in df.columns]
    if missing:
        raise ValueError(f"Missing {len(missing)} feature columns: {missing[:5]}...")

    X_raw = df[feature_order].copy()
    X_imp = preproc.transform(X_raw)
    X = pd.DataFrame(X_imp, index=df.index, columns=feature_order)
    print(f"  Feature matrix imputed: {X.shape}")

    # ── Train one model per horizon ───────────────────────────────────
    results = {}

    for horizon, model_name in [(4, "lookahead_4b"), (8, "lookahead_8b")]:
        mins = horizon * 15
        print(f"\n{'─'*65}")
        print(f"  Training {model_name}  ({horizon} bars = {mins} minutes lookahead)")
        print(f"{'─'*65}")

        # Compute label
        y_all = compute_direction(close, horizon)
        valid  = y_all.notna()
        X_v    = X[valid]
        y_v    = y_all[valid].astype(int)
        idx_v  = X_v.index

        # Walk-forward split (identical to window_000)
        train_m = (idx_v >= TRAIN_START) & (idx_v < TRAIN_END)
        val_m   = (idx_v >= TRAIN_END)   & (idx_v < VAL_END)
        test_m  = (idx_v >= VAL_END)     & (idx_v < TEST_END)

        X_train, y_train = X_v[train_m], y_v[train_m]
        X_val,   y_val   = X_v[val_m],   y_v[val_m]
        X_test,  y_test  = X_v[test_m],  y_v[test_m]

        print(f"  Rows  — train: {len(X_train):,}  val: {len(X_val):,}  test: {len(X_test):,}")

        # Label distribution (train)
        for cls, lbl in [(0,"SELL"),(1,"HOLD"),(2,"BUY")]:
            n = (y_train == cls).sum()
            print(f"    {lbl}: {n:,}  ({n/len(y_train)*100:.1f}%)")

        # Train XGBoost
        model = xgb.XGBClassifier(
            n_estimators      = model_params["n_estimators"],
            max_depth         = model_params["max_depth"],
            learning_rate     = model_params["learning_rate"],
            subsample         = model_params["subsample"],
            colsample_bytree  = model_params["colsample_bytree"],
            min_child_weight  = model_params["min_child_weight"],
            gamma             = model_params["gamma"],
            reg_alpha         = model_params["reg_alpha"],
            reg_lambda        = model_params["reg_lambda"],
            objective         = "multi:softprob",
            num_class         = 3,
            eval_metric       = "mlogloss",
            random_state      = 42,
            n_jobs            = -1,
            verbosity         = 0,
        )
        model.fit(
            X_train, y_train,
            eval_set          = [(X_val, y_val)],
            verbose           = 50,
        )

        # Evaluate
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1  = f1_score(y_test, y_pred, average="weighted")
        print(f"\n  Test accuracy : {acc:.4f}  ({acc*100:.1f}%)")
        print(f"  Test F1 (w)  : {f1:.4f}")

        # Per-class TP accuracy (directional trades only)
        for cls, lbl in [(0,"SELL"),(2,"BUY")]:
            mask = y_test == cls
            if mask.sum() > 0:
                tp_acc = accuracy_score(y_test[mask], y_pred[mask])
                print(f"  {lbl} accuracy  : {tp_acc:.4f}  ({tp_acc*100:.1f}%)  n={mask.sum()}")

        # Build metadata
        meta = {
            **best_meta,
            "target_column"    : f"direction_{horizon}b",
            "lookahead_bars"   : horizon,
            "lookahead_minutes": mins,
            "n_train_samples"  : int(len(X_train)),
            "n_val_samples"    : int(len(X_val)),
            "n_test_samples"   : int(len(X_test)),
            "training_window"  : {
                "start" : TRAIN_START.isoformat(),
                "end"   : TRAIN_END.isoformat(),
            },
            "validation_window": {"start": TRAIN_END.isoformat(), "end": VAL_END.isoformat()},
            "test_window"      : {"start": VAL_END.isoformat(),   "end": TEST_END.isoformat()},
            "test_accuracy"    : float(acc),
            "test_f1_weighted" : float(f1),
            "trained_at"       : datetime.now(timezone.utc).isoformat(),
            "base_model"       : "models/best_model",
            "neutral_threshold": NEUTRAL_THRESHOLD,
        }

        out_dir = ROOT / "models" / model_name
        save_bundle(model, preproc, feature_order, meta, out_dir)
        results[model_name] = {"acc": acc, "f1": f1}

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("  TRAINING COMPLETE")
    print(f"{'='*65}")
    print(f"  {'Model':<20} {'Test Acc':>10} {'F1 (w)':>10}")
    print(f"  {'─'*20} {'─'*10} {'─'*10}")
    print(f"  {'direction_1b (base)':<20} {73.8:>10.1f}%  {0.527:>9.3f}")
    for name, r in results.items():
        print(f"  {name:<20} {r['acc']*100:>10.1f}%  {r['f1']:>9.3f}")
    print()
    print("  Next step: integrate into InferenceEngine for combined conviction.")


if __name__ == "__main__":
    main()
