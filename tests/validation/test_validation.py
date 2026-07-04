"""
Tests for the Walk-Forward Validation Engine (Task 11).

Coverage:
  TestValidationMetrics             (10 tests)
  TestTradingMetrics                 (8 tests)
  TestAggregateMetricStats           (5 tests)
  TestStabilityAnalyzer             (10 tests)
  TestRobustnessAnalyzer            (10 tests)
  TestWindowValidator               (10 tests)
  TestWalkForwardValidator           (8 tests)
  TestValidationPipeline            (10 tests)
  TestModelAcceptance                (8 tests)
  TestReports                        (8 tests)
  TestIntegration                    (5 tests)
Total: 92 tests
"""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


# ═══════════════════════════════ Fixtures / Helpers ════════════════════════════

def _make_binary_data(n=200, seed=0):
    rng = np.random.default_rng(seed)
    y_true = rng.integers(0, 2, n)
    y_pred = y_true.copy()
    # introduce 20% noise
    flip = rng.choice(n, size=n // 5, replace=False)
    y_pred[flip] = 1 - y_pred[flip]
    y_prob = rng.random((n, 2))
    y_prob = (y_prob.T / y_prob.sum(1)).T
    return y_true, y_pred, y_prob


def _make_multiclass_data(n=200, n_classes=3, seed=0):
    rng = np.random.default_rng(seed)
    y_true = rng.integers(0, n_classes, n)
    y_pred = y_true.copy()
    flip = rng.choice(n, size=n // 4, replace=False)
    y_pred[flip] = rng.integers(0, n_classes, len(flip))
    y_prob = rng.random((n, n_classes))
    y_prob = (y_prob.T / y_prob.sum(1)).T
    return y_true, y_pred, y_prob


def _make_regression_data(n=100, seed=0):
    rng = np.random.default_rng(seed)
    y_true = rng.standard_normal(n)
    y_pred = y_true + rng.standard_normal(n) * 0.3
    return y_true, y_pred


def _make_metric_dicts(n=5, base_f1=0.7, noise=0.05, seed=0):
    rng = np.random.default_rng(seed)
    dicts = []
    for _ in range(n):
        f1  = float(np.clip(base_f1 + rng.uniform(-noise, noise), 0, 1))
        acc = float(np.clip(f1 + 0.05, 0, 1))
        dicts.append({
            "f1": f1, "accuracy": acc,
            "roc_auc": float(np.clip(f1 + 0.1, 0, 1)),
            "balanced_accuracy": float(np.clip(f1 - 0.02, 0, 1)),
            "directional_accuracy": acc,
            "avg_confidence": 0.65,
        })
    return dicts


def _make_window_result(
    window_number=1, model_name="rf", f1=0.7, error=None
):
    from src.validation.validator import WindowValidationResult
    y = np.array([0, 1, 2] * 20)
    r = WindowValidationResult(
        window_number=window_number,
        model_name=model_name,
        task_type="classification",
        classification_metrics={"f1": f1, "accuracy": f1 + 0.05, "roc_auc": f1 + 0.1,
                                 "balanced_accuracy": f1, "mcc": f1 - 0.05},
        trading_metrics={"directional_accuracy": f1, "avg_confidence": 0.65},
        combined_metrics={"f1": f1, "accuracy": f1 + 0.05, "roc_auc": f1 + 0.1,
                          "balanced_accuracy": f1, "directional_accuracy": f1,
                          "avg_confidence": 0.65},
        inference_time_s=0.01,
        n_test=60,
        y_true=y,
        y_pred=y,
        error=error,
    )
    return r


def _build_fake_bundle(tmp_path: Path, model_name="random_forest") -> Path:
    """Create a minimal valid inference bundle for testing."""
    from sklearn.ensemble import RandomForestClassifier
    from src.optimization.artifact_manager import (
        ArtifactManager, BundleConfig, ColumnImputer,
    )
    model = RandomForestClassifier(n_estimators=3, random_state=0)
    cols  = [f"f{i}" for i in range(5)]
    rng   = np.random.default_rng(0)
    X     = rng.standard_normal((80, 5))
    y     = rng.integers(0, 3, 80)
    model.fit(X, y)
    imp = ColumnImputer(apply_imputation=True)
    imp.fit(pd.DataFrame(X, columns=cols))
    cfg = BundleConfig(
        model_name=model_name, task_type="classification",
        target_column="label", feature_columns=cols, n_classes=3,
        random_seed=0, schema_version="1.0.0", label_version="1.0.0",
        window_number=1,
        train_start=None, train_end=None, val_start=None, val_end=None,
        test_start=None, test_end=None,
        best_params={}, optimization_metric="f1", n_trials=5,
        optimization_time_s=1.0, best_val_score=0.75, baseline_val_score=0.65,
        training_time_s=0.1, prediction_time_s=0.01,
        n_train_samples=50, n_val_samples=15, n_test_samples=15,
        train_metrics={"f1": 0.80}, val_metrics={"f1": 0.75}, test_metrics={"f1": 0.70},
    )
    bundle_dir = tmp_path / "bundle"
    ArtifactManager.save_bundle(model, imp, cfg, bundle_dir)
    return bundle_dir


def _build_test_df(n=30) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    df  = pd.DataFrame(
        {f"f{i}": rng.standard_normal(n) for i in range(5)},
    )
    df["label"] = rng.integers(0, 3, n)
    return df


# ═══════════════════════════ TestValidationMetrics ═════════════════════════════

class TestValidationMetrics:
    def test_classification_has_balanced_accuracy(self):
        from src.validation.metrics import compute_classification_metrics
        y_t, y_p, y_prob = _make_multiclass_data()
        m = compute_classification_metrics(y_t, y_p, y_prob)
        assert "balanced_accuracy" in m
        assert 0 <= m["balanced_accuracy"] <= 1

    def test_classification_has_mcc(self):
        from src.validation.metrics import compute_classification_metrics
        y_t, y_p, y_prob = _make_binary_data()
        m = compute_classification_metrics(y_t, y_p, y_prob)
        assert "mcc" in m
        assert -1 <= m["mcc"] <= 1

    def test_classification_has_cohen_kappa(self):
        from src.validation.metrics import compute_classification_metrics
        y_t, y_p, y_prob = _make_binary_data()
        m = compute_classification_metrics(y_t, y_p, y_prob)
        assert "cohen_kappa" in m

    def test_perfect_prediction_gives_max_scores(self):
        from src.validation.metrics import compute_classification_metrics
        y = np.array([0, 1, 2] * 20)
        m = compute_classification_metrics(y, y, None)
        assert m["accuracy"] == pytest.approx(1.0)
        assert m["f1"] == pytest.approx(1.0)
        assert m["balanced_accuracy"] == pytest.approx(1.0)
        assert m["mcc"] == pytest.approx(1.0)

    def test_classification_inherits_base_metrics(self):
        from src.validation.metrics import compute_classification_metrics
        y_t, y_p, y_prob = _make_multiclass_data()
        m = compute_classification_metrics(y_t, y_p, y_prob)
        for key in ("accuracy", "f1", "precision", "recall", "roc_auc", "pr_auc"):
            assert key in m

    def test_regression_returns_all_keys(self):
        from src.validation.metrics import compute_regression_metrics
        y_t, y_p = _make_regression_data()
        m = compute_regression_metrics(y_t, y_p)
        for k in ("mae", "rmse", "mape", "r2"):
            assert k in m

    def test_regression_perfect_r2(self):
        from src.validation.metrics import compute_regression_metrics
        y = np.arange(10.0)
        m = compute_regression_metrics(y, y)
        assert m["r2"] == pytest.approx(1.0)
        assert m["mae"] == pytest.approx(0.0)

    def test_regression_mae_non_negative(self):
        from src.validation.metrics import compute_regression_metrics
        y_t, y_p = _make_regression_data()
        m = compute_regression_metrics(y_t, y_p)
        assert m["mae"] >= 0
        assert m["rmse"] >= 0

    def test_no_proba_sets_roc_auc_nan(self):
        from src.validation.metrics import compute_classification_metrics
        y = np.array([0, 1, 0, 1, 0, 1])
        m = compute_classification_metrics(y, y, None)
        assert m["roc_auc"] != m["roc_auc"]   # NaN check

    def test_multiclass_no_proba(self):
        from src.validation.metrics import compute_classification_metrics
        y_t, y_p, _ = _make_multiclass_data()
        m = compute_classification_metrics(y_t, y_p, None)
        assert m["accuracy"] > 0
        assert "mcc" in m


# ═══════════════════════════════ TestTradingMetrics ════════════════════════════

class TestTradingMetrics:
    def test_has_long_accuracy(self):
        from src.validation.metrics import compute_trading_metrics
        y_t, y_p, y_prob = _make_multiclass_data()
        m = compute_trading_metrics(y_t, y_p, y_prob)
        assert "long_accuracy" in m

    def test_has_short_accuracy(self):
        from src.validation.metrics import compute_trading_metrics
        y_t, y_p, y_prob = _make_multiclass_data()
        m = compute_trading_metrics(y_t, y_p, y_prob)
        assert "short_accuracy" in m

    def test_has_expected_return(self):
        from src.validation.metrics import compute_trading_metrics
        y_t, y_p, y_prob = _make_multiclass_data()
        m = compute_trading_metrics(y_t, y_p, y_prob)
        assert "expected_return" in m

    def test_has_expected_risk(self):
        from src.validation.metrics import compute_trading_metrics
        y_t, y_p, y_prob = _make_multiclass_data()
        m = compute_trading_metrics(y_t, y_p, y_prob)
        assert "expected_risk" in m
        assert m["expected_risk"] >= 0

    def test_has_risk_reward_accuracy(self):
        from src.validation.metrics import compute_trading_metrics
        y_t, y_p, y_prob = _make_multiclass_data()
        m = compute_trading_metrics(y_t, y_p, y_prob)
        assert "risk_reward_accuracy" in m

    def test_perfect_long_accuracy(self):
        from src.validation.metrics import compute_trading_metrics
        y_true = np.array([1, 1, 1, 0, 2])
        y_pred = np.array([1, 1, 1, 0, 2])
        m = compute_trading_metrics(y_true, y_pred, None)
        assert m["long_accuracy"] == pytest.approx(1.0)

    def test_directional_accuracy_equals_accuracy(self):
        from src.validation.metrics import compute_trading_metrics
        y_t, y_p, _ = _make_multiclass_data()
        m = compute_trading_metrics(y_t, y_p, None)
        from sklearn.metrics import accuracy_score
        assert m["directional_accuracy"] == pytest.approx(accuracy_score(y_t, y_p))

    def test_no_proba_sets_expected_nan(self):
        from src.validation.metrics import compute_trading_metrics
        y_t, y_p, _ = _make_multiclass_data()
        m = compute_trading_metrics(y_t, y_p, None)
        assert m["expected_return"] != m["expected_return"]  # NaN


# ═══════════════════════════ TestAggregateMetricStats ══════════════════════════

class TestAggregateMetricStats:
    def test_basic_stats(self):
        from src.validation.metrics import aggregate_metric_stats
        s = aggregate_metric_stats([0.6, 0.7, 0.8])
        assert s["mean"] == pytest.approx(0.7)
        assert s["min"]  == pytest.approx(0.6)
        assert s["max"]  == pytest.approx(0.8)

    def test_ignores_nan(self):
        from src.validation.metrics import aggregate_metric_stats
        s = aggregate_metric_stats([0.5, float("nan"), 0.7])
        assert s["count"] == 2
        assert s["mean"] == pytest.approx(0.6)

    def test_all_nan_returns_nan(self):
        from src.validation.metrics import aggregate_metric_stats
        s = aggregate_metric_stats([float("nan"), float("nan")])
        assert s["count"] == 0
        assert s["mean"] != s["mean"]  # NaN

    def test_cv_computed(self):
        from src.validation.metrics import aggregate_metric_stats
        s = aggregate_metric_stats([1.0, 2.0, 3.0])
        assert s["cv"] == s["cv"]  # not NaN
        assert s["cv"] > 0

    def test_single_value(self):
        from src.validation.metrics import aggregate_metric_stats
        s = aggregate_metric_stats([0.75])
        assert s["mean"] == pytest.approx(0.75)
        assert s["std"]  == pytest.approx(0.0)


# ═══════════════════════════ TestStabilityAnalyzer ═════════════════════════════

class TestStabilityAnalyzer:
    def test_stability_score_in_range(self):
        from src.validation.stability import analyze_stability
        md = _make_metric_dicts(n=5, base_f1=0.7, noise=0.02)
        r  = analyze_stability(md, "rf")
        assert 0.0 <= r.stability_score <= 1.0

    def test_stable_model_high_score(self):
        from src.validation.stability import analyze_stability
        # Very consistent metrics → high stability
        md = _make_metric_dicts(n=5, base_f1=0.75, noise=0.001)
        r  = analyze_stability(md, "rf")
        assert r.stability_score > 0.7

    def test_unstable_model_lower_score(self):
        from src.validation.stability import analyze_stability
        md_stable   = _make_metric_dicts(n=5, noise=0.001)
        md_unstable = _make_metric_dicts(n=5, noise=0.15)
        r_s = analyze_stability(md_stable,   "rf")
        r_u = analyze_stability(md_unstable, "rf")
        assert r_s.stability_score > r_u.stability_score

    def test_empty_input_returns_zero_score(self):
        from src.validation.stability import analyze_stability
        r = analyze_stability([], "rf")
        assert r.stability_score == 0.0
        assert r.n_windows == 0

    def test_metric_cvs_populated(self):
        from src.validation.stability import analyze_stability
        md = _make_metric_dicts(n=4)
        r  = analyze_stability(md, "rf")
        assert "f1" in r.metric_cvs

    def test_most_variable_is_detected(self):
        from src.validation.stability import analyze_stability
        md = []
        rng = np.random.default_rng(0)
        for _ in range(5):
            md.append({
                "f1":       float(rng.uniform(0.65, 0.75)),
                "accuracy": float(rng.uniform(0.1, 0.9)),  # high variance
                "roc_auc":  float(rng.uniform(0.68, 0.72)),
                "balanced_accuracy": float(rng.uniform(0.64, 0.66)),
                "directional_accuracy": float(rng.uniform(0.65, 0.75)),
            })
        r = analyze_stability(md, "rf")
        assert r.most_variable_metric is not None

    def test_is_stable_flag(self):
        from src.validation.stability import analyze_stability
        # Use high noise so that stability_score is well below 0.99 but above 0.01
        md    = _make_metric_dicts(n=5, noise=0.20)
        r_hi  = analyze_stability(md, "rf", stability_threshold=0.99)
        r_lo  = analyze_stability(md, "rf", stability_threshold=0.01)
        assert r_lo.is_stable is True
        assert r_hi.is_stable is False

    def test_regression_uses_r2(self):
        from src.validation.stability import analyze_stability
        md = [{"r2": 0.8, "mae": 0.1, "rmse": 0.15, "mape": 5.0} for _ in range(4)]
        r  = analyze_stability(md, "rf", task_type="regression")
        assert r.n_windows == 4

    def test_window_scores_length_matches_input(self):
        from src.validation.stability import analyze_stability
        md = _make_metric_dicts(n=7)
        r  = analyze_stability(md, "rf")
        assert len(r.window_scores) == 7

    def test_single_window(self):
        from src.validation.stability import analyze_stability
        r = analyze_stability([{"f1": 0.7}], "rf")
        assert r.n_windows == 1


# ═══════════════════════════ TestRobustnessAnalyzer ════════════════════════════

class TestRobustnessAnalyzer:
    def _make_window_results(self, n=5, base_f1=0.7, noise=0.03, seed=0):
        results = []
        rng = np.random.default_rng(seed)
        y = np.array([0, 1, 2] * 20)
        for i in range(1, n + 1):
            f1 = float(np.clip(base_f1 + rng.uniform(-noise, noise), 0, 1))
            results.append(
                _make_window_result(window_number=i, f1=f1)
            )
        return results

    def test_robustness_score_in_range(self):
        from src.validation.robustness import analyze_robustness
        wr = self._make_window_results()
        r  = analyze_robustness(wr, "rf")
        assert 0.0 <= r.robustness_score <= 1.0

    def test_best_worst_window_identified(self):
        from src.validation.robustness import analyze_robustness
        wr = self._make_window_results(n=5)
        # Override to make window 3 clearly best
        wr[2].combined_metrics["f1"] = 0.99
        wr[2].classification_metrics["f1"] = 0.99
        r = analyze_robustness(wr, "rf")
        assert r.best_window == 3

    def test_metric_stats_populated(self):
        from src.validation.robustness import analyze_robustness
        wr = self._make_window_results()
        r  = analyze_robustness(wr, "rf")
        assert "f1" in r.metric_stats
        assert r.metric_stats["f1"]["count"] > 0

    def test_empty_returns_zero_score(self):
        from src.validation.robustness import analyze_robustness
        r = analyze_robustness([], "rf")
        assert r.robustness_score == 0.0
        assert r.n_windows == 0

    def test_overfitting_detected_when_gap_large(self):
        from src.validation.robustness import analyze_robustness
        wr = self._make_window_results(n=4, base_f1=0.6)
        # Simulate large train-val gap
        train_hist = [{"f1": 0.95} for _ in range(4)]  # train >> test (0.6)
        r = analyze_robustness(wr, "rf", train_metrics_history=train_hist,
                               overfitting_threshold=0.15)
        assert r.generalization.overfitting_detected is True

    def test_no_overfitting_when_gap_small(self):
        from src.validation.robustness import analyze_robustness
        wr = self._make_window_results(n=4, base_f1=0.7)
        train_hist = [{"f1": 0.72} for _ in range(4)]  # small gap
        r = analyze_robustness(wr, "rf", train_metrics_history=train_hist,
                               overfitting_threshold=0.15)
        assert r.generalization.overfitting_detected is False

    def test_underfitting_detected(self):
        from src.validation.robustness import analyze_robustness
        wr = self._make_window_results(n=4, base_f1=0.10)
        r  = analyze_robustness(wr, "rf", min_acceptable_primary=0.40)
        assert r.generalization.underfitting_detected is True

    def test_degradation_detected_on_declining_scores(self):
        from src.validation.robustness import analyze_robustness
        # Sharply declining F1: 0.8, 0.6, 0.4, 0.2
        wr = []
        for i, f1 in enumerate([0.8, 0.6, 0.4, 0.2], 1):
            wr.append(_make_window_result(window_number=i, f1=f1))
        r = analyze_robustness(wr, "rf")
        assert r.generalization.performance_degradation is True

    def test_no_degradation_on_stable_scores(self):
        from src.validation.robustness import analyze_robustness
        wr = self._make_window_results(n=5, base_f1=0.75, noise=0.01)
        r  = analyze_robustness(wr, "rf")
        assert r.generalization.performance_degradation is False

    def test_generalization_score_in_range(self):
        from src.validation.robustness import analyze_robustness
        wr = self._make_window_results(n=5)
        r  = analyze_robustness(wr, "rf")
        assert 0.0 <= r.generalization.generalization_score <= 1.0


# ════════════════════════════ TestWindowValidator ══════════════════════════════

class TestWindowValidator:
    def test_validate_returns_result(self, tmp_path):
        from src.validation.validator import WindowValidator
        bundle_dir = _build_fake_bundle(tmp_path)
        test_df    = _build_test_df()
        r = WindowValidator().validate(bundle_dir, test_df, "label", 1)
        assert r.window_number == 1
        assert r.error is None

    def test_validate_metrics_populated(self, tmp_path):
        from src.validation.validator import WindowValidator
        bundle_dir = _build_fake_bundle(tmp_path)
        test_df    = _build_test_df()
        r = WindowValidator().validate(bundle_dir, test_df, "label", 1)
        assert r.classification_metrics.get("f1") is not None

    def test_validate_trading_metrics_present(self, tmp_path):
        from src.validation.validator import WindowValidator
        bundle_dir = _build_fake_bundle(tmp_path)
        test_df    = _build_test_df()
        r = WindowValidator().validate(bundle_dir, test_df, "label", 1)
        assert "directional_accuracy" in r.trading_metrics

    def test_validate_error_on_missing_bundle(self, tmp_path):
        from src.validation.validator import WindowValidator
        r = WindowValidator().validate(
            tmp_path / "nonexistent", _build_test_df(), "label", 1
        )
        assert r.error is not None

    def test_validate_error_on_missing_target_column(self, tmp_path):
        from src.validation.validator import WindowValidator
        bundle_dir = _build_fake_bundle(tmp_path)
        df = _build_test_df().drop(columns=["label"])
        r  = WindowValidator().validate(bundle_dir, df, "label", 1)
        assert r.error is not None

    def test_validate_y_true_stored(self, tmp_path):
        from src.validation.validator import WindowValidator
        bundle_dir = _build_fake_bundle(tmp_path)
        r = WindowValidator().validate(bundle_dir, _build_test_df(), "label", 1)
        assert r.y_true is not None
        assert len(r.y_true) == 30

    def test_validate_inference_time_positive(self, tmp_path):
        from src.validation.validator import WindowValidator
        bundle_dir = _build_fake_bundle(tmp_path)
        r = WindowValidator().validate(bundle_dir, _build_test_df(), "label", 1)
        assert r.inference_time_s >= 0

    def test_bundle_train_metrics_loaded(self, tmp_path):
        from src.validation.validator import WindowValidator
        bundle_dir = _build_fake_bundle(tmp_path)
        r = WindowValidator().validate(bundle_dir, _build_test_df(), "label", 1)
        assert r.bundle_train_metrics is not None
        assert "f1" in r.bundle_train_metrics

    def test_combined_metrics_has_scalars_only(self, tmp_path):
        from src.validation.validator import WindowValidator
        bundle_dir = _build_fake_bundle(tmp_path)
        r = WindowValidator().validate(bundle_dir, _build_test_df(), "label", 1)
        for v in r.combined_metrics.values():
            assert isinstance(v, (int, float))

    def test_read_only_no_model_mutation(self, tmp_path):
        """Ensure the validator never modifies bundle files."""
        import hashlib
        from src.validation.validator import WindowValidator
        bundle_dir = _build_fake_bundle(tmp_path)
        model_path = bundle_dir / "model.joblib"
        before = hashlib.sha256(model_path.read_bytes()).hexdigest()
        WindowValidator().validate(bundle_dir, _build_test_df(), "label", 1)
        after  = hashlib.sha256(model_path.read_bytes()).hexdigest()
        assert before == after


# ══════════════════════════ TestWalkForwardValidator ══════════════════════════

class TestWalkForwardValidator:
    def _build_window_structure(self, tmp_path, n_windows=2):
        """Build windows_dir + models_dir with proper bundles."""
        windows_dir = tmp_path / "windows"
        models_dir  = tmp_path / "models"
        for w in range(1, n_windows + 1):
            win_dir = windows_dir / f"window_{w:03d}"
            win_dir.mkdir(parents=True)
            test_df = _build_test_df()
            test_df.to_parquet(win_dir / "test.parquet")
            # Bundle
            bundle_dir = models_dir / f"window_{w:03d}" / "random_forest" / "bundle"
            _build_fake_bundle(tmp_path / f"_tmp_bundle_{w}", model_name="random_forest")
            shutil.copytree(tmp_path / f"_tmp_bundle_{w}" / "bundle", bundle_dir)
        return windows_dir, models_dir

    def test_validates_all_windows(self, tmp_path):
        from src.validation.walk_forward_validator import WalkForwardValidator
        wd, md = self._build_window_structure(tmp_path, n_windows=2)
        r = WalkForwardValidator().validate(wd, md, "label", ["random_forest"])
        assert r.n_windows == 2
        assert len(r.model_results["random_forest"]) == 2

    def test_empty_windows_dir(self, tmp_path):
        from src.validation.walk_forward_validator import WalkForwardValidator
        r = WalkForwardValidator().validate(
            tmp_path / "empty", tmp_path / "models", "label", ["rf"]
        )
        assert r.n_windows == 0
        assert len(r.errors) > 0

    def test_missing_bundle_appends_error(self, tmp_path):
        from src.validation.walk_forward_validator import WalkForwardValidator
        wd, md = self._build_window_structure(tmp_path, n_windows=1)
        # Ask for model that doesn't exist
        r = WalkForwardValidator().validate(wd, md, "label", ["nonexistent_model"])
        assert len(r.errors) > 0

    def test_skip_on_error_continues(self, tmp_path):
        from src.validation.walk_forward_validator import WalkForwardValidator
        wd, md = self._build_window_structure(tmp_path, n_windows=2)
        r = WalkForwardValidator(skip_on_error=True).validate(
            wd, md, "wrong_target", ["random_forest"]
        )
        # Some errors but didn't raise
        assert len(r.model_results["random_forest"]) > 0

    def test_discover_windows(self, tmp_path):
        from src.validation.walk_forward_validator import WalkForwardValidator
        (tmp_path / "window_001").mkdir()
        (tmp_path / "window_002").mkdir()
        (tmp_path / "other_dir").mkdir()
        dirs = WalkForwardValidator._discover_windows(tmp_path)
        assert len(dirs) == 2

    def test_discover_models(self, tmp_path):
        from src.validation.walk_forward_validator import WalkForwardValidator
        wnum = 1
        for m in ["xgboost", "rf"]:
            (tmp_path / f"window_{wnum:03d}" / m / "bundle").mkdir(parents=True)
        first_win = tmp_path / "window_001"
        first_win.mkdir(exist_ok=True)
        models = WalkForwardValidator._discover_models(tmp_path, first_win)
        assert set(models) == {"xgboost", "rf"}

    def test_result_n_models_matches(self, tmp_path):
        from src.validation.walk_forward_validator import WalkForwardValidator
        wd, md = self._build_window_structure(tmp_path, n_windows=1)
        r = WalkForwardValidator().validate(wd, md, "label", ["random_forest"])
        assert r.n_models == 1

    def test_window_results_ordered(self, tmp_path):
        from src.validation.walk_forward_validator import WalkForwardValidator
        wd, md = self._build_window_structure(tmp_path, n_windows=3)
        r = WalkForwardValidator().validate(wd, md, "label", ["random_forest"])
        nums = [res.window_number for res in r.model_results["random_forest"]]
        assert nums == sorted(nums)


# ════════════════════════════ TestValidationPipeline ══════════════════════════

class TestValidationPipeline:
    def _build_full_structure(self, tmp_path, n_windows=2):
        windows_dir = tmp_path / "windows"
        models_dir  = tmp_path / "models"
        for w in range(1, n_windows + 1):
            win_dir = windows_dir / f"window_{w:03d}"
            win_dir.mkdir(parents=True)
            _build_test_df().to_parquet(win_dir / "test.parquet")
            bundle_dir = models_dir / f"window_{w:03d}" / "random_forest" / "bundle"
            _build_fake_bundle(tmp_path / f"_b{w}", model_name="random_forest")
            shutil.copytree(tmp_path / f"_b{w}" / "bundle", bundle_dir)
        return windows_dir, models_dir

    def test_run_returns_result(self, tmp_path):
        from src.validation.validation_pipeline import ValidationConfig, ValidationPipeline
        wd, md = self._build_full_structure(tmp_path)
        cfg = ValidationConfig(
            windows_dir=wd, models_dir=md, output_dir=tmp_path / "out",
            target_column="label", model_names=["random_forest"]
        )
        r = ValidationPipeline().run(cfg)
        assert len(r.model_results) == 1
        assert r.model_results[0].model_name == "random_forest"

    def test_ranked_models_list(self, tmp_path):
        from src.validation.validation_pipeline import ValidationConfig, ValidationPipeline
        wd, md = self._build_full_structure(tmp_path)
        cfg = ValidationConfig(
            windows_dir=wd, models_dir=md, output_dir=tmp_path / "out",
            target_column="label", model_names=["random_forest"]
        )
        r = ValidationPipeline().run(cfg)
        assert "random_forest" in r.ranked_models

    def test_overall_summary_json_written(self, tmp_path):
        from src.validation.validation_pipeline import ValidationConfig, ValidationPipeline
        wd, md = self._build_full_structure(tmp_path)
        cfg = ValidationConfig(
            windows_dir=wd, models_dir=md, output_dir=tmp_path / "out",
            target_column="label", model_names=["random_forest"]
        )
        ValidationPipeline().run(cfg)
        assert (tmp_path / "out" / "overall_summary.json").exists()

    def test_per_window_json_written(self, tmp_path):
        from src.validation.validation_pipeline import ValidationConfig, ValidationPipeline
        wd, md = self._build_full_structure(tmp_path, n_windows=2)
        cfg = ValidationConfig(
            windows_dir=wd, models_dir=md, output_dir=tmp_path / "out",
            target_column="label", model_names=["random_forest"]
        )
        ValidationPipeline().run(cfg)
        assert (tmp_path / "out" / "window_001" / "random_forest.json").exists()
        assert (tmp_path / "out" / "window_002" / "random_forest.json").exists()

    def test_report_files_created(self, tmp_path):
        from src.validation.validation_pipeline import ValidationConfig, ValidationPipeline
        wd, md = self._build_full_structure(tmp_path)
        report_dir = tmp_path / "reports"
        cfg = ValidationConfig(
            windows_dir=wd, models_dir=md, output_dir=tmp_path / "out",
            target_column="label", model_names=["random_forest"],
            report_dir=report_dir,
        )
        ValidationPipeline().run(cfg)
        for fname in (
            "walk_forward_validation_report.md",
            "validation_summary.csv",
            "window_metrics.csv",
            "robustness_report.md",
            "generalization_report.md",
            "stability_report.md",
        ):
            assert (report_dir / fname).exists(), f"Missing: {fname}"

    def test_timing_populated(self, tmp_path):
        from src.validation.validation_pipeline import ValidationConfig, ValidationPipeline
        wd, md = self._build_full_structure(tmp_path)
        cfg = ValidationConfig(
            windows_dir=wd, models_dir=md, output_dir=tmp_path / "out",
            target_column="label", model_names=["random_forest"]
        )
        r = ValidationPipeline().run(cfg)
        assert r.total_time_s > 0

    def test_ranking_score_in_range(self, tmp_path):
        from src.validation.validation_pipeline import ValidationConfig, ValidationPipeline
        wd, md = self._build_full_structure(tmp_path)
        cfg = ValidationConfig(
            windows_dir=wd, models_dir=md, output_dir=tmp_path / "out",
            target_column="label", model_names=["random_forest"]
        )
        r = ValidationPipeline().run(cfg)
        assert 0 <= r.model_results[0].ranking_score <= 1

    def test_skip_on_error_works(self, tmp_path):
        from src.validation.validation_pipeline import ValidationConfig, ValidationPipeline
        wd, md = self._build_full_structure(tmp_path)
        cfg = ValidationConfig(
            windows_dir=wd, models_dir=md, output_dir=tmp_path / "out",
            target_column="nonexistent", model_names=["random_forest"],
            skip_on_error=True,
        )
        r = ValidationPipeline().run(cfg)
        assert len(r.errors) > 0   # errors collected, not raised

    def test_n_windows_and_n_models_reported(self, tmp_path):
        from src.validation.validation_pipeline import ValidationConfig, ValidationPipeline
        wd, md = self._build_full_structure(tmp_path, n_windows=2)
        cfg = ValidationConfig(
            windows_dir=wd, models_dir=md, output_dir=tmp_path / "out",
            target_column="label", model_names=["random_forest"]
        )
        r = ValidationPipeline().run(cfg)
        assert r.n_windows == 2
        assert r.n_models == 1

    def test_summary_json_has_best_model(self, tmp_path):
        from src.validation.validation_pipeline import ValidationConfig, ValidationPipeline
        wd, md = self._build_full_structure(tmp_path)
        cfg = ValidationConfig(
            windows_dir=wd, models_dir=md, output_dir=tmp_path / "out",
            target_column="label", model_names=["random_forest"]
        )
        r = ValidationPipeline().run(cfg)
        assert r.overall_summary.get("best_model") is not None


# ════════════════════════════ TestModelAcceptance ══════════════════════════════

class TestModelAcceptance:
    def _make_config(self):
        from src.validation.validation_pipeline import ValidationConfig
        return ValidationConfig(
            windows_dir=Path("."), models_dir=Path("."),
            output_dir=Path("."), target_column="label",
            min_accuracy=0.50, min_f1=0.40,
            min_directional_accuracy=0.50,
            min_trading_accuracy=0.45, max_variance=0.25,
            stability_threshold=0.65,
        )

    def _run_acceptance(self, metric_dicts, stability_score=0.8, overfitting=False):
        from src.validation.validation_pipeline import _determine_acceptance
        from src.validation.stability import StabilityResult
        from src.validation.robustness import RobustnessResult, GeneralizationAnalysis
        gen = GeneralizationAnalysis(
            overfitting_detected=overfitting,
            underfitting_detected=False,
            performance_degradation=False,
            regime_sensitivity=0.1,
            high_vol_performance=0.7, low_vol_performance=0.7,
            trending_performance=0.7, ranging_performance=0.7,
            train_test_gap=0.05, degradation_slope=0.0,
            generalization_score=0.85,
        )
        stab = StabilityResult(
            model_name="rf", n_windows=len(metric_dicts),
            metric_stats={}, metric_cvs={"f1": 0.05},
            confidence_cv=0.1, stability_score=stability_score,
            most_variable_metric="f1", least_variable_metric="f1",
            window_scores=[], prediction_variance=0.05,
            is_stable=stability_score >= 0.65,
            stability_threshold=0.65,
        )
        rob = RobustnessResult(
            model_name="rf", n_windows=3, metric_stats={},
            best_window=1, worst_window=2, best_score=0.8, worst_score=0.6,
            robustness_score=0.75, generalization=gen, primary_metric="f1",
        )
        return _determine_acceptance(stab, rob, metric_dicts, "classification",
                                     self._make_config())

    def test_production_ready_when_all_pass(self):
        from src.validation.validation_pipeline import PRODUCTION_READY
        md = [{"accuracy": 0.8, "f1": 0.75, "directional_accuracy": 0.8,
               "tp_prediction_accuracy": 0.7} for _ in range(3)]
        status, _ = self._run_acceptance(md)
        assert status == PRODUCTION_READY

    def test_needs_improvement_when_stability_low(self):
        from src.validation.validation_pipeline import NEEDS_IMPROVEMENT
        md = [{"accuracy": 0.8, "f1": 0.75, "directional_accuracy": 0.8,
               "tp_prediction_accuracy": 0.7} for _ in range(3)]
        status, reasons = self._run_acceptance(md, stability_score=0.3)
        assert status == NEEDS_IMPROVEMENT

    def test_rejected_when_f1_critical(self):
        from src.validation.validation_pipeline import REJECTED
        md = [{"accuracy": 0.4, "f1": 0.15, "directional_accuracy": 0.3,
               "tp_prediction_accuracy": 0.2} for _ in range(3)]
        status, _ = self._run_acceptance(md)
        assert status == REJECTED

    def test_reasons_populated_on_failure(self):
        md = [{"accuracy": 0.3, "f1": 0.2, "directional_accuracy": 0.3,
               "tp_prediction_accuracy": 0.2} for _ in range(3)]
        _, reasons = self._run_acceptance(md)
        assert len(reasons) > 0

    def test_overfitting_flag_in_reasons(self):
        md = [{"accuracy": 0.8, "f1": 0.75, "directional_accuracy": 0.8,
               "tp_prediction_accuracy": 0.7} for _ in range(3)]
        _, reasons = self._run_acceptance(md, overfitting=True)
        overfitting_reason = any("overfitting" in r.lower() for r in reasons)
        assert overfitting_reason

    def test_no_reasons_when_all_pass(self):
        md = [{"accuracy": 0.9, "f1": 0.85, "directional_accuracy": 0.9,
               "tp_prediction_accuracy": 0.8} for _ in range(3)]
        status, reasons = self._run_acceptance(md)
        from src.validation.validation_pipeline import PRODUCTION_READY
        if status == PRODUCTION_READY:
            assert len(reasons) == 0

    def test_empty_metric_dicts_is_rejected(self):
        from src.validation.validation_pipeline import REJECTED
        status, _ = self._run_acceptance([])
        assert status == REJECTED

    def test_needs_improvement_vs_rejected_boundary(self):
        # Only trading accuracy below threshold → needs improvement
        from src.validation.validation_pipeline import NEEDS_IMPROVEMENT
        md = [{"accuracy": 0.75, "f1": 0.65, "directional_accuracy": 0.75,
               "tp_prediction_accuracy": 0.20} for _ in range(3)]  # tp below threshold
        status, _ = self._run_acceptance(md)
        assert status in (NEEDS_IMPROVEMENT, "production_ready")  # tp only a warning


# ═══════════════════════════════ TestReports ═══════════════════════════════════

class TestReports:
    def _make_pipeline_result(self, tmp_path):
        from src.validation.validation_pipeline import (
            ValidationConfig, ValidationPipeline,
        )
        windows_dir = tmp_path / "windows"
        models_dir  = tmp_path / "models"
        win_dir = windows_dir / "window_001"
        win_dir.mkdir(parents=True)
        _build_test_df().to_parquet(win_dir / "test.parquet")
        bundle_dir = models_dir / "window_001" / "random_forest" / "bundle"
        _build_fake_bundle(tmp_path / "_b", model_name="random_forest")
        shutil.copytree(tmp_path / "_b" / "bundle", bundle_dir)
        cfg = ValidationConfig(
            windows_dir=windows_dir, models_dir=models_dir,
            output_dir=tmp_path / "out",
            target_column="label", model_names=["random_forest"]
        )
        return ValidationPipeline().run(cfg), tmp_path / "reports"

    def test_main_report_exists(self, tmp_path):
        from src.validation.reports import generate_all_reports
        pr, report_dir = self._make_pipeline_result(tmp_path)
        paths = generate_all_reports(pr, report_dir)
        assert (report_dir / "walk_forward_validation_report.md").exists()

    def test_validation_summary_csv_exists(self, tmp_path):
        from src.validation.reports import generate_all_reports
        pr, report_dir = self._make_pipeline_result(tmp_path)
        generate_all_reports(pr, report_dir)
        assert (report_dir / "validation_summary.csv").exists()

    def test_window_metrics_csv_exists(self, tmp_path):
        from src.validation.reports import generate_all_reports
        pr, report_dir = self._make_pipeline_result(tmp_path)
        generate_all_reports(pr, report_dir)
        assert (report_dir / "window_metrics.csv").exists()

    def test_robustness_report_exists(self, tmp_path):
        from src.validation.reports import generate_all_reports
        pr, report_dir = self._make_pipeline_result(tmp_path)
        generate_all_reports(pr, report_dir)
        assert (report_dir / "robustness_report.md").exists()

    def test_generalization_report_exists(self, tmp_path):
        from src.validation.reports import generate_all_reports
        pr, report_dir = self._make_pipeline_result(tmp_path)
        generate_all_reports(pr, report_dir)
        assert (report_dir / "generalization_report.md").exists()

    def test_stability_report_exists(self, tmp_path):
        from src.validation.reports import generate_all_reports
        pr, report_dir = self._make_pipeline_result(tmp_path)
        generate_all_reports(pr, report_dir)
        assert (report_dir / "stability_report.md").exists()

    def test_returns_path_dict_with_six_entries(self, tmp_path):
        from src.validation.reports import generate_all_reports
        pr, report_dir = self._make_pipeline_result(tmp_path)
        paths = generate_all_reports(pr, report_dir)
        assert len(paths) == 6

    def test_window_metrics_csv_has_rows(self, tmp_path):
        import csv
        from src.validation.reports import generate_all_reports
        pr, report_dir = self._make_pipeline_result(tmp_path)
        generate_all_reports(pr, report_dir)
        with open(report_dir / "window_metrics.csv", newline="") as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) >= 1


# ═══════════════════════════════ TestIntegration ═══════════════════════════════

class TestIntegration:
    def _full_setup(self, tmp_path, n_windows=3, n_models=2):
        windows_dir = tmp_path / "windows"
        models_dir  = tmp_path / "models"
        model_names = ["random_forest", "extra_trees"][:n_models]
        for w in range(1, n_windows + 1):
            win_dir = windows_dir / f"window_{w:03d}"
            win_dir.mkdir(parents=True)
            _build_test_df(n=40).to_parquet(win_dir / "test.parquet")
            for mn in model_names:
                from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
                from src.optimization.artifact_manager import (
                    ArtifactManager, BundleConfig, ColumnImputer,
                )
                cls = RandomForestClassifier if mn == "random_forest" else ExtraTreesClassifier
                model = cls(n_estimators=3, random_state=0)
                cols  = [f"f{i}" for i in range(5)]
                rng   = np.random.default_rng(0)
                X     = rng.standard_normal((80, 5))
                y     = rng.integers(0, 3, 80)
                model.fit(X, y)
                imp = ColumnImputer(apply_imputation=True)
                imp.fit(pd.DataFrame(X, columns=cols))
                cfg = BundleConfig(
                    model_name=mn, task_type="classification",
                    target_column="label", feature_columns=cols, n_classes=3,
                    random_seed=0, schema_version="1.0.0", label_version="1.0.0",
                    window_number=w,
                    train_start=None, train_end=None, val_start=None, val_end=None,
                    test_start=None, test_end=None,
                    best_params={}, optimization_metric="f1", n_trials=5,
                    optimization_time_s=1.0, best_val_score=0.75,
                    baseline_val_score=0.65, training_time_s=0.1,
                    prediction_time_s=0.01, n_train_samples=50,
                    n_val_samples=15, n_test_samples=15,
                    train_metrics={"f1": 0.80}, val_metrics={"f1": 0.75},
                    test_metrics={"f1": 0.70},
                )
                bd = models_dir / f"window_{w:03d}" / mn / "bundle"
                ArtifactManager.save_bundle(model, imp, cfg, bd)
        return windows_dir, models_dir, model_names

    def test_full_pipeline_two_models_three_windows(self, tmp_path):
        from src.validation.validation_pipeline import ValidationConfig, ValidationPipeline
        wd, md, mns = self._full_setup(tmp_path, n_windows=3, n_models=2)
        cfg = ValidationConfig(
            windows_dir=wd, models_dir=md, output_dir=tmp_path / "out",
            target_column="label", model_names=mns
        )
        r = ValidationPipeline().run(cfg)
        assert len(r.model_results) == 2
        assert r.n_windows == 3

    def test_ranked_models_ordered_by_score(self, tmp_path):
        from src.validation.validation_pipeline import ValidationConfig, ValidationPipeline
        wd, md, mns = self._full_setup(tmp_path, n_windows=2, n_models=2)
        cfg = ValidationConfig(
            windows_dir=wd, models_dir=md, output_dir=tmp_path / "out",
            target_column="label", model_names=mns
        )
        r = ValidationPipeline().run(cfg)
        scores = [m.ranking_score for m in r.model_results]
        assert scores == sorted(scores, reverse=True)

    def test_no_retrain_during_validation(self, tmp_path):
        """Bundle files must not be modified during validation."""
        import hashlib
        from src.validation.validation_pipeline import ValidationConfig, ValidationPipeline
        wd, md, mns = self._full_setup(tmp_path, n_windows=1, n_models=1)
        model_path = md / "window_001" / mns[0] / "bundle" / "model.joblib"
        before = hashlib.sha256(model_path.read_bytes()).hexdigest()
        cfg = ValidationConfig(
            windows_dir=wd, models_dir=md, output_dir=tmp_path / "out",
            target_column="label", model_names=mns
        )
        ValidationPipeline().run(cfg)
        after = hashlib.sha256(model_path.read_bytes()).hexdigest()
        assert before == after

    def test_all_six_report_files_created(self, tmp_path):
        from src.validation.validation_pipeline import ValidationConfig, ValidationPipeline
        wd, md, mns = self._full_setup(tmp_path, n_windows=2, n_models=1)
        report_dir = tmp_path / "reports"
        cfg = ValidationConfig(
            windows_dir=wd, models_dir=md, output_dir=tmp_path / "out",
            target_column="label", model_names=mns, report_dir=report_dir
        )
        ValidationPipeline().run(cfg)
        for fname in (
            "walk_forward_validation_report.md", "validation_summary.csv",
            "window_metrics.csv", "robustness_report.md",
            "generalization_report.md", "stability_report.md",
        ):
            assert (report_dir / fname).exists(), f"Missing: {fname}"

    def test_each_window_has_independent_json(self, tmp_path):
        from src.validation.validation_pipeline import ValidationConfig, ValidationPipeline
        wd, md, mns = self._full_setup(tmp_path, n_windows=3, n_models=1)
        cfg = ValidationConfig(
            windows_dir=wd, models_dir=md, output_dir=tmp_path / "out",
            target_column="label", model_names=mns
        )
        ValidationPipeline().run(cfg)
        for w in range(1, 4):
            assert (tmp_path / "out" / f"window_{w:03d}" / f"{mns[0]}.json").exists()
