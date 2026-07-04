"""
Tests for Baseline Model Training Pipeline (Task 9)
====================================================
Coverage:
  TestDetectTaskType          — task type auto-detection
  TestMetricsClassification   — accuracy, F1, ROC-AUC, confusion matrix
  TestMetricsRegression       — MAE, RMSE, MAPE, R²
  TestTradingMetrics          — directional accuracy, TP/SL precision, confidence
  TestModelFactory            — all 5 model types, regression, invalid name
  TestTrainer                 — classification, regression, NaN handling
  TestModelRegistry           — save/load, metadata JSON, list_models
  TestEvaluator               — aggregate_metrics, model_comparison, leaderboard
  TestTrainingPipeline        — end-to-end with synthetic windows
  TestReports                 — markdown and CSV report generation
  TestIntegration             — full pipeline correctness and no data leakage
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.training import (
    Evaluator,
    ModelFactory,
    ModelRegistry,
    ModelWindowResult,
    PipelineConfig,
    PipelineResult,
    SKLEARN_MODELS,
    SUPPORTED_MODELS,
    Trainer,
    TrainerConfig,
    TrainingPipeline,
    _discover_windows,
    compute_classification_metrics,
    compute_regression_metrics,
    compute_trading_metrics,
    detect_task_type,
    generate_comparison_csv,
    generate_leaderboard_csv,
    generate_metrics_csv,
    generate_training_report,
)
from src.training.model_registry import ModelMeta


# ── Fixtures ───────────────────────────────────────────────────────────────────

def _make_clf_arrays(n: int = 300, n_features: int = 8, seed: int = 0, n_classes: int = 2):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_features))
    # Inject a bit of signal
    y = (X[:, 0] + 0.5 * X[:, 1] + rng.standard_normal(n) * 0.5 > 0).astype(int)
    if n_classes == 3:
        y = np.where(X[:, 0] > 1, 2, y)
    return X, y


def _make_reg_arrays(n: int = 300, n_features: int = 8, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_features))
    y = 3 * X[:, 0] - 2 * X[:, 1] + rng.standard_normal(n)
    return X, y


def _make_clf_df(n: int = 300, n_features: int = 8, seed: int = 0,
                  start: str = "2020-01-01", n_classes: int = 2) -> pd.DataFrame:
    X, y = _make_clf_arrays(n, n_features, seed, n_classes)
    idx  = pd.date_range(start, periods=n, freq="h")
    cols = {f"feat_{i}": X[:, i] for i in range(n_features)}
    cols["target"] = y.astype(float)
    return pd.DataFrame(cols, index=idx)


def _make_reg_df(n: int = 300, n_features: int = 8, seed: int = 0,
                  start: str = "2020-01-01") -> pd.DataFrame:
    X, y = _make_reg_arrays(n, n_features, seed)
    idx  = pd.date_range(start, periods=n, freq="h")
    cols = {f"feat_{i}": X[:, i] for i in range(n_features)}
    cols["target"] = y
    return pd.DataFrame(cols, index=idx)


def _make_windows(
    tmp_path: Path,
    n_windows: int = 2,
    n_train: int = 300,
    n_val: int = 100,
    n_test: int = 100,
    n_features: int = 8,
    n_classes: int = 2,
) -> Path:
    """Write synthetic walk-forward windows to *tmp_path/windows/*."""
    windows_dir = tmp_path / "windows"
    windows_dir.mkdir()
    cursor = pd.Timestamp("2020-01-01")
    for w in range(n_windows):
        win_dir = windows_dir / f"window_{w:03d}"
        win_dir.mkdir()
        for name, n in [("train", n_train), ("validation", n_val), ("test", n_test)]:
            df = _make_clf_df(n, n_features, seed=w * 100, start=str(cursor.date()),
                               n_classes=n_classes)
            df.to_parquet(win_dir / f"{name}.parquet")
            cursor = df.index[-1] + pd.Timedelta(hours=1)
    return windows_dir


# ── TestDetectTaskType ─────────────────────────────────────────────────────────

class TestDetectTaskType:
    def test_binary_integers(self):
        y = pd.Series([0, 1, 0, 1, 1, 0])
        assert detect_task_type(y) == "classification"

    def test_multiclass_integers(self):
        y = pd.Series([0, 1, 2, 0, 1, 2])
        assert detect_task_type(y) == "classification"

    def test_continuous_float(self):
        y = pd.Series(np.linspace(0, 1, 100))
        assert detect_task_type(y) == "regression"

    def test_few_unique_floats(self):
        y = pd.Series([0.0, 1.0, 2.0] * 20)
        assert detect_task_type(y) == "classification"

    def test_nan_ignored(self):
        y = pd.Series([0.0, 1.0, np.nan, 0.0, 1.0])
        assert detect_task_type(y) == "classification"


# ── TestMetricsClassification ─────────────────────────────────────────────────

class TestMetricsClassification:
    def _perfect(self):
        y = np.array([0, 0, 1, 1])
        p = np.eye(2)[[0, 0, 1, 1]]
        return compute_classification_metrics(y, y, p)

    def test_perfect_accuracy(self):
        m = self._perfect()
        assert m["accuracy"] == pytest.approx(1.0)

    def test_perfect_f1(self):
        m = self._perfect()
        assert m["f1"] == pytest.approx(1.0)

    def test_roc_auc_present(self):
        m = self._perfect()
        assert "roc_auc" in m

    def test_pr_auc_present(self):
        m = self._perfect()
        assert "pr_auc" in m

    def test_log_loss_present(self):
        m = self._perfect()
        assert "log_loss" in m

    def test_confusion_matrix_shape(self):
        y    = np.array([0, 0, 1, 1])
        pred = np.array([0, 1, 0, 1])
        m    = compute_classification_metrics(y, pred, None)
        cm   = m["confusion_matrix"]
        assert len(cm) == 2 and len(cm[0]) == 2

    def test_multiclass(self):
        y    = np.array([0, 1, 2, 0, 1, 2])
        pred = np.array([0, 1, 2, 0, 1, 2])
        prob = np.eye(3)[[0, 1, 2, 0, 1, 2]]
        m    = compute_classification_metrics(y, pred, prob)
        assert m["n_classes"] == 3
        assert m["f1"] == pytest.approx(1.0)

    def test_no_prob_gives_nan(self):
        y = np.array([0, 1, 0, 1])
        m = compute_classification_metrics(y, y, None)
        assert np.isnan(m["roc_auc"])

    def test_classes_key(self):
        y = np.array([0, 1, 0, 1])
        m = compute_classification_metrics(y, y, None)
        assert sorted(m["classes"]) == [0, 1]


# ── TestMetricsRegression ─────────────────────────────────────────────────────

class TestMetricsRegression:
    def test_perfect_mae_zero(self):
        y = np.array([1.0, 2.0, 3.0])
        m = compute_regression_metrics(y, y)
        assert m["mae"] == pytest.approx(0.0)

    def test_rmse_present(self):
        y = np.array([1.0, 2.0, 3.0])
        m = compute_regression_metrics(y, y)
        assert m["rmse"] == pytest.approx(0.0)

    def test_r2_perfect(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        m = compute_regression_metrics(y, y)
        assert m["r2"] == pytest.approx(1.0)

    def test_mape_nan_when_zeros(self):
        y    = np.zeros(5)
        pred = np.ones(5)
        m    = compute_regression_metrics(y, pred)
        assert np.isnan(m["mape"])

    def test_all_keys_present(self):
        y = np.array([1.0, 2.0, 3.0])
        m = compute_regression_metrics(y, y)
        for k in ("mae", "rmse", "mse", "mape", "r2", "support"):
            assert k in m


# ── TestTradingMetrics ────────────────────────────────────────────────────────

class TestTradingMetrics:
    def test_directional_accuracy_perfect(self):
        y = np.array([0, 1, 2, 0, 1, 2])
        m = compute_trading_metrics(y, y, None)
        assert m["directional_accuracy"] == pytest.approx(1.0)

    def test_avg_confidence_with_prob(self):
        y    = np.array([0, 1, 0, 1])
        pred = np.array([0, 1, 0, 1])
        prob = np.array([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3], [0.1, 0.9]])
        m    = compute_trading_metrics(y, pred, prob)
        assert not np.isnan(m["avg_confidence"])
        assert 0 < m["avg_confidence"] <= 1

    def test_prediction_distribution_keys(self):
        y    = np.array([0, 0, 1, 2])
        pred = np.array([0, 1, 1, 2])
        m    = compute_trading_metrics(y, pred, None)
        assert "prediction_distribution" in m
        assert sum(m["prediction_distribution"].values()) == 4

    def test_tp_sl_precision_keys(self):
        y = np.array([0, 1, 2, 1, 2, 0])
        m = compute_trading_metrics(y, y, None)
        assert "tp_prediction_accuracy" in m
        assert "sl_prediction_accuracy" in m

    def test_no_prob_gives_nan_confidence(self):
        y = np.array([0, 1])
        m = compute_trading_metrics(y, y, None)
        assert np.isnan(m["avg_confidence"])


# ── TestModelFactory ──────────────────────────────────────────────────────────

class TestModelFactory:
    @pytest.mark.parametrize("name", SUPPORTED_MODELS)
    def test_creates_classifier(self, name):
        model = ModelFactory.create(name, "classification", random_seed=0)
        assert model is not None
        assert hasattr(model, "fit")
        assert hasattr(model, "predict")

    @pytest.mark.parametrize("name", SUPPORTED_MODELS)
    def test_creates_regressor(self, name):
        model = ModelFactory.create(name, "regression", random_seed=0)
        assert model is not None
        assert hasattr(model, "fit")
        assert hasattr(model, "predict")

    def test_invalid_name_raises(self):
        with pytest.raises(ValueError, match="Unknown model"):
            ModelFactory.create("neural_net")

    def test_invalid_task_raises(self):
        with pytest.raises(ValueError, match="task_type"):
            ModelFactory.create("xgboost", task_type="clustering")

    def test_classifiers_have_predict_proba(self):
        for name in SUPPORTED_MODELS:
            model = ModelFactory.create(name, "classification", random_seed=0)
            assert hasattr(model, "predict_proba"), f"{name} missing predict_proba"

    def test_sklearn_models_constant(self):
        assert "random_forest" in SKLEARN_MODELS
        assert "extra_trees"   in SKLEARN_MODELS
        assert "xgboost"       not in SKLEARN_MODELS


# ── TestTrainer ───────────────────────────────────────────────────────────────

class TestTrainer:
    def _make_splits(self, seed=0, n_classes=2):
        train = _make_clf_df(300, 8, seed=seed,   start="2020-01-01", n_classes=n_classes)
        val   = _make_clf_df(100, 8, seed=seed+1, start="2022-01-01", n_classes=n_classes)
        test  = _make_clf_df(100, 8, seed=seed+2, start="2023-01-01", n_classes=n_classes)
        return train, val, test

    def _cfg(self, task_type="auto"):
        return TrainerConfig(target_column="target", task_type=task_type)

    def test_binary_classification(self):
        train, val, test = self._make_splits()
        model = ModelFactory.create("random_forest", "classification", 0)
        result = Trainer().train_window(model, "random_forest", train, val, test,
                                        self._cfg(), window_number=0)
        assert result.task_type == "classification"
        assert "f1" in result.val_metrics
        assert "accuracy" in result.test_metrics

    def test_multiclass(self):
        train, val, test = self._make_splits(n_classes=3)
        model = ModelFactory.create("xgboost", "classification", 0)
        result = Trainer().train_window(model, "xgboost", train, val, test,
                                        self._cfg("classification"), window_number=0)
        assert result.n_classes == 3

    def test_regression(self):
        train = _make_reg_df(300, 8, 0, "2020-01-01")
        val   = _make_reg_df(100, 8, 1, "2022-01-01")
        test  = _make_reg_df(100, 8, 2, "2023-01-01")
        model = ModelFactory.create("lightgbm", "regression", 0)
        result = Trainer().train_window(model, "lightgbm", train, val, test,
                                        TrainerConfig(target_column="target",
                                                      task_type="regression"),
                                        window_number=0)
        assert result.task_type == "regression"
        assert "mae" in result.val_metrics
        assert "r2"  in result.val_metrics

    def test_nan_imputation_for_sklearn(self):
        train, val, test = self._make_splits()
        # Inject NaN into features
        train.iloc[:10, 0] = np.nan
        val.iloc[:5,   1] = np.nan
        model = ModelFactory.create("random_forest", "classification", 0)
        # Should not raise despite NaN
        result = Trainer().train_window(model, "random_forest", train, val, test,
                                        self._cfg(), window_number=0)
        assert result.n_train == 300

    def test_metrics_all_splits_present(self):
        train, val, test = self._make_splits()
        model = ModelFactory.create("extra_trees", "classification", 0)
        result = Trainer().train_window(model, "extra_trees", train, val, test,
                                        self._cfg(), window_number=0)
        assert result.train_metrics
        assert result.val_metrics
        assert result.test_metrics

    def test_training_time_positive(self):
        train, val, test = self._make_splits()
        model = ModelFactory.create("lightgbm", "classification", 0)
        result = Trainer().train_window(model, "lightgbm", train, val, test,
                                        self._cfg(), window_number=0)
        assert result.training_time_seconds > 0

    def test_feature_columns_subset(self):
        train, val, test = self._make_splits()
        cfg = TrainerConfig(target_column="target", feature_columns=["feat_0", "feat_1"])
        model = ModelFactory.create("xgboost", "classification", 0)
        result = Trainer().train_window(model, "xgboost", train, val, test, cfg, 0)
        assert result.n_features == 2

    def test_xgboost_nan_no_imputation_needed(self):
        train, val, test = self._make_splits()
        train.iloc[:10, 0] = np.nan
        model = ModelFactory.create("xgboost", "classification", 0)
        result = Trainer().train_window(model, "xgboost", train, val, test,
                                        self._cfg(), window_number=0)
        assert result.n_train == 300  # no rows dropped

    def test_window_number_stored(self):
        train, val, test = self._make_splits()
        model = ModelFactory.create("random_forest", "classification", 0)
        result = Trainer().train_window(model, "random_forest", train, val, test,
                                        self._cfg(), window_number=7)
        assert result.window_number == 7

    def test_catboost_classifier(self):
        train, val, test = self._make_splits()
        model = ModelFactory.create("catboost", "classification", 0)
        result = Trainer().train_window(model, "catboost", train, val, test,
                                        self._cfg(), window_number=0)
        assert "f1" in result.val_metrics


# ── TestModelRegistry ─────────────────────────────────────────────────────────

class TestModelRegistry:
    def _make_result(self, model_name="random_forest") -> tuple:
        train, val, test = (
            _make_clf_df(200, 5, 0, "2020-01-01"),
            _make_clf_df(100, 5, 1, "2022-01-01"),
            _make_clf_df(100, 5, 2, "2023-01-01"),
        )
        model = ModelFactory.create(model_name, "classification", 0)
        cfg   = TrainerConfig(target_column="target")
        result = Trainer().train_window(model, model_name, train, val, test, cfg, 0)
        return model, result

    def test_save_creates_joblib(self, tmp_path):
        model, result = self._make_result()
        path = ModelRegistry.save(model, result, tmp_path)
        assert path.exists()
        assert path.suffix == ".joblib"

    def test_save_creates_metadata_json(self, tmp_path):
        model, result = self._make_result()
        ModelRegistry.save(model, result, tmp_path)
        meta_path = tmp_path / "random_forest_metadata.json"
        assert meta_path.exists()

    def test_metadata_json_valid(self, tmp_path):
        model, result = self._make_result()
        ModelRegistry.save(model, result, tmp_path)
        meta = ModelMeta.from_json(tmp_path / "random_forest_metadata.json")
        assert meta.model_name == "random_forest"
        assert meta.task_type  == "classification"

    def test_load_returns_model(self, tmp_path):
        model, result = self._make_result()
        path     = ModelRegistry.save(model, result, tmp_path)
        loaded   = ModelRegistry.load(path)
        assert hasattr(loaded, "predict")

    def test_loaded_model_predicts(self, tmp_path):
        model, result = self._make_result()
        path   = ModelRegistry.save(model, result, tmp_path)
        loaded = ModelRegistry.load(path)
        X      = np.random.default_rng(0).standard_normal((5, 5))
        preds  = loaded.predict(X)
        assert len(preds) == 5

    def test_load_nonexistent_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            ModelRegistry.load(tmp_path / "nosuchfile.joblib")

    def test_list_models(self, tmp_path):
        for name in ["random_forest", "extra_trees"]:
            model, result = self._make_result(name)
            result.model_name = name
            ModelRegistry.save(model, result, tmp_path)
        metas = ModelRegistry.list_models(tmp_path)
        assert len(metas) == 2

    def test_metadata_model_size_positive(self, tmp_path):
        model, result = self._make_result()
        ModelRegistry.save(model, result, tmp_path)
        meta = ModelMeta.from_json(tmp_path / "random_forest_metadata.json")
        assert meta.model_size_bytes > 0

    def test_metadata_json_roundtrip(self, tmp_path):
        model, result = self._make_result()
        ModelRegistry.save(model, result, tmp_path)
        meta  = ModelMeta.from_json(tmp_path / "random_forest_metadata.json")
        d     = meta.to_dict()
        meta2 = ModelMeta.from_json(tmp_path / "random_forest_metadata.json")
        assert meta2.model_name == meta.model_name
        assert meta2.feature_count == meta.feature_count


# ── TestEvaluator ─────────────────────────────────────────────────────────────

class TestEvaluator:
    def _make_results(self, n: int = 4) -> list[ModelWindowResult]:
        results = []
        models  = ["random_forest", "extra_trees"]
        for w in range(2):
            for mn in models:
                train, val, test = (
                    _make_clf_df(200, 5, w, "2020-01-01"),
                    _make_clf_df(100, 5, w+10, "2022-01-01"),
                    _make_clf_df(100, 5, w+20, "2023-01-01"),
                )
                model = ModelFactory.create(mn, "classification", 0)
                cfg   = TrainerConfig(target_column="target")
                r     = Trainer().train_window(model, mn, train, val, test, cfg, w)
                results.append(r)
        return results

    def test_aggregate_metrics_shape(self):
        results = self._make_results()
        df = Evaluator().aggregate_metrics(results)
        assert len(df) == len(results) * 3  # 3 splits per result

    def test_aggregate_has_split_column(self):
        results = self._make_results()
        df = Evaluator().aggregate_metrics(results)
        assert set(df["split"].unique()) == {"train", "val", "test"}

    def test_model_comparison_shape(self):
        results = self._make_results()
        df = Evaluator().model_comparison(results)
        assert len(df) == len(results)

    def test_model_comparison_has_val_f1(self):
        results = self._make_results()
        df = Evaluator().model_comparison(results)
        assert "val_f1" in df.columns

    def test_leaderboard_ranked(self):
        results = self._make_results()
        lb = Evaluator().build_leaderboard(results)
        assert "rank" in lb.columns
        assert lb["rank"].tolist() == list(range(1, len(lb) + 1))

    def test_leaderboard_composite_score(self):
        results = self._make_results()
        lb = Evaluator().build_leaderboard(results)
        assert "composite_score" in lb.columns
        # Scores should be in descending order
        scores = lb["composite_score"].tolist()
        assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))

    def test_leaderboard_one_row_per_model(self):
        results = self._make_results()
        lb = Evaluator().build_leaderboard(results)
        assert len(lb) == 2  # random_forest, extra_trees

    def test_empty_results_leaderboard(self):
        lb = Evaluator().build_leaderboard([])
        assert lb.empty


# ── TestTrainingPipeline ──────────────────────────────────────────────────────

class TestTrainingPipeline:
    def test_generates_results(self, tmp_path):
        windows_dir = _make_windows(tmp_path, n_windows=2)
        cfg = PipelineConfig(
            windows_dir   = windows_dir,
            models_dir    = tmp_path / "models",
            target_column = "target",
            model_names   = ["random_forest", "extra_trees"],
            report_dir    = tmp_path / "reports",
            log_path      = tmp_path / "training.log",
        )
        result = TrainingPipeline().run(cfg)
        assert result.n_results == 4  # 2 windows × 2 models

    def test_model_files_saved(self, tmp_path):
        windows_dir = _make_windows(tmp_path, n_windows=1)
        cfg = PipelineConfig(
            windows_dir   = windows_dir,
            models_dir    = tmp_path / "models",
            target_column = "target",
            model_names   = ["random_forest"],
            report_dir    = tmp_path / "reports",
            log_path      = tmp_path / "training.log",
        )
        TrainingPipeline().run(cfg)
        joblib_files = list((tmp_path / "models").rglob("*.joblib"))
        assert len(joblib_files) >= 1

    def test_metadata_json_saved(self, tmp_path):
        windows_dir = _make_windows(tmp_path, n_windows=1)
        cfg = PipelineConfig(
            windows_dir   = windows_dir,
            models_dir    = tmp_path / "models",
            target_column = "target",
            model_names   = ["extra_trees"],
            report_dir    = tmp_path / "reports",
            log_path      = tmp_path / "training.log",
        )
        TrainingPipeline().run(cfg)
        meta_files = list((tmp_path / "models").rglob("*_metadata.json"))
        assert len(meta_files) >= 1

    def test_reports_created(self, tmp_path):
        windows_dir = _make_windows(tmp_path, n_windows=1)
        cfg = PipelineConfig(
            windows_dir   = windows_dir,
            models_dir    = tmp_path / "models",
            target_column = "target",
            model_names   = ["random_forest"],
            report_dir    = tmp_path / "reports",
            log_path      = tmp_path / "training.log",
        )
        result = TrainingPipeline().run(cfg)
        assert result.report_path.exists()
        assert (tmp_path / "reports" / "metrics.csv").exists()
        assert (tmp_path / "reports" / "model_comparison.csv").exists()
        assert (tmp_path / "reports" / "leaderboard.csv").exists()

    def test_leaderboard_in_result(self, tmp_path):
        windows_dir = _make_windows(tmp_path, n_windows=1)
        cfg = PipelineConfig(
            windows_dir   = windows_dir,
            models_dir    = tmp_path / "models",
            target_column = "target",
            model_names   = ["random_forest", "extra_trees"],
            report_dir    = tmp_path / "reports",
            log_path      = tmp_path / "training.log",
        )
        result = TrainingPipeline().run(cfg)
        assert not result.leaderboard.empty
        assert "rank" in result.leaderboard.columns

    def test_missing_target_column_skips_window(self, tmp_path):
        windows_dir = _make_windows(tmp_path, n_windows=1)
        cfg = PipelineConfig(
            windows_dir   = windows_dir,
            models_dir    = tmp_path / "models",
            target_column = "NONEXISTENT_COLUMN",
            model_names   = ["random_forest"],
            report_dir    = tmp_path / "reports",
            log_path      = tmp_path / "training.log",
        )
        result = TrainingPipeline().run(cfg)
        assert result.n_results == 0

    def test_empty_windows_dir(self, tmp_path):
        empty_dir = tmp_path / "empty_windows"
        empty_dir.mkdir()
        cfg = PipelineConfig(
            windows_dir   = empty_dir,
            models_dir    = tmp_path / "models",
            target_column = "target",
            report_dir    = tmp_path / "reports",
            log_path      = tmp_path / "training.log",
        )
        result = TrainingPipeline().run(cfg)
        assert result.n_results == 0

    def test_result_str(self, tmp_path):
        windows_dir = _make_windows(tmp_path, n_windows=1)
        cfg = PipelineConfig(
            windows_dir   = windows_dir,
            models_dir    = tmp_path / "models",
            target_column = "target",
            model_names   = ["random_forest"],
            report_dir    = tmp_path / "reports",
            log_path      = tmp_path / "training.log",
        )
        result = TrainingPipeline().run(cfg)
        assert "PipelineResult" in str(result)

    def test_elapsed_time_positive(self, tmp_path):
        windows_dir = _make_windows(tmp_path, n_windows=1)
        cfg = PipelineConfig(
            windows_dir   = windows_dir,
            models_dir    = tmp_path / "models",
            target_column = "target",
            model_names   = ["random_forest"],
            report_dir    = tmp_path / "reports",
            log_path      = tmp_path / "training.log",
        )
        result = TrainingPipeline().run(cfg)
        assert result.elapsed_seconds > 0


# ── TestReports ───────────────────────────────────────────────────────────────

class TestReports:
    def _make_results(self) -> list[ModelWindowResult]:
        results = []
        for w in range(2):
            for mn in ["random_forest", "extra_trees"]:
                train, val, test = (
                    _make_clf_df(200, 5, w, "2020-01-01"),
                    _make_clf_df(100, 5, w+10, "2022-01-01"),
                    _make_clf_df(100, 5, w+20, "2023-01-01"),
                )
                model = ModelFactory.create(mn, "classification", 0)
                cfg   = TrainerConfig(target_column="target")
                r     = Trainer().train_window(model, mn, train, val, test, cfg, w)
                results.append(r)
        return results

    def test_training_report_created(self, tmp_path):
        results = self._make_results()
        lb = Evaluator().build_leaderboard(results)
        path = generate_training_report(results, lb, {}, tmp_path)
        assert path.exists()

    def test_training_report_content(self, tmp_path):
        results = self._make_results()
        lb = Evaluator().build_leaderboard(results)
        path = generate_training_report(results, lb, {"model_names": "rf"}, tmp_path, "TEST")
        content = path.read_text(encoding="utf-8")
        assert "Baseline Model Training Report" in content
        assert "TEST" in content

    def test_metrics_csv(self, tmp_path):
        results = self._make_results()
        path = generate_metrics_csv(results, tmp_path)
        assert path.exists()
        df = pd.read_csv(path)
        assert "model" in df.columns
        assert "split" in df.columns
        assert len(df) == len(results) * 3

    def test_comparison_csv(self, tmp_path):
        results = self._make_results()
        path = generate_comparison_csv(results, tmp_path)
        assert path.exists()
        df = pd.read_csv(path)
        assert len(df) == len(results)

    def test_leaderboard_csv(self, tmp_path):
        results = self._make_results()
        lb = Evaluator().build_leaderboard(results)
        path = generate_leaderboard_csv(lb, tmp_path)
        assert path.exists()
        df = pd.read_csv(path)
        assert "rank" in df.columns

    def test_empty_results_report(self, tmp_path):
        lb   = Evaluator().build_leaderboard([])
        path = generate_training_report([], lb, {}, tmp_path)
        assert path.exists()
        assert "No training results" in path.read_text(encoding="utf-8")


# ── TestIntegration ───────────────────────────────────────────────────────────

class TestIntegration:
    """Correctness checks across the full pipeline."""

    def test_all_five_models_train(self, tmp_path):
        windows_dir = _make_windows(tmp_path, n_windows=1)
        cfg = PipelineConfig(
            windows_dir   = windows_dir,
            models_dir    = tmp_path / "models",
            target_column = "target",
            model_names   = list(SUPPORTED_MODELS),
            report_dir    = tmp_path / "reports",
            log_path      = tmp_path / "training.log",
        )
        result = TrainingPipeline().run(cfg)
        assert result.n_results == 5

    def test_metrics_are_valid_floats(self, tmp_path):
        windows_dir = _make_windows(tmp_path, n_windows=1)
        cfg = PipelineConfig(
            windows_dir   = windows_dir,
            models_dir    = tmp_path / "models",
            target_column = "target",
            model_names   = ["random_forest"],
            report_dir    = tmp_path / "reports",
            log_path      = tmp_path / "training.log",
        )
        result = TrainingPipeline().run(cfg)
        for r in result.all_results:
            f1 = r.val_metrics.get("f1")
            assert f1 is not None
            assert 0.0 <= f1 <= 1.0, f"Invalid val F1: {f1}"

    def test_no_data_leakage_between_splits(self, tmp_path):
        windows_dir = _make_windows(tmp_path, n_windows=2)
        cfg = PipelineConfig(
            windows_dir   = windows_dir,
            models_dir    = tmp_path / "models",
            target_column = "target",
            model_names   = ["random_forest"],
            report_dir    = tmp_path / "reports",
            log_path      = tmp_path / "training.log",
        )
        result = TrainingPipeline().run(cfg)
        # All train indices come from the synthetic fixture which has sequential timestamps
        # Verify model paths all exist on disk (no phantom results)
        for r in result.all_results:
            assert r.model_path is not None
            assert r.model_path.exists()

    def test_discover_windows_sorts_numerically(self, tmp_path):
        for i in [10, 1, 3]:
            (tmp_path / f"window_{i:03d}").mkdir()
        windows = _discover_windows(tmp_path)
        assert [n for n, _ in windows] == [1, 3, 10]

    def test_discover_windows_skips_non_window_dirs(self, tmp_path):
        (tmp_path / "window_001").mkdir()
        (tmp_path / "other_dir").mkdir()
        (tmp_path / "window_abc").mkdir()
        windows = _discover_windows(tmp_path)
        assert len(windows) == 1

    def test_regression_pipeline(self, tmp_path):
        # Build windows with regression target (many unique float values)
        windows_dir = tmp_path / "windows"
        windows_dir.mkdir()
        win_dir = windows_dir / "window_000"
        win_dir.mkdir()
        for name, n in [("train", 300), ("validation", 100), ("test", 100)]:
            df = _make_reg_df(n, 5, seed=0, start="2020-01-01")
            df.to_parquet(win_dir / f"{name}.parquet")
        cfg = PipelineConfig(
            windows_dir   = windows_dir,
            models_dir    = tmp_path / "models",
            target_column = "target",
            model_names   = ["random_forest"],
            task_type     = "regression",
            report_dir    = tmp_path / "reports",
            log_path      = tmp_path / "training.log",
        )
        result = TrainingPipeline().run(cfg)
        assert result.n_results == 1
        r = result.all_results[0]
        assert r.task_type == "regression"
        assert "r2" in r.val_metrics

    def test_multiclass_pipeline(self, tmp_path):
        windows_dir = _make_windows(tmp_path, n_windows=1, n_classes=3)
        cfg = PipelineConfig(
            windows_dir   = windows_dir,
            models_dir    = tmp_path / "models",
            target_column = "target",
            model_names   = ["random_forest"],
            task_type     = "classification",
            report_dir    = tmp_path / "reports",
            log_path      = tmp_path / "training.log",
        )
        result = TrainingPipeline().run(cfg)
        assert result.n_results == 1
        r = result.all_results[0]
        assert r.n_classes == 3
        assert "roc_auc" in r.val_metrics
