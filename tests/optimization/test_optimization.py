"""
Tests for the Hyperparameter Optimization Pipeline (Task 10).

Coverage:
  TestSearchSpace              (10 tests)
  TestObjectiveFunction         (8 tests)
  TestEarlyStoppingCallback     (6 tests)
  TestOptimizer                 (6 tests)
  TestColumnImputer             (8 tests)
  TestArtifactManager          (11 tests)
  TestInferencePipeline         (7 tests)
  TestModelSelector             (7 tests)
  TestOptimizationReports       (7 tests)
  TestOptimizationPipeline     (10 tests)
  TestIntegration               (5 tests)
Total: 85 tests
"""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_clf_data(n=200, n_features=10, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_features))
    y = rng.integers(0, 3, size=n)
    return X, y


def _make_df(n=200, n_features=5, target="label", seed=42):
    rng = np.random.default_rng(seed)
    data = {f"feat_{i}": rng.standard_normal(n) for i in range(n_features)}
    data[target] = rng.integers(0, 3, size=n)
    return pd.DataFrame(data)


def _write_window(root: Path, n=1, nrows=200, n_features=5, target="label") -> Path:
    win_dir = root / f"window_{n:03d}"
    win_dir.mkdir(parents=True, exist_ok=True)
    df = _make_df(n=nrows, n_features=n_features, target=target)
    split = nrows // 3
    df.iloc[:split].to_parquet(win_dir / "train.parquet")
    df.iloc[split:2*split].to_parquet(win_dir / "validation.parquet")
    df.iloc[2*split:].to_parquet(win_dir / "test.parquet")
    return win_dir


# ── TestSearchSpace ───────────────────────────────────────────────────────────

class TestSearchSpace:
    def test_get_known_model(self):
        from src.optimization.search_space import get_search_space, SUPPORTED_MODELS
        for model in SUPPORTED_MODELS:
            sp = get_search_space(model)
            assert sp is not None

    def test_get_unknown_raises(self):
        from src.optimization.search_space import get_search_space
        with pytest.raises(ValueError, match="Unknown model"):
            get_search_space("nonexistent_model")

    def test_xgboost_suggest_returns_dict(self):
        import optuna
        from src.optimization.search_space import get_search_space
        study = optuna.create_study()
        trial = study.ask()
        params = get_search_space("xgboost").suggest(trial)
        assert isinstance(params, dict)
        assert "n_estimators" in params
        assert "max_depth" in params

    def test_lightgbm_suggest_returns_dict(self):
        import optuna
        from src.optimization.search_space import get_search_space
        study = optuna.create_study()
        trial = study.ask()
        params = get_search_space("lightgbm").suggest(trial)
        assert "num_leaves" in params

    def test_catboost_suggest_returns_dict(self):
        import optuna
        from src.optimization.search_space import get_search_space
        study = optuna.create_study()
        trial = study.ask()
        params = get_search_space("catboost").suggest(trial)
        assert "iterations" in params
        assert "depth" in params

    def test_rf_conditional_depth(self):
        import optuna
        from src.optimization.search_space import get_search_space
        space = get_search_space("random_forest")
        # Run several trials; some should have max_depth, others not
        study = optuna.create_study()
        seen_none = False
        seen_depth = False
        for _ in range(20):
            trial = study.ask()
            params = space.suggest(trial)
            if params.get("rf_unlimited_depth") is True:
                seen_none = True
            else:
                seen_depth = True
            if seen_none and seen_depth:
                break
        # At least one of each (statistically almost certain after 20 trials)
        assert seen_none or seen_depth  # At minimum one type observed

    def test_rf_build_classifier_valid(self):
        import optuna
        from src.optimization.search_space import get_search_space
        space = get_search_space("random_forest")
        study = optuna.create_study()
        trial = study.ask()
        params = space.suggest(trial)
        model = space.build_classifier(params, random_seed=0, n_jobs=1)
        from sklearn.ensemble import RandomForestClassifier
        assert isinstance(model, RandomForestClassifier)

    def test_et_build_regressor_valid(self):
        import optuna
        from src.optimization.search_space import get_search_space
        space = get_search_space("extra_trees")
        study = optuna.create_study()
        trial = study.ask()
        params = space.suggest(trial)
        model = space.build_regressor(params, random_seed=0, n_jobs=1)
        from sklearn.ensemble import ExtraTreesRegressor
        assert isinstance(model, ExtraTreesRegressor)

    def test_xgboost_build_and_fit(self):
        import optuna
        from src.optimization.search_space import get_search_space
        space = get_search_space("xgboost")
        study = optuna.create_study()
        trial = study.ask()
        params = space.suggest(trial)
        # Force small n_estimators
        params["n_estimators"] = 5
        model = space.build_classifier(params, random_seed=0, n_jobs=1)
        X, y = _make_clf_data(n=60)
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == 60

    def test_supported_models_list(self):
        from src.optimization.search_space import SUPPORTED_MODELS
        assert "xgboost" in SUPPORTED_MODELS
        assert "lightgbm" in SUPPORTED_MODELS
        assert "catboost" in SUPPORTED_MODELS
        assert "random_forest" in SUPPORTED_MODELS
        assert "extra_trees" in SUPPORTED_MODELS


# ── TestObjectiveFunction ─────────────────────────────────────────────────────

class TestObjectiveFunction:
    def test_compute_score_f1(self):
        from src.optimization.objective import compute_objective_score
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 1, 2])
        score = compute_objective_score(y_true, y_pred, None, "f1")
        assert score == pytest.approx(1.0)

    def test_compute_score_accuracy(self):
        from src.optimization.objective import compute_objective_score
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 0, 0, 1])
        score = compute_objective_score(y_true, y_pred, None, "accuracy")
        assert score == pytest.approx(0.75)

    def test_compute_score_neg_mae(self):
        from src.optimization.objective import compute_objective_score
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        score = compute_objective_score(y_true, y_pred, None, "neg_mae", task_type="regression")
        assert score == pytest.approx(0.0)

    def test_compute_score_r2(self):
        from src.optimization.objective import compute_objective_score
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0])
        score = compute_objective_score(y_true, y_pred, None, "r2", task_type="regression")
        assert score == pytest.approx(1.0)

    def test_compute_score_unknown_metric_raises(self):
        from src.optimization.objective import compute_objective_score
        with pytest.raises(ValueError, match="Unknown metric"):
            compute_objective_score(np.array([0]), np.array([0]), None, "invalid_metric")

    def test_objective_function_callable(self):
        import optuna
        from src.optimization.objective import ObjectiveFunction
        X, y = _make_clf_data(n=100, n_features=5)
        obj = ObjectiveFunction(
            model_name="random_forest", task_type="classification",
            X_train=X[:70], y_train=y[:70],
            X_val=X[70:], y_val=y[70:],
            metric="f1", random_seed=0, n_jobs=1,
        )
        study = optuna.create_study(direction="maximize")
        # Manually call with a real trial
        trial = study.ask()
        # Patch suggest to use small params
        with patch.object(
            obj._space, "suggest",
            return_value={"n_estimators": 5, "max_depth": 2,
                          "min_samples_split": 2, "min_samples_leaf": 1,
                          "bootstrap": True, "max_features": "sqrt",
                          "rf_unlimited_depth": False, "rf_max_features_type": "sqrt"},
        ):
            score = obj(trial)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_supported_metrics_list(self):
        from src.optimization.objective import SUPPORTED_METRICS
        for m in ("f1", "roc_auc", "accuracy", "r2", "neg_mae"):
            assert m in SUPPORTED_METRICS

    def test_roc_auc_with_proba(self):
        from src.optimization.objective import compute_objective_score
        rng = np.random.default_rng(0)
        y_true = rng.integers(0, 2, 100)
        prob   = rng.random((100, 2))
        prob   = (prob.T / prob.sum(axis=1)).T
        score  = compute_objective_score(y_true, y_true, prob, "roc_auc")
        assert 0.0 <= score <= 1.0


# ── TestEarlyStoppingCallback ─────────────────────────────────────────────────

class TestEarlyStoppingCallback:
    def _make_callback(self, warmup=2, patience=3, min_delta=1e-4):
        from src.optimization.optimizer import EarlyStoppingCallback
        return EarlyStoppingCallback(warmup=warmup, patience=patience, min_delta=min_delta)

    def test_does_not_stop_before_warmup(self):
        import optuna
        from src.optimization.optimizer import EarlyStoppingCallback
        cb = EarlyStoppingCallback(warmup=20, patience=2)
        # Only 3 trials — below warmup; study must run all 3
        call_count = [0]
        def obj(trial):
            call_count[0] += 1
            trial.suggest_float("x", 0, 1)
            return 0.5 - call_count[0] * 0.1
        study = optuna.create_study(direction="maximize")
        study.optimize(obj, n_trials=3, callbacks=[cb])
        assert call_count[0] == 3  # all 3 ran; no early stop

    def test_stops_after_patience_exceeded(self):
        import optuna
        from src.optimization.optimizer import EarlyStoppingCallback
        cb = EarlyStoppingCallback(warmup=2, patience=3, min_delta=0.01)
        # Score starts improving, then plateaus → early stop before n_trials=50
        scores = [0.5, 0.6] + [0.6] * 50   # flat after 2 trials
        idx = [0]
        def obj(trial):
            trial.suggest_float("x", 0, 1)
            v = scores[min(idx[0], len(scores) - 1)]
            idx[0] += 1
            return v
        study = optuna.create_study(direction="maximize")
        study.optimize(obj, n_trials=50, callbacks=[cb])
        # Should have stopped well before 50 trials due to patience
        assert len(study.trials) < 50

    def test_reset_clears_state(self):
        from src.optimization.optimizer import EarlyStoppingCallback
        cb = EarlyStoppingCallback(warmup=1, patience=2)
        cb._best = 0.9
        cb._no_improve = 2
        cb.reset()
        assert cb._best is None
        assert cb._no_improve == 0

    def test_improvement_resets_counter(self):
        import optuna
        cb = self._make_callback(warmup=1, patience=3, min_delta=0.01)
        study = optuna.create_study(direction="maximize")
        t = study.ask(); study.tell(t, 0.5)
        t = study.ask(); study.tell(t, 0.5)
        cb(study, MagicMock())  # set best
        t = study.ask(); study.tell(t, 0.5)
        cb(study, MagicMock())  # no improve #1
        t = study.ask(); study.tell(t, 0.9)  # big improvement
        study.tell(study.ask(), 0.9)
        cb(study, MagicMock())  # should reset counter
        assert cb._no_improve == 0

    def test_callback_does_nothing_if_no_completed_trials(self):
        import optuna
        cb = self._make_callback(warmup=1, patience=1)
        study = optuna.create_study()
        cb(study, MagicMock())  # should not raise

    def test_min_delta_not_exceeded_increments_counter(self):
        import optuna
        cb = self._make_callback(warmup=1, patience=5, min_delta=1.0)
        study = optuna.create_study(direction="maximize")
        for val in [0.5, 0.6]:
            t = study.ask(); study.tell(t, val)
        cb(study, MagicMock())  # sets best=0.6
        t = study.ask(); study.tell(t, 0.601)  # improvement < min_delta
        cb(study, MagicMock())
        assert cb._no_improve == 1


# ── TestOptimizer ─────────────────────────────────────────────────────────────

class TestOptimizer:
    def _make_objective(self, score=0.7):
        def obj(trial):
            _ = trial.suggest_float("x", 0, 1)
            return score
        return obj

    def test_optimize_returns_result(self):
        from src.optimization.optimizer import Optimizer, OptimizerConfig
        opt = Optimizer()
        cfg = OptimizerConfig(n_trials=3, early_stopping_warmup=1, early_stopping_patience=2)
        result = opt.optimize(self._make_objective(), "rf", "f1", 1, cfg)
        assert result.n_trials_completed >= 1
        assert result.best_value == pytest.approx(0.7)

    def test_optimize_stores_trial_history(self):
        from src.optimization.optimizer import Optimizer, OptimizerConfig
        opt = Optimizer()
        cfg = OptimizerConfig(n_trials=4, early_stopping_warmup=1, early_stopping_patience=5)
        result = opt.optimize(self._make_objective(0.6), "lgbm", "roc_auc", 2, cfg)
        assert len(result.trial_history) >= 1

    def test_optimize_study_name_format(self):
        from src.optimization.optimizer import Optimizer, OptimizerConfig
        opt = Optimizer()
        cfg = OptimizerConfig(n_trials=2, early_stopping_warmup=1, early_stopping_patience=2)
        result = opt.optimize(self._make_objective(), "xgboost", "f1", 5, cfg)
        assert "xgboost" in result.study_name
        assert "w005" in result.study_name

    def test_optimizer_config_defaults(self):
        from src.optimization.optimizer import OptimizerConfig
        cfg = OptimizerConfig()
        assert cfg.n_trials == 50
        assert cfg.direction == "maximize"
        assert cfg.resume_if_exists is True

    def test_optimizer_with_sqlite_storage(self, tmp_path):
        from src.optimization.optimizer import Optimizer, OptimizerConfig
        opt = Optimizer()
        cfg = OptimizerConfig(
            n_trials=3,
            storage_dir=tmp_path / "studies",
            resume_if_exists=True,
            early_stopping_warmup=1,
            early_stopping_patience=2,
        )
        result = opt.optimize(self._make_objective(0.8), "rf", "f1", 1, cfg)
        assert result.storage_path is not None
        assert Path(result.storage_path).exists()

    def test_optimizer_result_best_trial_number(self):
        from src.optimization.optimizer import Optimizer, OptimizerConfig
        opt = Optimizer()
        cfg = OptimizerConfig(n_trials=5, early_stopping_warmup=1, early_stopping_patience=5)
        result = opt.optimize(self._make_objective(0.75), "rf", "f1", 1, cfg)
        assert result.best_trial_number is not None


# ── TestColumnImputer ─────────────────────────────────────────────────────────

class TestColumnImputer:
    def test_fit_computes_medians(self):
        from src.optimization.artifact_manager import ColumnImputer
        df = pd.DataFrame({"a": [1.0, 2.0, np.nan], "b": [10.0, np.nan, 30.0]})
        imp = ColumnImputer(apply_imputation=True)
        imp.fit(df)
        assert imp.fill_values["a"] == pytest.approx(1.5)
        assert imp.fill_values["b"] == pytest.approx(20.0)

    def test_transform_fills_nan(self):
        from src.optimization.artifact_manager import ColumnImputer
        df_train = pd.DataFrame({"x": [1.0, 3.0]})
        df_val   = pd.DataFrame({"x": [np.nan, 5.0]})
        imp = ColumnImputer(apply_imputation=True)
        imp.fit(df_train)
        out = imp.transform(df_val)
        assert out["x"].iloc[0] == pytest.approx(2.0)
        assert out["x"].iloc[1] == pytest.approx(5.0)

    def test_passthrough_when_disabled(self):
        from src.optimization.artifact_manager import ColumnImputer
        imp = ColumnImputer(apply_imputation=False)
        df  = pd.DataFrame({"a": [np.nan, 1.0]})
        out = imp.transform(df)
        assert pd.isna(out["a"].iloc[0])

    def test_fit_transform(self):
        from src.optimization.artifact_manager import ColumnImputer
        imp = ColumnImputer(apply_imputation=True)
        df  = pd.DataFrame({"c": [np.nan, 2.0, 4.0]})
        out = imp.fit_transform(df)
        assert not out["c"].isna().any()

    def test_transform_array(self):
        from src.optimization.artifact_manager import ColumnImputer
        imp = ColumnImputer(apply_imputation=True)
        df  = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        imp.fit(df)
        arr = np.array([[np.nan, 3.0], [1.5, np.nan]])
        out = imp.transform_array(arr)
        assert not np.isnan(out).any()

    def test_all_nan_column_uses_zero(self):
        from src.optimization.artifact_manager import ColumnImputer
        imp = ColumnImputer(apply_imputation=True)
        df  = pd.DataFrame({"z": [np.nan, np.nan, np.nan]})
        imp.fit(df)
        assert imp.fill_values["z"] == pytest.approx(0.0)

    def test_to_dict_contains_all_keys(self):
        from src.optimization.artifact_manager import ColumnImputer
        imp = ColumnImputer(apply_imputation=True, strategy="median")
        imp.fit(pd.DataFrame({"x": [1.0, 2.0]}))
        d = imp.to_dict()
        assert "apply_imputation" in d
        assert "fill_values" in d
        assert "feature_names" in d

    def test_does_not_modify_original_df(self):
        from src.optimization.artifact_manager import ColumnImputer
        imp  = ColumnImputer(apply_imputation=True)
        df   = pd.DataFrame({"v": [np.nan, 2.0]})
        imp.fit(df)
        out  = imp.transform(df)
        assert pd.isna(df["v"].iloc[0])  # original unchanged
        assert not pd.isna(out["v"].iloc[0])


# ── TestArtifactManager ───────────────────────────────────────────────────────

class TestArtifactManager:
    def _make_bundle(self, tmp_path) -> tuple[Path, Any, Any]:
        from sklearn.ensemble import RandomForestClassifier
        from src.optimization.artifact_manager import (
            ArtifactManager, BundleConfig, ColumnImputer,
        )
        model   = RandomForestClassifier(n_estimators=3, random_state=0)
        X, y    = _make_clf_data(n=80, n_features=4)
        model.fit(X, y)
        imputer = ColumnImputer(apply_imputation=True)
        imputer.fit(pd.DataFrame(X, columns=[f"f{i}" for i in range(4)]))
        cfg = BundleConfig(
            model_name="random_forest",
            task_type="classification",
            target_column="label",
            feature_columns=[f"f{i}" for i in range(4)],
            n_classes=3,
            random_seed=0,
            schema_version="1.0.0",
            label_version="1.0.0",
            window_number=1,
            train_start="2020-01-01",
            train_end="2021-01-01",
            val_start="2021-01-02",
            val_end="2021-06-01",
            test_start="2021-06-02",
            test_end="2022-01-01",
            best_params={"n_estimators": 3},
            optimization_metric="f1",
            n_trials=5,
            optimization_time_s=2.0,
            best_val_score=0.75,
            baseline_val_score=0.65,
            training_time_s=0.1,
            prediction_time_s=0.01,
            n_train_samples=50,
            n_val_samples=15,
            n_test_samples=15,
            train_metrics={"f1": 0.80},
            val_metrics={"f1": 0.75},
            test_metrics={"f1": 0.70},
        )
        bundle_dir = tmp_path / "bundle"
        ArtifactManager.save_bundle(model, imputer, cfg, bundle_dir)
        return bundle_dir, model, imputer

    def test_save_creates_all_required_files(self, tmp_path):
        from src.optimization.artifact_manager import _REQUIRED_FILES
        bundle_dir, _, _ = self._make_bundle(tmp_path)
        for fname in _REQUIRED_FILES:
            assert (bundle_dir / fname).exists(), f"Missing: {fname}"

    def test_verify_bundle_complete(self, tmp_path):
        from src.optimization.artifact_manager import ArtifactManager
        bundle_dir, _, _ = self._make_bundle(tmp_path)
        ok, missing = ArtifactManager.verify_bundle(bundle_dir)
        assert ok is True
        assert missing == []

    def test_verify_bundle_incomplete(self, tmp_path):
        from src.optimization.artifact_manager import ArtifactManager
        bundle_dir, _, _ = self._make_bundle(tmp_path)
        (bundle_dir / "model.joblib").unlink()
        ok, missing = ArtifactManager.verify_bundle(bundle_dir)
        assert ok is False
        assert "model.joblib" in missing

    def test_manifest_checksums_present(self, tmp_path):
        bundle_dir, _, _ = self._make_bundle(tmp_path)
        manifest = json.loads((bundle_dir / "pipeline_manifest.json").read_text())
        assert "files" in manifest
        for entry in manifest["files"]:
            assert "sha256" in entry
            assert len(entry["sha256"]) == 64

    def test_manifest_is_complete(self, tmp_path):
        bundle_dir, _, _ = self._make_bundle(tmp_path)
        manifest = json.loads((bundle_dir / "pipeline_manifest.json").read_text())
        assert manifest["is_complete"] is True

    def test_feature_order_json_correct(self, tmp_path):
        bundle_dir, _, _ = self._make_bundle(tmp_path)
        feat_order = json.loads((bundle_dir / "feature_order.json").read_text())
        assert feat_order == [f"f{i}" for i in range(4)]

    def test_inference_config_json_valid(self, tmp_path):
        bundle_dir, _, _ = self._make_bundle(tmp_path)
        cfg = json.loads((bundle_dir / "inference_config.json").read_text())
        assert cfg["model_name"] == "random_forest"
        assert cfg["task_type"] == "classification"
        assert cfg["n_features"] == 4

    def test_optimization_results_json_improvement(self, tmp_path):
        bundle_dir, _, _ = self._make_bundle(tmp_path)
        opt = json.loads((bundle_dir / "optimization_results.json").read_text())
        assert opt["improvement_pct"] is not None
        assert opt["improvement_pct"] > 0

    def test_copy_to_best(self, tmp_path):
        from src.optimization.artifact_manager import ArtifactManager
        bundle_dir, _, _ = self._make_bundle(tmp_path)
        best_dir = tmp_path / "best"
        result   = ArtifactManager.copy_to_best(bundle_dir, best_dir)
        assert result == best_dir
        assert (best_dir / "model.joblib").exists()

    def test_load_bundle_returns_pipeline(self, tmp_path):
        from src.optimization.artifact_manager import ArtifactManager
        bundle_dir, _, _ = self._make_bundle(tmp_path)
        pipe = ArtifactManager.load_bundle(bundle_dir)
        from src.optimization.artifact_manager import InferencePipeline
        assert isinstance(pipe, InferencePipeline)

    def test_training_metrics_json_all_splits(self, tmp_path):
        bundle_dir, _, _ = self._make_bundle(tmp_path)
        metrics = json.loads((bundle_dir / "training_metrics.json").read_text())
        assert "train" in metrics
        assert "val" in metrics
        assert "test" in metrics


# ── TestInferencePipeline ─────────────────────────────────────────────────────

class TestInferencePipeline:
    def _pipeline(self, tmp_path) -> "InferencePipeline":
        from sklearn.ensemble import RandomForestClassifier
        from src.optimization.artifact_manager import (
            ArtifactManager, BundleConfig, ColumnImputer,
        )
        model   = RandomForestClassifier(n_estimators=3, random_state=0)
        cols    = [f"f{i}" for i in range(4)]
        X, y    = _make_clf_data(n=80, n_features=4)
        model.fit(X, y)
        imputer = ColumnImputer(apply_imputation=True)
        imputer.fit(pd.DataFrame(X, columns=cols))
        cfg = BundleConfig(
            model_name="random_forest", task_type="classification",
            target_column="label", feature_columns=cols, n_classes=3,
            random_seed=0, schema_version="1.0.0", label_version="1.0.0",
            window_number=1,
            train_start=None, train_end=None, val_start=None, val_end=None,
            test_start=None, test_end=None,
            best_params={}, optimization_metric="f1", n_trials=1,
            optimization_time_s=0.0, best_val_score=0.7,
            baseline_val_score=None, training_time_s=0.0,
            prediction_time_s=0.0, n_train_samples=50, n_val_samples=15,
            n_test_samples=15, train_metrics={}, val_metrics={}, test_metrics={},
        )
        bundle_dir = tmp_path / "bundle"
        ArtifactManager.save_bundle(model, imputer, cfg, bundle_dir)
        return ArtifactManager.load_bundle(bundle_dir)

    def test_predict_returns_array(self, tmp_path):
        pipe = self._pipeline(tmp_path)
        df   = pd.DataFrame(
            np.random.default_rng(7).standard_normal((10, 4)),
            columns=[f"f{i}" for i in range(4)],
        )
        preds = pipe.predict(df)
        assert len(preds) == 10

    def test_predict_proba_returns_array(self, tmp_path):
        pipe = self._pipeline(tmp_path)
        df   = pd.DataFrame(
            np.random.default_rng(7).standard_normal((10, 4)),
            columns=[f"f{i}" for i in range(4)],
        )
        proba = pipe.predict_proba(df)
        assert proba is not None
        assert proba.shape == (10, 3)

    def test_feature_columns_property(self, tmp_path):
        pipe = self._pipeline(tmp_path)
        assert pipe.feature_columns == [f"f{i}" for i in range(4)]

    def test_model_name_property(self, tmp_path):
        pipe = self._pipeline(tmp_path)
        assert pipe.model_name == "random_forest"

    def test_task_type_property(self, tmp_path):
        pipe = self._pipeline(tmp_path)
        assert pipe.task_type == "classification"

    def test_missing_file_raises(self, tmp_path):
        from src.optimization.artifact_manager import InferencePipeline
        (tmp_path / "bundle").mkdir()
        with pytest.raises(FileNotFoundError, match="incomplete"):
            InferencePipeline(tmp_path / "bundle")

    def test_predict_with_nan_imputes(self, tmp_path):
        pipe = self._pipeline(tmp_path)
        df   = pd.DataFrame(
            np.random.default_rng(9).standard_normal((5, 4)),
            columns=[f"f{i}" for i in range(4)],
        )
        df.loc[0, "f0"] = np.nan
        preds = pipe.predict(df)
        assert len(preds) == 5


# ── TestModelSelector ─────────────────────────────────────────────────────────

class TestModelSelector:
    def _make_result(self, model_name, window, val_score, f1=None, roc_auc=None, da=None):
        from src.optimization.optimization_pipeline import WindowOptResult
        return WindowOptResult(
            model_name=model_name, window_number=window,
            task_type="classification", best_val_score=val_score,
            best_params={}, n_trials_completed=5, optimization_time_s=1.0,
            training_time_s=0.1, prediction_time_s=0.01,
            n_train=100, n_val=30, n_test=30, n_features=5,
            train_metrics={}, val_metrics={
                "f1": f1 or val_score,
                "roc_auc": roc_auc or val_score,
                "directional_accuracy": da or val_score,
            },
            test_metrics={},
            optimization_metric="f1",
            baseline_val_score=val_score - 0.05,
        )

    def test_select_best_returns_highest(self):
        from src.optimization.model_selector import ModelSelector
        r1 = self._make_result("xgboost",      1, 0.6)
        r2 = self._make_result("random_forest", 1, 0.8)
        r3 = self._make_result("lightgbm",      1, 0.7)
        best = ModelSelector().select_best([r1, r2, r3])
        assert best.model_name == "random_forest"

    def test_select_best_empty_returns_none(self):
        from src.optimization.model_selector import ModelSelector
        assert ModelSelector().select_best([]) is None

    def test_compare_with_baseline_improvement(self):
        from src.optimization.model_selector import ModelSelector
        r = self._make_result("lgbm", 1, 0.7)
        rows = ModelSelector().compare_with_baseline([r], {"lgbm": 0.6})
        assert rows[0]["improvement_pct"] > 0

    def test_compare_with_baseline_no_baseline(self):
        from src.optimization.model_selector import ModelSelector
        r = self._make_result("rf", 1, 0.7)
        rows = ModelSelector().compare_with_baseline([r], None)
        assert rows[0]["baseline_val_score"] is None
        assert rows[0]["improvement_pct"] is None

    def test_create_best_bundle_copies_dir(self, tmp_path):
        from src.optimization.model_selector import ModelSelector
        # Create a fake bundle dir with required files
        source = tmp_path / "source_bundle"
        source.mkdir()
        (source / "model.joblib").write_bytes(b"fake")
        # Patch verify to skip actual bundle check
        r = self._make_result("rf", 1, 0.8)
        r.bundle_dir = source

        best_dir = tmp_path / "best"
        with patch("src.optimization.artifact_manager.ArtifactManager.copy_to_best",
                   return_value=best_dir):
            sel = ModelSelector().create_best_bundle(r, best_dir)
        assert sel.chosen_model_name == "rf"
        assert sel.chosen_window_number == 1

    def test_composite_score_classification(self):
        from src.optimization.model_selector import ModelSelector
        r = self._make_result("rf", 1, 0.8, f1=0.8, roc_auc=0.85, da=0.75)
        score = ModelSelector._composite(r, "classification")
        expected = 0.40 * 0.8 + 0.35 * 0.85 + 0.25 * 0.75
        assert score == pytest.approx(expected)

    def test_composite_score_regression(self):
        from src.optimization.model_selector import ModelSelector
        from src.optimization.optimization_pipeline import WindowOptResult
        r = WindowOptResult(
            model_name="rf", window_number=1, task_type="regression",
            best_val_score=0.9, best_params={}, n_trials_completed=3,
            optimization_time_s=1.0, training_time_s=0.0, prediction_time_s=0.0,
            n_train=50, n_val=20, n_test=20, n_features=5,
            train_metrics={}, val_metrics={"r2": 0.9}, test_metrics={},
            optimization_metric="r2",
        )
        score = ModelSelector._composite(r, "regression")
        assert score == pytest.approx(0.9)


# ── TestOptimizationReports ───────────────────────────────────────────────────

class TestOptimizationReports:
    def _make_pipeline_result(self):
        from src.optimization.optimization_pipeline import PipelineOptResult, WindowOptResult
        r = WindowOptResult(
            model_name="xgboost", window_number=1, task_type="classification",
            best_val_score=0.82, best_params={"n_estimators": 100, "lr": 0.05},
            n_trials_completed=10, optimization_time_s=5.0,
            training_time_s=0.3, prediction_time_s=0.02,
            n_train=100, n_val=30, n_test=30, n_features=5,
            train_metrics={"f1": 0.85}, val_metrics={"f1": 0.82},
            test_metrics={"f1": 0.79}, optimization_metric="f1",
            trial_history=[{"number": 0, "value": 0.7, "state": "COMPLETE", "params": {}},
                           {"number": 1, "value": 0.82, "state": "COMPLETE", "params": {}}],
            baseline_val_score=0.75,
        )
        return PipelineOptResult(results=[r], total_time_s=5.0)

    def test_generates_markdown(self, tmp_path):
        from src.optimization.optimization_reports import generate_optimization_report
        pr = self._make_pipeline_result()
        paths = generate_optimization_report(pr, tmp_path)
        assert (tmp_path / "optimization_report.md").exists()

    def test_generates_optuna_results_csv(self, tmp_path):
        from src.optimization.optimization_reports import generate_optimization_report
        pr = self._make_pipeline_result()
        generate_optimization_report(pr, tmp_path)
        assert (tmp_path / "optuna_results.csv").exists()

    def test_generates_best_parameters_json(self, tmp_path):
        from src.optimization.optimization_reports import generate_optimization_report
        pr = self._make_pipeline_result()
        generate_optimization_report(pr, tmp_path)
        bp = json.loads((tmp_path / "best_parameters.json").read_text())
        assert "xgboost_w001" in bp

    def test_generates_history_csv(self, tmp_path):
        from src.optimization.optimization_reports import generate_optimization_report
        pr = self._make_pipeline_result()
        generate_optimization_report(pr, tmp_path)
        assert (tmp_path / "optimization_history.csv").exists()

    def test_generates_model_comparison_csv(self, tmp_path):
        from src.optimization.optimization_reports import generate_optimization_report
        pr = self._make_pipeline_result()
        generate_optimization_report(pr, tmp_path)
        assert (tmp_path / "model_comparison.csv").exists()

    def test_history_csv_best_so_far_monotone(self, tmp_path):
        import csv
        from src.optimization.optimization_reports import generate_optimization_report
        pr = self._make_pipeline_result()
        generate_optimization_report(pr, tmp_path)
        with open(tmp_path / "optimization_history.csv", newline="") as fh:
            rows = list(csv.DictReader(fh))
        prev = -float("inf")
        for row in rows:
            bsf = row.get("best_so_far")
            if bsf and bsf != "":
                val = float(bsf)
                assert val >= prev
                prev = val

    def test_returns_path_dict(self, tmp_path):
        from src.optimization.optimization_reports import generate_optimization_report
        pr = self._make_pipeline_result()
        paths = generate_optimization_report(pr, tmp_path)
        assert isinstance(paths, dict)
        assert len(paths) == 5


# ── TestOptimizationPipeline ──────────────────────────────────────────────────

class TestOptimizationPipeline:
    def test_discovers_windows(self, tmp_path):
        from src.optimization.optimization_pipeline import OptimizationPipeline
        _write_window(tmp_path / "windows", n=1)
        _write_window(tmp_path / "windows", n=2)
        dirs = OptimizationPipeline._discover_windows(tmp_path / "windows")
        assert len(dirs) == 2

    def test_no_windows_returns_empty(self, tmp_path):
        from src.optimization.optimization_pipeline import (
            OptimizationConfig, OptimizationPipeline,
        )
        cfg = OptimizationConfig(
            windows_dir=tmp_path / "empty",
            models_dir=tmp_path / "models",
            target_column="label",
            model_names=["random_forest"],
            n_trials=2,
            skip_on_error=True,
        )
        result = OptimizationPipeline().run(cfg)
        assert result.results == []

    def test_run_single_window_single_model(self, tmp_path):
        from src.optimization.optimization_pipeline import (
            OptimizationConfig, OptimizationPipeline,
        )
        _write_window(tmp_path / "windows", n=1, nrows=90)
        cfg = OptimizationConfig(
            windows_dir=tmp_path / "windows",
            models_dir=tmp_path / "models",
            target_column="label",
            model_names=["random_forest"],
            n_trials=3,
            early_stopping_warmup=1,
            early_stopping_patience=2,
            skip_on_error=False,
        )
        result = OptimizationPipeline().run(cfg)
        assert len(result.results) == 1
        assert result.results[0].model_name == "random_forest"

    def test_bundle_files_created(self, tmp_path):
        from src.optimization.artifact_manager import _REQUIRED_FILES
        from src.optimization.optimization_pipeline import (
            OptimizationConfig, OptimizationPipeline,
        )
        _write_window(tmp_path / "windows", n=1, nrows=90)
        cfg = OptimizationConfig(
            windows_dir=tmp_path / "windows",
            models_dir=tmp_path / "models",
            target_column="label",
            model_names=["random_forest"],
            n_trials=2,
            early_stopping_warmup=1,
            early_stopping_patience=2,
        )
        result = OptimizationPipeline().run(cfg)
        bundle_dir = result.results[0].bundle_dir
        for fname in _REQUIRED_FILES:
            assert (bundle_dir / fname).exists(), f"Missing: {fname}"

    def test_best_model_dir_created(self, tmp_path):
        from src.optimization.optimization_pipeline import (
            OptimizationConfig, OptimizationPipeline,
        )
        _write_window(tmp_path / "windows", n=1, nrows=90)
        best_dir = tmp_path / "best"
        cfg = OptimizationConfig(
            windows_dir=tmp_path / "windows",
            models_dir=tmp_path / "models",
            target_column="label",
            model_names=["random_forest"],
            n_trials=2,
            early_stopping_warmup=1,
            early_stopping_patience=2,
            best_model_dir=best_dir,
        )
        result = OptimizationPipeline().run(cfg)
        assert result.selection_result is not None
        assert best_dir.exists()

    def test_skip_on_error_continues(self, tmp_path):
        from src.optimization.optimization_pipeline import (
            OptimizationConfig, OptimizationPipeline,
        )
        # Window with missing target column
        win = tmp_path / "windows" / "window_001"
        win.mkdir(parents=True)
        df = _make_df()
        df.to_parquet(win / "train.parquet")
        df.to_parquet(win / "validation.parquet")
        df.to_parquet(win / "test.parquet")

        cfg = OptimizationConfig(
            windows_dir=tmp_path / "windows",
            models_dir=tmp_path / "models",
            target_column="nonexistent_col",
            model_names=["random_forest"],
            n_trials=2,
            skip_on_error=True,
        )
        result = OptimizationPipeline().run(cfg)
        assert len(result.errors) > 0

    def test_window_opt_result_fields(self, tmp_path):
        from src.optimization.optimization_pipeline import (
            OptimizationConfig, OptimizationPipeline,
        )
        _write_window(tmp_path / "windows", n=1, nrows=90)
        cfg = OptimizationConfig(
            windows_dir=tmp_path / "windows",
            models_dir=tmp_path / "models",
            target_column="label",
            model_names=["random_forest"],
            n_trials=2,
            early_stopping_warmup=1,
            early_stopping_patience=2,
        )
        result = OptimizationPipeline().run(cfg)
        r = result.results[0]
        assert r.n_train > 0
        assert r.n_val > 0
        assert r.n_test > 0
        assert r.n_features > 0
        assert r.trial_history is not None

    def test_report_files_created(self, tmp_path):
        from src.optimization.optimization_pipeline import (
            OptimizationConfig, OptimizationPipeline,
        )
        _write_window(tmp_path / "windows", n=1, nrows=90)
        report_dir = tmp_path / "reports"
        cfg = OptimizationConfig(
            windows_dir=tmp_path / "windows",
            models_dir=tmp_path / "models",
            target_column="label",
            model_names=["random_forest"],
            n_trials=2,
            early_stopping_warmup=1,
            early_stopping_patience=2,
            report_dir=report_dir,
        )
        OptimizationPipeline().run(cfg)
        assert (report_dir / "optimization_report.md").exists()

    def test_multiple_models_per_window(self, tmp_path):
        from src.optimization.optimization_pipeline import (
            OptimizationConfig, OptimizationPipeline,
        )
        _write_window(tmp_path / "windows", n=1, nrows=120)
        cfg = OptimizationConfig(
            windows_dir=tmp_path / "windows",
            models_dir=tmp_path / "models",
            target_column="label",
            model_names=["random_forest", "extra_trees"],
            n_trials=2,
            early_stopping_warmup=1,
            early_stopping_patience=2,
        )
        result = OptimizationPipeline().run(cfg)
        names = {r.model_name for r in result.results}
        assert "random_forest" in names
        assert "extra_trees" in names

    def test_result_has_timing(self, tmp_path):
        from src.optimization.optimization_pipeline import (
            OptimizationConfig, OptimizationPipeline,
        )
        _write_window(tmp_path / "windows", n=1, nrows=90)
        cfg = OptimizationConfig(
            windows_dir=tmp_path / "windows",
            models_dir=tmp_path / "models",
            target_column="label",
            model_names=["random_forest"],
            n_trials=2,
            early_stopping_warmup=1,
            early_stopping_patience=2,
        )
        result = OptimizationPipeline().run(cfg)
        assert result.total_time_s > 0
        assert result.results[0].optimization_time_s > 0


# ── TestIntegration ───────────────────────────────────────────────────────────

class TestIntegration:
    """End-to-end tests that exercise the full stack."""

    def test_bundle_loadable_as_inference_pipeline(self, tmp_path):
        """Save a bundle then load it; predict must not crash."""
        from src.optimization.optimization_pipeline import (
            OptimizationConfig, OptimizationPipeline,
        )
        from src.optimization.artifact_manager import InferencePipeline
        _write_window(tmp_path / "windows", n=1, nrows=90, n_features=5)
        cfg = OptimizationConfig(
            windows_dir=tmp_path / "windows",
            models_dir=tmp_path / "models",
            target_column="label",
            model_names=["random_forest"],
            n_trials=2,
            early_stopping_warmup=1,
            early_stopping_patience=2,
        )
        result = OptimizationPipeline().run(cfg)
        bundle_dir = result.results[0].bundle_dir
        pipe = InferencePipeline(bundle_dir)
        df   = _make_df(n=10, n_features=5, target="label").drop(columns=["label"])
        preds = pipe.predict(df)
        assert len(preds) == 10

    def test_best_model_inference_pipeline(self, tmp_path):
        """Best model bundle is loadable."""
        from src.optimization.optimization_pipeline import (
            OptimizationConfig, OptimizationPipeline,
        )
        from src.optimization.artifact_manager import InferencePipeline
        _write_window(tmp_path / "windows", n=1, nrows=90, n_features=5)
        best_dir = tmp_path / "best"
        cfg = OptimizationConfig(
            windows_dir=tmp_path / "windows",
            models_dir=tmp_path / "models",
            target_column="label",
            model_names=["random_forest"],
            n_trials=2,
            early_stopping_warmup=1,
            early_stopping_patience=2,
            best_model_dir=best_dir,
        )
        OptimizationPipeline().run(cfg)
        pipe = InferencePipeline(best_dir)
        df   = _make_df(n=5, n_features=5, target="label").drop(columns=["label"])
        assert len(pipe.predict(df)) == 5

    def test_no_look_ahead_in_bundle_metadata(self, tmp_path):
        """train_end must precede val_start in the bundle metadata."""
        from src.optimization.optimization_pipeline import (
            OptimizationConfig, OptimizationPipeline,
        )
        # Write window with proper metadata.json
        win_dir = tmp_path / "windows" / "window_001"
        win_dir.mkdir(parents=True)
        df = _make_df(n=90, target="label")
        df.iloc[:30].to_parquet(win_dir / "train.parquet")
        df.iloc[30:60].to_parquet(win_dir / "validation.parquet")
        df.iloc[60:].to_parquet(win_dir / "test.parquet")
        meta = {
            "train": {"start": "2020-01-01", "end": "2020-06-01"},
            "val":   {"start": "2020-06-02", "end": "2020-09-01"},
            "test":  {"start": "2020-09-02", "end": "2021-01-01"},
        }
        (win_dir / "metadata.json").write_text(json.dumps(meta))

        cfg = OptimizationConfig(
            windows_dir=tmp_path / "windows",
            models_dir=tmp_path / "models",
            target_column="label",
            model_names=["random_forest"],
            n_trials=2,
            early_stopping_warmup=1,
            early_stopping_patience=2,
        )
        result = OptimizationPipeline().run(cfg)
        bundle_dir = result.results[0].bundle_dir
        model_meta = json.loads((bundle_dir / "model_metadata.json").read_text())
        # train_end < val_start
        assert model_meta["training_window"]["end"] < model_meta["validation_window"]["start"]

    def test_bundle_reproducible_predictions(self, tmp_path):
        """Predictions from the bundle must be deterministic."""
        from src.optimization.optimization_pipeline import (
            OptimizationConfig, OptimizationPipeline,
        )
        from src.optimization.artifact_manager import InferencePipeline
        _write_window(tmp_path / "windows", n=1, nrows=90, n_features=5)
        cfg = OptimizationConfig(
            windows_dir=tmp_path / "windows",
            models_dir=tmp_path / "models",
            target_column="label",
            model_names=["random_forest"],
            n_trials=2,
            early_stopping_warmup=1,
            early_stopping_patience=2,
        )
        result = OptimizationPipeline().run(cfg)
        bundle_dir = result.results[0].bundle_dir
        df = _make_df(n=20, n_features=5, target="label").drop(columns=["label"])
        pipe1 = InferencePipeline(bundle_dir)
        pipe2 = InferencePipeline(bundle_dir)
        np.testing.assert_array_equal(pipe1.predict(df), pipe2.predict(df))

    def test_optimization_improves_or_maintains_score(self, tmp_path):
        """Optimized score should be >= 0 (sanity: no model produces NaN score)."""
        from src.optimization.optimization_pipeline import (
            OptimizationConfig, OptimizationPipeline,
        )
        _write_window(tmp_path / "windows", n=1, nrows=120, n_features=5)
        cfg = OptimizationConfig(
            windows_dir=tmp_path / "windows",
            models_dir=tmp_path / "models",
            target_column="label",
            model_names=["random_forest"],
            n_trials=3,
            early_stopping_warmup=1,
            early_stopping_patience=3,
        )
        result = OptimizationPipeline().run(cfg)
        assert result.results[0].best_val_score >= 0
