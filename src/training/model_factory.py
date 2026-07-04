"""
Model Factory
=============
Creates ML model instances with sensible default (baseline) parameters.

No hyperparameter tuning is performed.  All models use their out-of-the-box
configurations augmented with a fixed random seed and parallelism setting.

Supported models
----------------
xgboost       — XGBClassifier / XGBRegressor
lightgbm      — LGBMClassifier / LGBMRegressor
catboost      — CatBoostClassifier / CatBoostRegressor
random_forest — RandomForestClassifier / RandomForestRegressor
extra_trees   — ExtraTreesClassifier / ExtraTreesRegressor

Constants
---------
SUPPORTED_MODELS : ordered list of model name strings
SKLEARN_MODELS   : subset that does NOT handle NaN natively (need imputation)
"""
from __future__ import annotations

from typing import Any

SUPPORTED_MODELS: list[str] = [
    "xgboost",
    "lightgbm",
    "catboost",
    "random_forest",
    "extra_trees",
]

# These sklearn estimators do not handle NaN internally; the Trainer imputes.
SKLEARN_MODELS: frozenset[str] = frozenset({"random_forest", "extra_trees"})


class ModelFactory:
    """Creates fresh, unfitted model instances with default parameters."""

    @staticmethod
    def create(
        model_name:  str,
        task_type:   str = "classification",
        random_seed: int = 42,
        n_jobs:      int = -1,
    ) -> Any:
        """Return a freshly constructed (unfitted) estimator.

        Args:
            model_name:  One of SUPPORTED_MODELS.
            task_type:   "classification" or "regression".
            random_seed: RNG seed for reproducibility.
            n_jobs:      Parallelism level (-1 = all cores).

        Raises:
            ValueError:  If model_name or task_type is unrecognised.
            ImportError: If the required library is not installed.
        """
        name = model_name.lower().strip()
        if name not in SUPPORTED_MODELS:
            raise ValueError(
                f"Unknown model '{model_name}'. "
                f"Choose from: {SUPPORTED_MODELS}."
            )
        if task_type == "classification":
            return ModelFactory._classifier(name, random_seed, n_jobs)
        if task_type == "regression":
            return ModelFactory._regressor(name, random_seed, n_jobs)
        raise ValueError(
            f"task_type must be 'classification' or 'regression'. Got: '{task_type}'."
        )

    # ── Classifiers ───────────────────────────────────────────────────────────

    @staticmethod
    def _classifier(name: str, seed: int, n_jobs: int) -> Any:
        if name == "xgboost":
            from xgboost import XGBClassifier
            return XGBClassifier(
                n_estimators=100, max_depth=6, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8,
                random_state=seed, n_jobs=n_jobs,
                verbosity=0, eval_metric="logloss",
            )
        if name == "lightgbm":
            from lightgbm import LGBMClassifier
            return LGBMClassifier(
                n_estimators=100, num_leaves=31, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8,
                random_state=seed, n_jobs=n_jobs, verbose=-1,
            )
        if name == "catboost":
            from catboost import CatBoostClassifier
            return CatBoostClassifier(
                iterations=100, depth=6, learning_rate=0.1,
                random_seed=seed, verbose=0,
                thread_count=n_jobs, task_type="CPU",
            )
        if name == "random_forest":
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(
                n_estimators=100, max_depth=None, min_samples_split=2,
                random_state=seed, n_jobs=n_jobs,
            )
        if name == "extra_trees":
            from sklearn.ensemble import ExtraTreesClassifier
            return ExtraTreesClassifier(
                n_estimators=100, max_depth=None, min_samples_split=2,
                random_state=seed, n_jobs=n_jobs,
            )

    # ── Regressors ────────────────────────────────────────────────────────────

    @staticmethod
    def _regressor(name: str, seed: int, n_jobs: int) -> Any:
        if name == "xgboost":
            from xgboost import XGBRegressor
            return XGBRegressor(
                n_estimators=100, max_depth=6, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8,
                random_state=seed, n_jobs=n_jobs, verbosity=0,
            )
        if name == "lightgbm":
            from lightgbm import LGBMRegressor
            return LGBMRegressor(
                n_estimators=100, num_leaves=31, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8,
                random_state=seed, n_jobs=n_jobs, verbose=-1,
            )
        if name == "catboost":
            from catboost import CatBoostRegressor
            return CatBoostRegressor(
                iterations=100, depth=6, learning_rate=0.1,
                random_seed=seed, verbose=0,
                thread_count=n_jobs, task_type="CPU",
            )
        if name == "random_forest":
            from sklearn.ensemble import RandomForestRegressor
            return RandomForestRegressor(
                n_estimators=100, max_depth=None, min_samples_split=2,
                random_state=seed, n_jobs=n_jobs,
            )
        if name == "extra_trees":
            from sklearn.ensemble import ExtraTreesRegressor
            return ExtraTreesRegressor(
                n_estimators=100, max_depth=None, min_samples_split=2,
                random_state=seed, n_jobs=n_jobs,
            )
