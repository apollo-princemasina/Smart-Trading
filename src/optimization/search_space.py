"""
Search Space
============
Per-model hyperparameter search space definitions for Optuna.

Each ``SearchSpace`` class handles two concerns:
1. *suggest* — propose hyperparameter values to an Optuna trial.
2. *build*   — construct a model from a (possibly saved) params dict.

Design note on conditional parameters
--------------------------------------
Optuna records every ``trial.suggest_*`` call by name.  When a parameter
is only suggested conditionally (e.g. ``max_depth`` when depth is not
unlimited), it is simply absent from ``study.best_params`` for trials where
the condition was False.  ``build_*`` methods handle this via ``params.get``.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Optional

import optuna

logger = logging.getLogger(__name__)

SUPPORTED_MODELS: list[str] = [
    "xgboost",
    "lightgbm",
    "catboost",
    "random_forest",
    "extra_trees",
]


# ── Abstract base ─────────────────────────────────────────────────────────────

class BaseSearchSpace(ABC):
    """Abstract base for per-model search spaces."""

    @abstractmethod
    def suggest(self, trial: optuna.Trial) -> dict:
        """Suggest a hyperparameter configuration for this trial.

        Returns a dict that Optuna stores as ``trial.params``.
        The dict may contain auxiliary keys (e.g. ``"unlimited_depth"``)
        that are not passed directly to the model constructor; ``build_*``
        methods strip them.
        """
        ...

    @abstractmethod
    def build_classifier(
        self, params: dict, random_seed: int = 42, n_jobs: int = -1
    ) -> Any:
        """Build a classifier from a params dict (from suggest or best_params)."""
        ...

    @abstractmethod
    def build_regressor(
        self, params: dict, random_seed: int = 42, n_jobs: int = -1
    ) -> Any:
        """Build a regressor from a params dict."""
        ...

    def build(
        self, params: dict, task_type: str, random_seed: int = 42, n_jobs: int = -1
    ) -> Any:
        """Dispatch to build_classifier or build_regressor based on task_type."""
        if task_type == "classification":
            return self.build_classifier(params, random_seed, n_jobs)
        return self.build_regressor(params, random_seed, n_jobs)


# ── XGBoost ───────────────────────────────────────────────────────────────────

class XGBoostSearchSpace(BaseSearchSpace):
    def suggest(self, trial: optuna.Trial) -> dict:
        return {
            "n_estimators":     trial.suggest_int("n_estimators",     50,  500),
            "max_depth":        trial.suggest_int("max_depth",         3,   10),
            "learning_rate":    trial.suggest_float("learning_rate",   1e-3, 0.3,  log=True),
            "subsample":        trial.suggest_float("subsample",       0.5,  1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree",0.5,  1.0),
            "min_child_weight": trial.suggest_int("min_child_weight",  1,   10),
            "gamma":            trial.suggest_float("gamma",           0.0,  1.0),
            "reg_alpha":        trial.suggest_float("reg_alpha",       1e-8, 1.0,  log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda",      1e-8, 10.0, log=True),
        }

    def build_classifier(self, params: dict, random_seed: int = 42, n_jobs: int = -1) -> Any:
        from xgboost import XGBClassifier
        kw = self._clean(params)
        return XGBClassifier(
            **kw, random_state=random_seed, n_jobs=n_jobs,
            verbosity=0, eval_metric="logloss",
        )

    def build_regressor(self, params: dict, random_seed: int = 42, n_jobs: int = -1) -> Any:
        from xgboost import XGBRegressor
        kw = self._clean(params)
        return XGBRegressor(**kw, random_state=random_seed, n_jobs=n_jobs, verbosity=0)

    @staticmethod
    def _clean(params: dict) -> dict:
        return {k: v for k, v in params.items() if k in (
            "n_estimators", "max_depth", "learning_rate", "subsample",
            "colsample_bytree", "min_child_weight", "gamma", "reg_alpha", "reg_lambda",
        )}


# ── LightGBM ──────────────────────────────────────────────────────────────────

class LightGBMSearchSpace(BaseSearchSpace):
    def suggest(self, trial: optuna.Trial) -> dict:
        return {
            "n_estimators":      trial.suggest_int("n_estimators",      50,   500),
            "num_leaves":        trial.suggest_int("num_leaves",         10,   200),
            "learning_rate":     trial.suggest_float("learning_rate",    1e-3, 0.3, log=True),
            "subsample":         trial.suggest_float("subsample",        0.5,  1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5,  1.0),
            "min_child_samples": trial.suggest_int("min_child_samples",  5,    100),
            "reg_alpha":         trial.suggest_float("reg_alpha",        1e-8, 1.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda",       1e-8, 10.0, log=True),
        }

    def build_classifier(self, params: dict, random_seed: int = 42, n_jobs: int = -1) -> Any:
        from lightgbm import LGBMClassifier
        kw = self._clean(params)
        return LGBMClassifier(**kw, random_state=random_seed, n_jobs=n_jobs, verbose=-1)

    def build_regressor(self, params: dict, random_seed: int = 42, n_jobs: int = -1) -> Any:
        from lightgbm import LGBMRegressor
        kw = self._clean(params)
        return LGBMRegressor(**kw, random_state=random_seed, n_jobs=n_jobs, verbose=-1)

    @staticmethod
    def _clean(params: dict) -> dict:
        return {k: v for k, v in params.items() if k in (
            "n_estimators", "num_leaves", "learning_rate", "subsample",
            "colsample_bytree", "min_child_samples", "reg_alpha", "reg_lambda",
        )}


# ── CatBoost ──────────────────────────────────────────────────────────────────

class CatBoostSearchSpace(BaseSearchSpace):
    def suggest(self, trial: optuna.Trial) -> dict:
        return {
            "iterations":          trial.suggest_int("iterations",          50,   500),
            "depth":               trial.suggest_int("depth",               3,    10),
            "learning_rate":       trial.suggest_float("learning_rate",     1e-3, 0.3, log=True),
            "l2_leaf_reg":         trial.suggest_float("l2_leaf_reg",       1e-8, 10.0, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature",0.0,  1.0),
            "border_count":        trial.suggest_int("border_count",        32,   255),
        }

    def build_classifier(self, params: dict, random_seed: int = 42, n_jobs: int = -1) -> Any:
        from catboost import CatBoostClassifier
        kw = self._clean(params)
        return CatBoostClassifier(
            **kw, random_seed=random_seed, verbose=0,
            thread_count=n_jobs, task_type="CPU",
        )

    def build_regressor(self, params: dict, random_seed: int = 42, n_jobs: int = -1) -> Any:
        from catboost import CatBoostRegressor
        kw = self._clean(params)
        return CatBoostRegressor(
            **kw, random_seed=random_seed, verbose=0,
            thread_count=n_jobs, task_type="CPU",
        )

    @staticmethod
    def _clean(params: dict) -> dict:
        return {k: v for k, v in params.items() if k in (
            "iterations", "depth", "learning_rate",
            "l2_leaf_reg", "bagging_temperature", "border_count",
        )}


# ── Random Forest ─────────────────────────────────────────────────────────────

class RandomForestSearchSpace(BaseSearchSpace):
    def suggest(self, trial: optuna.Trial) -> dict:
        unlimited = trial.suggest_categorical("rf_unlimited_depth", [True, False])
        feat_type = trial.suggest_categorical("rf_max_features_type", ["sqrt", "log2", "fraction"])

        params: dict = {
            "rf_unlimited_depth":    unlimited,
            "rf_max_features_type":  feat_type,
            "n_estimators":          trial.suggest_int("n_estimators",     50,  500),
            "min_samples_split":     trial.suggest_int("min_samples_split", 2,   20),
            "min_samples_leaf":      trial.suggest_int("min_samples_leaf",  1,   10),
            "bootstrap":             trial.suggest_categorical("bootstrap", [True, False]),
        }
        if not unlimited:
            params["max_depth"] = trial.suggest_int("max_depth", 3, 20)
        if feat_type == "fraction":
            params["max_features"] = trial.suggest_float("max_features_fraction", 0.1, 1.0)
        else:
            params["max_features"] = feat_type
        return params

    def _constructor_kwargs(self, params: dict) -> dict:
        unlimited = params.get("rf_unlimited_depth", False)
        return {
            "n_estimators":     params["n_estimators"],
            "max_depth":        None if unlimited else params.get("max_depth"),
            "min_samples_split": params["min_samples_split"],
            "min_samples_leaf":  params["min_samples_leaf"],
            "max_features":      params.get("max_features", "sqrt"),
            "bootstrap":         params.get("bootstrap", True),
        }

    def build_classifier(self, params: dict, random_seed: int = 42, n_jobs: int = -1) -> Any:
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(
            **self._constructor_kwargs(params), random_state=random_seed, n_jobs=n_jobs,
        )

    def build_regressor(self, params: dict, random_seed: int = 42, n_jobs: int = -1) -> Any:
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(
            **self._constructor_kwargs(params), random_state=random_seed, n_jobs=n_jobs,
        )


# ── Extra Trees ───────────────────────────────────────────────────────────────

class ExtraTreesSearchSpace(BaseSearchSpace):
    def suggest(self, trial: optuna.Trial) -> dict:
        unlimited = trial.suggest_categorical("et_unlimited_depth", [True, False])
        feat_type = trial.suggest_categorical("et_max_features_type", ["sqrt", "log2", "fraction"])

        params: dict = {
            "et_unlimited_depth":   unlimited,
            "et_max_features_type": feat_type,
            "n_estimators":         trial.suggest_int("n_estimators",     50,  500),
            "min_samples_split":    trial.suggest_int("min_samples_split", 2,   20),
            "min_samples_leaf":     trial.suggest_int("min_samples_leaf",  1,   10),
            "bootstrap":            trial.suggest_categorical("et_bootstrap", [True, False]),
        }
        if not unlimited:
            params["max_depth"] = trial.suggest_int("max_depth", 3, 20)
        if feat_type == "fraction":
            params["max_features"] = trial.suggest_float("et_max_features_fraction", 0.1, 1.0)
        else:
            params["max_features"] = feat_type
        return params

    def _constructor_kwargs(self, params: dict) -> dict:
        unlimited = params.get("et_unlimited_depth", False)
        return {
            "n_estimators":      params["n_estimators"],
            "max_depth":         None if unlimited else params.get("max_depth"),
            "min_samples_split": params["min_samples_split"],
            "min_samples_leaf":  params["min_samples_leaf"],
            "max_features":      params.get("max_features", "sqrt"),
            "bootstrap":         params.get("et_bootstrap", False),
        }

    def build_classifier(self, params: dict, random_seed: int = 42, n_jobs: int = -1) -> Any:
        from sklearn.ensemble import ExtraTreesClassifier
        return ExtraTreesClassifier(
            **self._constructor_kwargs(params), random_state=random_seed, n_jobs=n_jobs,
        )

    def build_regressor(self, params: dict, random_seed: int = 42, n_jobs: int = -1) -> Any:
        from sklearn.ensemble import ExtraTreesRegressor
        return ExtraTreesRegressor(
            **self._constructor_kwargs(params), random_state=random_seed, n_jobs=n_jobs,
        )


# ── Registry ──────────────────────────────────────────────────────────────────

SEARCH_SPACES: dict[str, BaseSearchSpace] = {
    "xgboost":       XGBoostSearchSpace(),
    "lightgbm":      LightGBMSearchSpace(),
    "catboost":      CatBoostSearchSpace(),
    "random_forest": RandomForestSearchSpace(),
    "extra_trees":   ExtraTreesSearchSpace(),
}


def get_search_space(model_name: str) -> BaseSearchSpace:
    """Return the search space for *model_name*.

    Raises:
        ValueError: If model_name is not in SUPPORTED_MODELS.
    """
    name = model_name.lower().strip()
    if name not in SEARCH_SPACES:
        raise ValueError(
            f"Unknown model '{model_name}'. Choose from: {SUPPORTED_MODELS}."
        )
    return SEARCH_SPACES[name]
