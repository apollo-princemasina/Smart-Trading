"""
Baseline Model Training Pipeline
=================================
Trains XGBoost, LightGBM, CatBoost, Random Forest, and Extra Trees on
walk-forward chronological windows.  No hyperparameter optimisation — only
establishes baseline performance for each window.

Quick start
-----------
    from src.training import TrainingPipeline, PipelineConfig
    from pathlib import Path

    cfg = PipelineConfig(
        windows_dir   = Path("data/ml/windows"),
        models_dir    = Path("models"),
        target_column = "direction_1b",
        model_names   = ["xgboost", "lightgbm", "catboost",
                         "random_forest", "extra_trees"],
    )
    result = TrainingPipeline().run(cfg)
    print(result)
    print(result.leaderboard)
"""
from .evaluation import Evaluator
from .metrics import (
    compute_classification_metrics,
    compute_regression_metrics,
    compute_trading_metrics,
    detect_task_type,
)
from .model_factory import SKLEARN_MODELS, SUPPORTED_MODELS, ModelFactory
from .model_registry import ModelMeta, ModelRegistry
from .reports import (
    generate_comparison_csv,
    generate_leaderboard_csv,
    generate_metrics_csv,
    generate_training_report,
)
from .trainer import ModelWindowResult, Trainer, TrainerConfig
from .training_pipeline import (
    PipelineConfig,
    PipelineResult,
    TrainingPipeline,
    _discover_windows,
)

__all__ = [
    # Pipeline
    "TrainingPipeline",
    "PipelineConfig",
    "PipelineResult",
    # Trainer
    "Trainer",
    "TrainerConfig",
    "ModelWindowResult",
    # Factory
    "ModelFactory",
    "SUPPORTED_MODELS",
    "SKLEARN_MODELS",
    # Registry
    "ModelRegistry",
    "ModelMeta",
    # Evaluation
    "Evaluator",
    # Metrics
    "detect_task_type",
    "compute_classification_metrics",
    "compute_regression_metrics",
    "compute_trading_metrics",
    # Reports
    "generate_training_report",
    "generate_metrics_csv",
    "generate_comparison_csv",
    "generate_leaderboard_csv",
    # Helpers
    "_discover_windows",
]
