"""
src.optimization
================
Hyperparameter Optimization Pipeline using Optuna (Bayesian TPE).

Public API
----------
    from src.optimization import (
        OptimizationPipeline, OptimizationConfig,
        WindowOptResult, PipelineOptResult,
        InferencePipeline, ArtifactManager,
        BundleConfig, ColumnImputer,
        ModelSelector, SelectionResult,
        Optimizer, OptimizerConfig, OptimizationResult,
        EarlyStoppingCallback,
        ObjectiveFunction, compute_objective_score, SUPPORTED_METRICS,
        get_search_space, SEARCH_SPACES, SUPPORTED_MODELS,
        generate_optimization_report,
    )
"""
from .artifact_manager import (
    ArtifactManager,
    BundleConfig,
    ColumnImputer,
    InferencePipeline,
)
from .model_selector import ModelSelector, SelectionResult
from .objective import (
    ObjectiveFunction,
    SUPPORTED_METRICS,
    compute_objective_score,
)
from .optimization_pipeline import (
    OptimizationConfig,
    OptimizationPipeline,
    PipelineOptResult,
    WindowOptResult,
)
from .optimization_reports import generate_optimization_report
from .optimizer import (
    EarlyStoppingCallback,
    OptimizationResult,
    Optimizer,
    OptimizerConfig,
)
from .search_space import (
    SEARCH_SPACES,
    SUPPORTED_MODELS,
    BaseSearchSpace,
    get_search_space,
)

__all__ = [
    # artifact
    "ArtifactManager",
    "BundleConfig",
    "ColumnImputer",
    "InferencePipeline",
    # selector
    "ModelSelector",
    "SelectionResult",
    # objective
    "ObjectiveFunction",
    "SUPPORTED_METRICS",
    "compute_objective_score",
    # pipeline
    "OptimizationConfig",
    "OptimizationPipeline",
    "PipelineOptResult",
    "WindowOptResult",
    # reports
    "generate_optimization_report",
    # optimizer
    "EarlyStoppingCallback",
    "OptimizationResult",
    "Optimizer",
    "OptimizerConfig",
    # search space
    "SEARCH_SPACES",
    "SUPPORTED_MODELS",
    "BaseSearchSpace",
    "get_search_space",
]
