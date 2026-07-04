"""
src.validation
==============
Walk-Forward Validation Engine — read-only evaluation of optimized models.

Public API
----------
    from src.validation import (
        # Top-level pipeline
        ValidationPipeline,
        ValidationConfig,
        ValidationPipelineResult,
        ModelValidationResult,
        # Acceptance status constants
        PRODUCTION_READY,
        NEEDS_IMPROVEMENT,
        REJECTED,
        # Per-window evaluation
        WindowValidator,
        WindowValidationResult,
        # Multi-window orchestration
        WalkForwardValidator,
        WalkForwardValidationResult,
        # Analysis
        StabilityAnalyzer,
        StabilityResult,
        RobustnessAnalyzer,
        RobustnessResult,
        GeneralizationAnalysis,
        # Metrics
        compute_classification_metrics,
        compute_regression_metrics,
        compute_trading_metrics,
        aggregate_metric_stats,
        # Reports
        generate_all_reports,
    )
"""
from .metrics import (
    aggregate_metric_stats,
    compute_classification_metrics,
    compute_regression_metrics,
    compute_trading_metrics,
)
from .reports import generate_all_reports
from .robustness import (
    GeneralizationAnalysis,
    RobustnessAnalyzer,
    RobustnessResult,
    analyze_robustness,
)
from .stability import (
    StabilityAnalyzer,
    StabilityResult,
    analyze_stability,
)
from .validation_pipeline import (
    NEEDS_IMPROVEMENT,
    PRODUCTION_READY,
    REJECTED,
    ModelValidationResult,
    ValidationConfig,
    ValidationPipeline,
    ValidationPipelineResult,
)
from .validator import WindowValidationResult, WindowValidator
from .walk_forward_validator import WalkForwardValidationResult, WalkForwardValidator

__all__ = [
    # pipeline
    "ValidationPipeline",
    "ValidationConfig",
    "ValidationPipelineResult",
    "ModelValidationResult",
    "PRODUCTION_READY",
    "NEEDS_IMPROVEMENT",
    "REJECTED",
    # per-window
    "WindowValidator",
    "WindowValidationResult",
    # multi-window
    "WalkForwardValidator",
    "WalkForwardValidationResult",
    # analysis
    "StabilityAnalyzer",
    "StabilityResult",
    "analyze_stability",
    "RobustnessAnalyzer",
    "RobustnessResult",
    "GeneralizationAnalysis",
    "analyze_robustness",
    # metrics
    "compute_classification_metrics",
    "compute_regression_metrics",
    "compute_trading_metrics",
    "aggregate_metric_stats",
    # reports
    "generate_all_reports",
]
