"""
src.feature_quality
===================

Feature Quality Analysis, Feature Selection, and Feature Explainability framework.

Quick start
-----------
::

    from src.feature_quality import FeatureQualityPipeline

    pipeline = FeatureQualityPipeline(
        feature_store=store,
        output_dir="reports",
        config={"classification": True, "skip_boruta": False},
    )
    results = pipeline.run_for_symbol("EURUSD", target_col="label_direction")

    # Access results
    top50 = results.top_features(50)
    print(results.selection_result.selected_features)
"""

from __future__ import annotations

# ── Core ─────────────────────────────────────────────────────────────────────
from .feature_quality import (
    FeatureQualityAnalyzer,
    FeatureQualityResults,
    FeatureScore,
)
from .quality_pipeline import FeatureQualityPipeline

# ── Module reports ────────────────────────────────────────────────────────────
from .missing_values     import MissingValueAnalyzer,   MissingValueReport
from .duplicate_features import DuplicateFeatureDetector, DuplicateReport
from .constant_features  import ConstantFeatureDetector,  ConstantReport
from .variance_filter    import VarianceFilter,           VarianceReport
from .correlation        import CorrelationAnalyzer,      CorrelationReport
from .vif                import VIFAnalyzer,              VIFReport
from .psi                import PSICalculator,            PSIReport, compute_psi
from .drift_detection    import DriftDetector,            DriftReport
from .leakage_detector   import LeakageDetector,          LeakageReport
from .feature_importance import TreeImportanceAnalyzer,   ImportanceReport
from .permutation_importance import PermutationImportanceAnalyzer, PermImportanceReport
from .mutual_information import MutualInformationAnalyzer, MIReport
from .shap_analysis      import SHAPAnalyzer,             SHAPReport
from .boruta_selection   import BorutaSelector,           BorutaReport
from .recursive_feature_elimination import RFESelector,   RFEReport
from .stability_analysis import StabilityAnalyzer,        StabilityReport
from .feature_clustering import FeatureClusterer,         ClusterReport
from .feature_selector   import FeatureSelector,          SelectionResult
from .feature_reports    import FeatureReportGenerator

__all__ = [
    # Core
    "FeatureQualityAnalyzer",
    "FeatureQualityResults",
    "FeatureQualityPipeline",
    "FeatureScore",
    # Analysers
    "MissingValueAnalyzer",       "MissingValueReport",
    "DuplicateFeatureDetector",   "DuplicateReport",
    "ConstantFeatureDetector",    "ConstantReport",
    "VarianceFilter",             "VarianceReport",
    "CorrelationAnalyzer",        "CorrelationReport",
    "VIFAnalyzer",                "VIFReport",
    "PSICalculator",              "PSIReport",       "compute_psi",
    "DriftDetector",              "DriftReport",
    "LeakageDetector",            "LeakageReport",
    "TreeImportanceAnalyzer",     "ImportanceReport",
    "PermutationImportanceAnalyzer", "PermImportanceReport",
    "MutualInformationAnalyzer",  "MIReport",
    "SHAPAnalyzer",               "SHAPReport",
    "BorutaSelector",             "BorutaReport",
    "RFESelector",                "RFEReport",
    "StabilityAnalyzer",          "StabilityReport",
    "FeatureClusterer",           "ClusterReport",
    "FeatureSelector",            "SelectionResult",
    "FeatureReportGenerator",
]
