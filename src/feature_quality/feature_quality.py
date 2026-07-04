"""
FeatureQualityAnalyzer — main orchestrator for the feature quality framework.

Every feature receives six sub-scores and a composite score:

  composite = 0.20 × quality
            + 0.30 × importance
            + 0.20 × stability
            + 0.15 × leakage        (100 = no leakage)
            + 0.15 × drift_inv      (100 = no drift)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .boruta_selection import BorutaReport, BorutaSelector
from .constant_features import ConstantReport, ConstantFeatureDetector
from .correlation import CorrelationReport, CorrelationAnalyzer
from .drift_detection import DriftReport, DriftDetector
from .duplicate_features import DuplicateReport, DuplicateFeatureDetector
from .feature_clustering import ClusterReport, FeatureClusterer
from .feature_importance import ImportanceReport, TreeImportanceAnalyzer
from .feature_selector import SelectionResult, FeatureSelector
from .leakage_detector import LeakageReport, LeakageDetector
from .missing_values import MissingValueReport, MissingValueAnalyzer
from .mutual_information import MIReport, MutualInformationAnalyzer
from .permutation_importance import PermImportanceReport, PermutationImportanceAnalyzer
from .psi import PSIReport, PSICalculator
from .recursive_feature_elimination import RFEReport, RFESelector
from .shap_analysis import SHAPReport, SHAPAnalyzer
from .stability_analysis import StabilityReport, StabilityAnalyzer
from .variance_filter import VarianceReport, VarianceFilter
from .vif import VIFReport, VIFAnalyzer

logger = logging.getLogger(__name__)


# =============================================================================
# Data structures
# =============================================================================


@dataclass
class FeatureScore:
    """Per-feature aggregated quality score."""

    name:                  str
    quality_score:         float = 0.0
    importance_score:      float = 0.0
    stability_score:       float = 0.0
    leakage_score:         float = 0.0    # 100 = no leakage
    drift_score:           float = 0.0    # 100 = no drift
    interpretability_score: float = 0.0
    composite_score:       float = 0.0
    rank:                  int   = 0
    flags:                 list[str] = field(default_factory=list)
    metadata:              dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name":                   self.name,
            "rank":                   self.rank,
            "composite_score":        round(self.composite_score, 2),
            "quality_score":          round(self.quality_score, 2),
            "importance_score":       round(self.importance_score, 2),
            "stability_score":        round(self.stability_score, 2),
            "leakage_score":          round(self.leakage_score, 2),
            "drift_score":            round(self.drift_score, 2),
            "interpretability_score": round(self.interpretability_score, 2),
            "flags":                  self.flags,
        }


@dataclass
class FeatureQualityResults:
    """Aggregated results from all quality-analysis modules."""

    symbol:          str
    pipeline_version: str = "1.0.0"

    # Per-feature scores (computed last, after all modules)
    feature_scores:  dict[str, FeatureScore] = field(default_factory=dict)

    # Module reports (all optional — some may be skipped)
    missing_report:     MissingValueReport  | None = None
    duplicate_report:   DuplicateReport     | None = None
    constant_report:    ConstantReport      | None = None
    variance_report:    VarianceReport      | None = None
    correlation_report: CorrelationReport   | None = None
    vif_report:         VIFReport           | None = None
    leakage_report:     LeakageReport       | None = None
    psi_report:         PSIReport           | None = None
    drift_report:       DriftReport         | None = None
    importance_report:  ImportanceReport    | None = None
    perm_report:        PermImportanceReport | None = None
    mi_report:          MIReport            | None = None
    shap_report:        SHAPReport          | None = None
    boruta_report:      BorutaReport        | None = None
    rfe_report:         RFEReport           | None = None
    stability_report:   StabilityReport     | None = None
    cluster_report:     ClusterReport       | None = None
    selection_result:   SelectionResult     | None = None

    # ── Derived properties ────────────────────────────────────────────────────

    def ranked_features(self) -> list[str]:
        """Feature names sorted by composite score (best first)."""
        return [
            name
            for name, _ in sorted(
                self.feature_scores.items(),
                key=lambda x: x[1].composite_score,
                reverse=True,
            )
        ]

    def top_features(self, n: int = 50) -> list[str]:
        return self.ranked_features()[:n]

    def flagged_features(self) -> dict[str, list[str]]:
        """Return a dict of {feature: [flags]} for all features with flags."""
        return {
            name: s.flags
            for name, s in self.feature_scores.items()
            if s.flags
        }


# =============================================================================
# Main analyser
# =============================================================================


class FeatureQualityAnalyzer:
    """
    Orchestrate all feature quality, importance, stability, and selection modules.

    Parameters
    ----------
    config:
        Optional dict overriding any default setting (see ``DEFAULT_CONFIG``).
    """

    DEFAULT_CONFIG: dict[str, Any] = {
        "missing_threshold":           0.30,
        "variance_threshold":          1e-5,
        "correlation_threshold":       0.95,
        "vif_threshold":               10.0,
        "psi_threshold":               0.20,
        "leakage_correlation_threshold": 0.90,
        "drift_ks_alpha":              0.05,
        "max_importance_samples":      50_000,
        "max_shap_samples":            10_000,
        "max_rfe_samples":             30_000,
        "max_boruta_samples":          50_000,
        "max_stability_samples":       10_000,
        "n_stability_windows":         10,
        "boruta_max_iter":             50,
        "feature_selection_strategy":  "voting",
        "min_selection_votes":         2,
        "classification":              True,
        "skip_boruta":                 False,
        "skip_rfe":                    False,
        "skip_shap":                   False,
        "skip_permutation":            False,
        "skip_stability":              False,
        "skip_vif":                    False,
        "skip_correlation":            False,
    }

    def __init__(
        self,
        feature_store=None,
        config: dict[str, Any] | None = None,
        pipeline_version: str = "1.0.0",
    ):
        self._store   = feature_store
        self._cfg     = {**self.DEFAULT_CONFIG, **(config or {})}
        self._pip_ver = pipeline_version

    def run(
        self,
        df:          pd.DataFrame,
        symbol:      str,
        target:      pd.Series | None = None,
        target_col:  str | None = None,
    ) -> FeatureQualityResults:
        """
        Run the full feature quality pipeline on *df*.

        Parameters
        ----------
        df:
            Feature DataFrame (read-only — never modified).
        symbol:
            Instrument identifier.
        target:
            Optional target Series (aligned index).
        target_col:
            If provided, extract the target from this column of *df* and drop
            it from the feature set.
        """
        results = FeatureQualityResults(symbol=symbol, pipeline_version=self._pip_ver)
        cfg     = self._cfg

        # ── Extract target ────────────────────────────────────────────────────
        if target is None and target_col and target_col in df.columns:
            target = df[target_col]
            df     = df.drop(columns=[target_col])

        # Work on a copy (never mutate caller's data)
        df = df.copy()
        # Drop non-numeric columns silently for analysis
        feature_df = df.select_dtypes(include=[np.number])

        logger.info(
            "FeatureQualityAnalyzer.run: symbol=%s  features=%d  rows=%d",
            symbol, feature_df.shape[1], len(feature_df),
        )

        # ── Module 1 — Data quality ───────────────────────────────────────────
        results.missing_report   = MissingValueAnalyzer(cfg["missing_threshold"]).fit(df)
        results.duplicate_report = DuplicateFeatureDetector().fit(feature_df)
        results.constant_report  = ConstantFeatureDetector().fit(feature_df)
        results.variance_report  = VarianceFilter(cfg["variance_threshold"]).fit(feature_df)

        # ── Module 2 — Correlation ────────────────────────────────────────────
        if not cfg["skip_correlation"]:
            results.correlation_report = CorrelationAnalyzer(
                threshold=cfg["correlation_threshold"]
            ).fit(feature_df)

        # ── Module 3 — VIF ────────────────────────────────────────────────────
        if not cfg["skip_vif"]:
            results.vif_report = VIFAnalyzer(threshold=cfg["vif_threshold"]).fit(feature_df)

        # ── Module 4 — Leakage ───────────────────────────────────────────────
        results.leakage_report = LeakageDetector(
            correlation_threshold=cfg["leakage_correlation_threshold"]
        ).fit(feature_df, target)

        # ── Module 5 — Drift / PSI ────────────────────────────────────────────
        results.psi_report   = PSICalculator(threshold=cfg["psi_threshold"]).fit(feature_df)
        results.drift_report = DriftDetector(psi_threshold=cfg["psi_threshold"]).fit(feature_df)

        # ── Modules 6-8 — Importance (require target) ─────────────────────────
        if target is not None:
            clf = bool(cfg["classification"])

            results.importance_report = TreeImportanceAnalyzer(
                max_samples=cfg["max_importance_samples"],
                classification=clf,
            ).fit(feature_df, target)

            results.mi_report = MutualInformationAnalyzer(
                classification=clf,
                max_samples=cfg["max_importance_samples"],
            ).fit(feature_df, target)

            if not cfg["skip_permutation"]:
                results.perm_report = PermutationImportanceAnalyzer(
                    max_samples=min(20_000, cfg["max_importance_samples"]),
                    classification=clf,
                ).fit(feature_df, target)

            if not cfg["skip_shap"]:
                results.shap_report = SHAPAnalyzer(
                    max_samples=cfg["max_shap_samples"],
                    classification=clf,
                ).fit(feature_df, target)

            if not cfg["skip_stability"]:
                results.stability_report = StabilityAnalyzer(
                    n_windows=cfg["n_stability_windows"],
                    max_samples_per_window=cfg["max_stability_samples"],
                    classification=clf,
                ).fit(feature_df, target)

            if not cfg["skip_boruta"]:
                results.boruta_report = BorutaSelector(
                    max_iter=cfg["boruta_max_iter"],
                    classification=clf,
                    max_samples=cfg["max_boruta_samples"],
                ).fit(feature_df, target)

            if not cfg["skip_rfe"]:
                results.rfe_report = RFESelector(
                    classification=clf,
                    max_samples=cfg["max_rfe_samples"],
                ).fit(feature_df, target)

        # ── Module 10 — Feature clustering ────────────────────────────────────
        results.cluster_report = FeatureClusterer(
            correlation_threshold=cfg["correlation_threshold"]
        ).fit(feature_df)

        # ── Module 11 — Composite scoring ─────────────────────────────────────
        results.feature_scores = self._compute_scores(results, feature_df.columns.tolist())

        # ── Module 9 — Feature selection ──────────────────────────────────────
        selector = FeatureSelector(
            strategy=cfg["feature_selection_strategy"],
            min_votes=cfg["min_selection_votes"],
        )
        results.selection_result = selector.select_from_report(results)

        logger.info(
            "Analysis complete: %d features scored, %d selected",
            len(results.feature_scores),
            len(results.selection_result.selected_features) if results.selection_result else 0,
        )
        return results

    # ── Scoring engine ────────────────────────────────────────────────────────

    def _compute_scores(
        self,
        r:    FeatureQualityResults,
        cols: list[str],
    ) -> dict[str, FeatureScore]:

        scores: dict[str, FeatureScore] = {}

        # ── Pre-compute normalised importance (0-100) ──────────────────────────
        imp_norm = _normalise_to_100(
            self._combined_importance(r, cols)
        )

        # ── Pre-compute stability scores ────────────────────────────────────────
        stab_norm = (
            r.stability_report.stability_scores.reindex(cols).fillna(50.0)
            if r.stability_report is not None
            else pd.Series(50.0, index=cols)
        )

        # ── Pre-compute leakage penalties ──────────────────────────────────────
        leakage_raw = (
            r.leakage_report.leakage_scores.reindex(cols).fillna(0.0)
            if r.leakage_report is not None
            else pd.Series(0.0, index=cols)
        )
        leakage_score = (1.0 - leakage_raw.clip(0, 1)) * 100.0

        # ── Pre-compute drift (PSI-based) ──────────────────────────────────────
        psi_raw = (
            r.psi_report.psi_scores.reindex(cols).fillna(0.0)
            if r.psi_report is not None
            else pd.Series(0.0, index=cols)
        )
        drift_score = (1.0 - (psi_raw / 0.20).clip(0, 1)) * 100.0

        # ── SHAP interpretability ─────────────────────────────────────────────
        shap_norm = _normalise_to_100(
            r.shap_report.mean_abs_shap.reindex(cols).fillna(0.0)
            if (r.shap_report is not None and r.shap_report.available)
            else pd.Series(0.0, index=cols)
        )

        for col in cols:
            flags: list[str] = []

            # ── Quality score ─────────────────────────────────────────────────
            q = 100.0
            if r.missing_report is not None:
                missing_rate = float(r.missing_report.missing_rates.get(col, 0.0))
                q *= (1.0 - missing_rate)
                if missing_rate > self._cfg["missing_threshold"]:
                    flags.append("high_missing")
            if r.constant_report is not None:
                if col in r.constant_report.constant_features:
                    q = 0.0
                    flags.append("constant")
                elif col in r.constant_report.near_constant_features:
                    q *= 0.10
                    flags.append("near_constant")
            if r.variance_report is not None:
                if col in r.variance_report.below_threshold:
                    q = min(q, 10.0)
                    flags.append("low_variance")
            if r.duplicate_report is not None:
                if col in r.duplicate_report.features_to_drop:
                    q = min(q, 0.0)
                    flags.append("duplicate")
            if r.missing_report is not None:
                if col in r.missing_report.flagged_infinite:
                    q = min(q, 50.0)
                    flags.append("has_infinite")

            # ── Leakage flags ─────────────────────────────────────────────────
            if r.leakage_report is not None:
                if col in r.leakage_report.flagged_features:
                    ltype = r.leakage_report.leakage_types.get(col, "leakage")
                    flags.append(f"leakage:{ltype}")

            # ── Drift flags ───────────────────────────────────────────────────
            if r.drift_report is not None:
                if col in r.drift_report.drifted_features:
                    flags.append("drift")

            # ── VIF flags ─────────────────────────────────────────────────────
            if r.vif_report is not None:
                if col in r.vif_report.high_vif_features:
                    flags.append("high_vif")

            # ── Stability flags ───────────────────────────────────────────────
            if r.stability_report is not None:
                if col in r.stability_report.unstable_features:
                    flags.append("unstable")

            imp = float(imp_norm.get(col, 0.0))
            stab = float(stab_norm.get(col, 50.0))
            leak = float(leakage_score.get(col, 100.0))
            drft = float(drift_score.get(col, 100.0))
            interp = float(shap_norm.get(col, 0.0))

            composite = (
                0.20 * q
                + 0.30 * imp
                + 0.20 * stab
                + 0.15 * leak
                + 0.15 * drft
            )

            scores[col] = FeatureScore(
                name=col,
                quality_score=round(q, 2),
                importance_score=round(imp, 2),
                stability_score=round(stab, 2),
                leakage_score=round(leak, 2),
                drift_score=round(drft, 2),
                interpretability_score=round(interp, 2),
                composite_score=round(composite, 2),
                flags=flags,
            )

        # ── Assign ranks ──────────────────────────────────────────────────────
        ranked = sorted(
            scores.items(), key=lambda x: x[1].composite_score, reverse=True
        )
        for rank, (name, fs) in enumerate(ranked, start=1):
            fs.rank = rank

        return scores

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _combined_importance(
        self,
        r:    FeatureQualityResults,
        cols: list[str],
    ) -> pd.Series:
        """Average all available importance signals into one series."""
        components: list[pd.Series] = []

        if r.importance_report is not None:
            s = r.importance_report.combined_importance.reindex(cols).fillna(0.0)
            components.append(s)

        if r.mi_report is not None:
            s = r.mi_report.mi_scores_norm.reindex(cols).fillna(0.0)
            components.append(s)

        if r.perm_report is not None:
            s = r.perm_report.mean_importance.reindex(cols).fillna(0.0)
            s = _normalise_to_1(s)
            components.append(s)

        if r.shap_report is not None and r.shap_report.available:
            s = r.shap_report.mean_abs_shap.reindex(cols).fillna(0.0)
            s = _normalise_to_1(s)
            components.append(s)

        if not components:
            return pd.Series(0.0, index=cols)

        avg = pd.concat(components, axis=1).mean(axis=1)
        return avg.reindex(cols).fillna(0.0)


# ── Utilities ─────────────────────────────────────────────────────────────────


def _normalise_to_1(s: pd.Series) -> pd.Series:
    mx = s.max()
    return s / mx if mx > 0 else s.copy()


def _normalise_to_100(s: pd.Series) -> pd.Series:
    return _normalise_to_1(s) * 100.0
