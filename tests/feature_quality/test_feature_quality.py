"""
Comprehensive tests for the Feature Quality Analysis framework.

Test data
---------
All tests use synthetic DataFrames (n=400 rows) built by _make_df().
The dataset contains deliberate problems:

  - constant_feat:     exact constant column
  - near_constant:     near-zero variance
  - h1_rsi_dup:        exact duplicate of h1_rsi
  - h1_high_corr:      highly correlated (0.99) with h1_rsi
  - h1_missing:        20 % NaN
  - future_label:      future-prefix (should be flagged as leaky)
  - label_direction:   binary classification target
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ── Modules under test ────────────────────────────────────────────────────────
from src.feature_quality import (
    BorutaReport,
    BorutaSelector,
    ClusterReport,
    ConstantFeatureDetector,
    ConstantReport,
    CorrelationAnalyzer,
    CorrelationReport,
    DriftDetector,
    DriftReport,
    DuplicateFeatureDetector,
    DuplicateReport,
    FeatureClusterer,
    FeatureQualityAnalyzer,
    FeatureQualityResults,
    FeatureReportGenerator,
    FeatureScore,
    FeatureSelector,
    FeatureQualityPipeline,
    ImportanceReport,
    LeakageDetector,
    LeakageReport,
    MIReport,
    MissingValueAnalyzer,
    MissingValueReport,
    MutualInformationAnalyzer,
    PermImportanceReport,
    PermutationImportanceAnalyzer,
    PSICalculator,
    PSIReport,
    RFEReport,
    RFESelector,
    SelectionResult,
    SHAPAnalyzer,
    SHAPReport,
    StabilityAnalyzer,
    StabilityReport,
    TreeImportanceAnalyzer,
    VIFAnalyzer,
    VIFReport,
    VarianceFilter,
    VarianceReport,
    compute_psi,
)

# ── Helpers ───────────────────────────────────────────────────────────────────

N = 400


def _make_df(n: int = N) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    idx = pd.date_range("2022-01-01", periods=n, freq="1h", tz="UTC")

    df   = pd.DataFrame(index=idx)

    # Normal features (valid prefixes) — independent random arrays
    df["h1_rsi"]       = rng.random(n) * 100
    df["m15_ema"]      = rng.standard_normal(n)
    df["daily_trend"]  = rng.standard_normal(n)
    df["tech_macd"]    = rng.standard_normal(n)
    df["vol_atr"]      = rng.exponential(0.5, n)

    # Deliberate problems
    df["constant_feat"]  = 7.0
    df["near_constant"]  = 7.0 + rng.uniform(0, 1e-7, n)
    df["h1_rsi_dup"]     = df["h1_rsi"].copy()
    df["h1_high_corr"]   = df["h1_rsi"] + rng.standard_normal(n) * 0.1

    missing = df["m15_ema"].copy()
    missing.iloc[:80] = np.nan              # 20 % missing
    df["h1_missing"] = missing

    # Future-prefix with its OWN independent base (so h1_rsi is NOT correlated)
    future_base = rng.random(n)
    df["future_return"]  = future_base   # future-prefix → leakage by prefix alone

    # Target (independent)
    df["label_direction"] = (rng.standard_normal(n) > 0).astype(float)

    return df


def _feature_df(df: pd.DataFrame) -> pd.DataFrame:
    """Drop target/label columns."""
    return df.drop(columns=["label_direction"])


def _target(df: pd.DataFrame) -> pd.Series:
    return df["label_direction"]


# =============================================================================
# 1. Missing Values
# =============================================================================


class TestMissingValues:
    def test_detects_nan(self):
        df = _make_df()
        r  = MissingValueAnalyzer().fit(df)
        assert r.missing_rates["h1_missing"] == pytest.approx(0.20, abs=0.01)

    def test_flags_high_missing(self):
        df = _make_df()
        r  = MissingValueAnalyzer(missing_threshold=0.10).fit(df)
        assert "h1_missing" in r.flagged_missing

    def test_clean_column_not_flagged(self):
        df = _make_df()
        r  = MissingValueAnalyzer().fit(df)
        assert "h1_rsi" not in r.flagged_missing

    def test_detects_infinite(self):
        df = _make_df()
        df["h1_rsi"].iloc[0] = np.inf
        r  = MissingValueAnalyzer().fit(df)
        assert "h1_rsi" in r.flagged_infinite

    def test_quality_scores_range(self):
        df = _make_df()
        r  = MissingValueAnalyzer().fit(df)
        scores = MissingValueAnalyzer().quality_scores(r)
        assert (scores >= 0).all() and (scores <= 100).all()


# =============================================================================
# 2. Duplicate Features
# =============================================================================


class TestDuplicateFeatures:
    def test_detects_exact_duplicate(self):
        df = _feature_df(_make_df())
        r  = DuplicateFeatureDetector().fit(df)
        assert "h1_rsi_dup" in r.features_to_drop

    def test_original_not_dropped(self):
        df = _feature_df(_make_df())
        r  = DuplicateFeatureDetector().fit(df)
        assert "h1_rsi" not in r.features_to_drop

    def test_pair_recorded(self):
        df = _feature_df(_make_df())
        r  = DuplicateFeatureDetector().fit(df)
        pairs = [(a, b) for a, b in r.duplicate_pairs]
        assert any("h1_rsi" in p for p in pairs)

    def test_no_false_positives(self):
        df = _feature_df(_make_df())
        r  = DuplicateFeatureDetector().fit(df)
        assert "m15_ema" not in r.features_to_drop
        assert "daily_trend" not in r.features_to_drop


# =============================================================================
# 3. Constant Features
# =============================================================================


class TestConstantFeatures:
    def test_detects_constant(self):
        df = _feature_df(_make_df())
        r  = ConstantFeatureDetector().fit(df)
        assert "constant_feat" in r.constant_features

    def test_detects_near_constant(self):
        # near_constant has std < 1e-5 → caught by VarianceFilter
        df = _feature_df(_make_df())
        vr = VarianceFilter(threshold=1e-5).fit(df)
        assert "near_constant" in vr.below_threshold or \
               "near_constant" in ConstantFeatureDetector().fit(df).constant_features

    def test_normal_feature_not_flagged(self):
        df = _feature_df(_make_df())
        r  = ConstantFeatureDetector().fit(df)
        assert "h1_rsi" not in r.constant_features
        assert "h1_rsi" not in r.near_constant_features


# =============================================================================
# 4. Variance Filter
# =============================================================================


class TestVarianceFilter:
    def test_flags_low_variance(self):
        df = _feature_df(_make_df())
        r  = VarianceFilter(threshold=1e-5).fit(df)
        assert "constant_feat" in r.below_threshold or \
               "near_constant" in r.below_threshold

    def test_normal_feature_has_variance(self):
        df = _feature_df(_make_df())
        r  = VarianceFilter().fit(df)
        assert float(r.variances["h1_rsi"]) > 1e-5


# =============================================================================
# 5. PSI
# =============================================================================


class TestPSI:
    def test_compute_psi_identical(self):
        x = np.random.default_rng(0).random(500)
        assert compute_psi(x, x) == pytest.approx(0.0, abs=0.05)

    def test_compute_psi_different(self):
        rng = np.random.default_rng(0)
        ref = rng.normal(0, 1, 500)
        cur = rng.normal(5, 1, 500)   # big shift
        assert compute_psi(ref, cur) > 0.20

    def test_fit_returns_all_features(self):
        df = _feature_df(_make_df())
        r  = PSICalculator().fit(df)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        assert set(r.psi_scores.index) == set(numeric_cols)

    def test_labels(self):
        df = _feature_df(_make_df())
        r  = PSICalculator().fit(df)
        assert r.psi_labels.isin(["stable", "minor_shift", "major_shift"]).all()


# =============================================================================
# 6. Correlation
# =============================================================================


class TestCorrelation:
    def test_flags_high_correlation(self):
        df = _feature_df(_make_df())
        r  = CorrelationAnalyzer(threshold=0.90).fit(df)
        flagged = {p["feat_a"] for p in r.high_corr_pairs} | \
                  {p["feat_b"] for p in r.high_corr_pairs}
        assert "h1_rsi" in flagged or "h1_high_corr" in flagged

    def test_features_to_drop(self):
        df = _feature_df(_make_df())
        r  = CorrelationAnalyzer(threshold=0.90).fit(df)
        assert len(r.features_to_drop) >= 1

    def test_pearson_is_symmetric(self):
        df = _feature_df(_make_df())
        r  = CorrelationAnalyzer().fit(df)
        assert r.pearson.shape[0] == r.pearson.shape[1]
        pd.testing.assert_frame_equal(r.pearson, r.pearson.T)

    def test_distance_correlation(self):
        rng = np.random.default_rng(0)
        x   = rng.random(100)
        y   = x + rng.random(100) * 0.1
        dc  = CorrelationAnalyzer.distance_correlation(x, y)
        assert 0.5 < dc <= 1.0

    def test_cluster_groups(self):
        df = _feature_df(_make_df())
        r  = CorrelationAnalyzer(threshold=0.90).fit(df)
        # h1_rsi + h1_rsi_dup + h1_high_corr should form a cluster
        assert len(r.cluster_groups) >= 1


# =============================================================================
# 7. VIF
# =============================================================================


class TestVIF:
    def test_returns_vif_per_feature(self):
        df = _feature_df(_make_df()).select_dtypes(include=[np.number]).dropna()
        r  = VIFAnalyzer().fit(df)
        assert len(r.vif_scores) > 0

    def test_vif_values_positive(self):
        df = _feature_df(_make_df()).select_dtypes(include=[np.number]).dropna()
        r  = VIFAnalyzer().fit(df)
        assert (r.vif_scores >= 0).all()

    def test_high_vif_flagged(self):
        rng = np.random.default_rng(42)
        n   = 200
        a   = rng.random(n)
        df  = pd.DataFrame({"h1_a": a, "h1_b": a + rng.random(n) * 0.01, "h1_c": rng.random(n)})
        r   = VIFAnalyzer(threshold=5.0).fit(df)
        assert len(r.high_vif_features) >= 1


# =============================================================================
# 8. Drift Detection
# =============================================================================


class TestDrift:
    def test_stable_series_not_flagged(self):
        rng = np.random.default_rng(0)
        x   = rng.standard_normal(400)
        df  = pd.DataFrame({f"h1_f{i}": x + rng.standard_normal(400) * 0.05 for i in range(5)})
        # Use strict alpha so minor random variation doesn't flag everything
        r   = DriftDetector(split_ratio=0.5, ks_alpha=0.001, psi_threshold=0.30, js_threshold=0.25).fit(df)
        n_drifted = len(r.drifted_features)
        assert n_drifted < 5

    def test_shifted_distribution_flagged(self):
        rng = np.random.default_rng(0)
        n   = 400
        ref = rng.normal(0, 1, n // 2)
        cur = rng.normal(10, 1, n // 2)  # huge shift
        x   = np.concatenate([ref, cur])
        df  = pd.DataFrame({"h1_feature": x})
        r   = DriftDetector(split_ratio=0.5).fit(df)
        assert "h1_feature" in r.drifted_features

    def test_ks_statistics_range(self):
        df  = _feature_df(_make_df()).select_dtypes(include=[np.number]).dropna()
        r   = DriftDetector().fit(df)
        valid = r.ks_statistics.dropna()
        assert (valid >= 0).all() and (valid <= 1).all()


# =============================================================================
# 9. Leakage Detection
# =============================================================================


class TestLeakage:
    def test_future_prefix_flagged(self):
        df = _feature_df(_make_df())
        r  = LeakageDetector().fit(df)
        assert "future_return" in r.flagged_features

    def test_clean_feature_not_flagged(self):
        df = _feature_df(_make_df())
        r  = LeakageDetector().fit(df)
        assert "h1_rsi" not in r.flagged_features
        assert "m15_ema" not in r.flagged_features

    def test_leakage_score_future_prefix(self):
        df = _feature_df(_make_df())
        r  = LeakageDetector().fit(df)
        assert r.leakage_scores["future_return"] == 1.0

    def test_target_leakage_detected(self):
        rng    = np.random.default_rng(0)
        n      = 300
        target = pd.Series((rng.standard_normal(n) > 0).astype(float))
        leak   = target + rng.standard_normal(n) * 0.01   # 99%+ correlated with target
        df     = pd.DataFrame({"h1_a": rng.standard_normal(n), "h1_leak": leak})
        r      = LeakageDetector(correlation_threshold=0.80).fit(df, target)
        assert "h1_leak" in r.flagged_features


# =============================================================================
# 10. Feature Importance
# =============================================================================


class TestFeatureImportance:
    def test_returns_all_features(self):
        df  = _feature_df(_make_df()).select_dtypes(include=[np.number]).fillna(0)
        tgt = _target(_make_df())
        r   = TreeImportanceAnalyzer(max_samples=200, n_estimators=10).fit(df, tgt)
        assert set(r.combined_importance.index) == set(df.columns)

    def test_non_negative_importance(self):
        df  = _feature_df(_make_df()).select_dtypes(include=[np.number]).fillna(0)
        tgt = _target(_make_df())
        r   = TreeImportanceAnalyzer(max_samples=200, n_estimators=10).fit(df, tgt)
        assert (r.combined_importance >= 0).all()

    def test_rf_always_available(self):
        df  = _feature_df(_make_df()).select_dtypes(include=[np.number]).fillna(0)
        tgt = _target(_make_df())
        r   = TreeImportanceAnalyzer(max_samples=200, n_estimators=5).fit(df, tgt)
        assert r.rf_importance is not None
        assert len(r.rf_importance) > 0


# =============================================================================
# 11. Mutual Information
# =============================================================================


class TestMutualInformation:
    def test_returns_scores(self):
        df  = _feature_df(_make_df()).select_dtypes(include=[np.number]).fillna(0)
        tgt = _target(_make_df())
        r   = MutualInformationAnalyzer(max_samples=200).fit(df, tgt)
        assert len(r.mi_scores) == df.shape[1]

    def test_scores_non_negative(self):
        df  = _feature_df(_make_df()).select_dtypes(include=[np.number]).fillna(0)
        tgt = _target(_make_df())
        r   = MutualInformationAnalyzer(max_samples=200).fit(df, tgt)
        assert (r.mi_scores >= 0).all()

    def test_norm_scores_range(self):
        df  = _feature_df(_make_df()).select_dtypes(include=[np.number]).fillna(0)
        tgt = _target(_make_df())
        r   = MutualInformationAnalyzer(max_samples=200).fit(df, tgt)
        assert (r.mi_scores_norm >= 0).all()
        assert r.mi_scores_norm.max() <= 1.0 + 1e-6


# =============================================================================
# 12. Permutation Importance
# =============================================================================


class TestPermutationImportance:
    def test_returns_all_features(self):
        df  = _feature_df(_make_df()).select_dtypes(include=[np.number]).fillna(0)
        tgt = _target(_make_df())
        r   = PermutationImportanceAnalyzer(n_repeats=3, max_samples=200).fit(df, tgt)
        assert set(r.mean_importance.index) == set(df.columns)

    def test_baseline_score_reasonable(self):
        df  = _feature_df(_make_df()).select_dtypes(include=[np.number]).fillna(0)
        tgt = _target(_make_df())
        r   = PermutationImportanceAnalyzer(n_repeats=3, max_samples=200).fit(df, tgt)
        # AUC should be in [0, 1]
        assert 0.0 <= r.baseline_score <= 1.0


# =============================================================================
# 13. SHAP Analysis
# =============================================================================


class TestSHAP:
    def test_returns_report(self):
        df  = _feature_df(_make_df()).select_dtypes(include=[np.number]).fillna(0)
        tgt = _target(_make_df())
        r   = SHAPAnalyzer(max_samples=200).fit(df, tgt)
        assert isinstance(r, SHAPReport)

    def test_mean_abs_shap_non_negative(self):
        df  = _feature_df(_make_df()).select_dtypes(include=[np.number]).fillna(0)
        tgt = _target(_make_df())
        r   = SHAPAnalyzer(max_samples=200).fit(df, tgt)
        if r.available:
            assert (r.mean_abs_shap >= 0).all()

    def test_save_summary_parquet(self, tmp_path):
        df  = _feature_df(_make_df()).select_dtypes(include=[np.number]).fillna(0)
        tgt = _target(_make_df())
        r   = SHAPAnalyzer(max_samples=200).fit(df, tgt)
        if r.shap_values is not None:
            path = tmp_path / "shap.parquet"
            SHAPAnalyzer().save_summary_parquet(r, path)
            assert path.exists()


# =============================================================================
# 14. Boruta
# =============================================================================


class TestBoruta:
    def test_returns_report(self):
        df  = _feature_df(_make_df()).select_dtypes(include=[np.number]).fillna(0)
        tgt = _target(_make_df())
        r   = BorutaSelector(n_estimators=10, max_iter=10, max_samples=200).fit(df, tgt)
        assert isinstance(r, BorutaReport)

    def test_all_features_classified(self):
        df  = _feature_df(_make_df()).select_dtypes(include=[np.number]).fillna(0)
        tgt = _target(_make_df())
        r   = BorutaSelector(n_estimators=10, max_iter=10, max_samples=200).fit(df, tgt)
        classified = set(r.accepted) | set(r.rejected) | set(r.tentative)
        assert classified == set(df.columns)

    def test_hit_counts_non_negative(self):
        df  = _feature_df(_make_df()).select_dtypes(include=[np.number]).fillna(0)
        tgt = _target(_make_df())
        r   = BorutaSelector(n_estimators=10, max_iter=10, max_samples=200).fit(df, tgt)
        assert (r.hit_counts >= 0).all()


# =============================================================================
# 15. Recursive Feature Elimination
# =============================================================================


class TestRFE:
    def test_returns_selected_features(self):
        df  = _feature_df(_make_df()).select_dtypes(include=[np.number]).fillna(0)
        tgt = _target(_make_df())
        r   = RFESelector(n_features_to_select=3, max_samples=200).fit(df, tgt)
        assert len(r.selected_features) == 3

    def test_ranking_contains_all_features(self):
        df  = _feature_df(_make_df()).select_dtypes(include=[np.number]).fillna(0)
        tgt = _target(_make_df())
        r   = RFESelector(n_features_to_select=3, max_samples=200).fit(df, tgt)
        assert set(r.feature_ranking.index) == set(df.columns)

    def test_selected_have_rank_1(self):
        df  = _feature_df(_make_df()).select_dtypes(include=[np.number]).fillna(0)
        tgt = _target(_make_df())
        r   = RFESelector(n_features_to_select=4, max_samples=200).fit(df, tgt)
        for f in r.selected_features:
            assert r.feature_ranking[f] == 1


# =============================================================================
# 16. Stability Analysis
# =============================================================================


class TestStability:
    def test_stability_scores_range(self):
        df  = _feature_df(_make_df()).select_dtypes(include=[np.number]).fillna(0)
        tgt = _target(_make_df())
        r   = StabilityAnalyzer(
            n_windows=5,
            window_frac=0.40,
            max_samples_per_window=200,
        ).fit(df, tgt)
        assert (r.stability_scores >= 0).all()
        assert (r.stability_scores <= 100).all()

    def test_rolling_importance_shape(self):
        df  = _feature_df(_make_df()).select_dtypes(include=[np.number]).fillna(0)
        tgt = _target(_make_df())
        r   = StabilityAnalyzer(
            n_windows=5,
            window_frac=0.40,
            max_samples_per_window=200,
        ).fit(df, tgt)
        assert r.rolling_importance.shape[1] == df.shape[1]
        assert r.n_windows <= 5


# =============================================================================
# 17. Feature Clustering
# =============================================================================


class TestFeatureClustering:
    def test_returns_cluster_labels(self):
        df  = _feature_df(_make_df()).select_dtypes(include=[np.number]).fillna(0)
        r   = FeatureClusterer(correlation_threshold=0.70).fit(df)
        assert len(r.cluster_labels) == df.shape[1]

    def test_representative_in_cluster(self):
        df  = _feature_df(_make_df()).select_dtypes(include=[np.number]).fillna(0)
        r   = FeatureClusterer().fit(df)
        for cid, rep in r.cluster_representatives.items():
            assert rep in r.clusters[cid]

    def test_high_corr_features_in_same_cluster(self):
        df  = _feature_df(_make_df()).select_dtypes(include=[np.number]).fillna(0)
        r   = FeatureClusterer(correlation_threshold=0.90).fit(df)
        # h1_rsi + h1_rsi_dup should be in the same cluster
        labels = r.cluster_labels
        if "h1_rsi" in labels.index and "h1_rsi_dup" in labels.index:
            assert labels["h1_rsi"] == labels["h1_rsi_dup"]


# =============================================================================
# 18. Feature Selector
# =============================================================================


class TestFeatureSelector:
    def test_union_strategy(self):
        scores = pd.Series({"a": 80.0, "b": 60.0, "c": 40.0})
        sel    = FeatureSelector(strategy="union")
        r      = sel.select(scores, {"m1": ["a", "b"], "m2": ["b", "c"]})
        assert set(r.selected_features) == {"a", "b", "c"}

    def test_intersection_strategy(self):
        scores = pd.Series({"a": 80.0, "b": 60.0, "c": 40.0})
        sel    = FeatureSelector(strategy="intersection")
        r      = sel.select(scores, {"m1": ["a", "b"], "m2": ["b", "c"]})
        assert r.selected_features == ["b"]

    def test_voting_strategy(self):
        scores = pd.Series({"a": 80.0, "b": 60.0, "c": 40.0})
        sel    = FeatureSelector(strategy="voting", min_votes=2)
        r      = sel.select(scores, {"m1": ["a", "b"], "m2": ["b", "c"], "m3": ["a", "c"]})
        assert "b" in r.selected_features
        assert "a" in r.selected_features

    def test_top_n_dict_populated(self):
        scores = pd.Series({f"f{i}": float(100 - i) for i in range(60)})
        sel    = FeatureSelector(top_n_counts=[10, 25])
        r      = sel.select(scores)
        assert 10 in r.top_n
        assert len(r.top_n[10]) == 10

    def test_exclude_features(self):
        scores = pd.Series({"a": 90.0, "b": 80.0, "c": 70.0})
        sel    = FeatureSelector(strategy="union")
        r      = sel.select(scores, {"m1": ["a", "b"]}, exclude_features=["a"])
        assert "a" not in r.selected_features


# =============================================================================
# 19. Feature Reports
# =============================================================================


class TestFeatureReports:
    @pytest.fixture
    def minimal_results(self):
        """Build a minimal FeatureQualityResults for report generation."""
        r = FeatureQualityResults(symbol="EURUSD", pipeline_version="1.0.0")
        scores = {}
        for i, name in enumerate(["h1_rsi", "m15_ema", "constant_feat"]):
            scores[name] = FeatureScore(
                name=name,
                quality_score=max(0.0, 90.0 - i * 30),
                importance_score=80.0 - i * 20,
                stability_score=70.0,
                leakage_score=100.0,
                drift_score=95.0,
                interpretability_score=60.0,
                composite_score=max(0.0, 85.0 - i * 25),
                rank=i + 1,
                flags=["constant"] if name == "constant_feat" else [],
            )
        r.feature_scores = scores

        # Minimal selection result
        from src.feature_quality.feature_selector import SelectionResult
        r.selection_result = SelectionResult(
            selected_features=["h1_rsi", "m15_ema"],
            method_votes={},
            vote_counts=pd.Series({"h1_rsi": 2, "m15_ema": 1, "constant_feat": 0}),
            strategy="voting",
            top_n={25: ["h1_rsi", "m15_ema"]},
        )
        return r

    def test_quality_report_md(self, minimal_results, tmp_path):
        gen = FeatureReportGenerator(tmp_path)
        gen.generate_all(minimal_results)
        assert (tmp_path / "feature_quality_report.md").exists()
        content = (tmp_path / "feature_quality_report.md").read_text()
        assert "EURUSD" in content

    def test_importance_csv(self, minimal_results, tmp_path):
        gen = FeatureReportGenerator(tmp_path)
        gen.generate_all(minimal_results)
        df = pd.read_csv(tmp_path / "feature_importance.csv")
        assert "feature" in df.columns
        assert len(df) == 3

    def test_rankings_csv(self, minimal_results, tmp_path):
        gen = FeatureReportGenerator(tmp_path)
        gen.generate_all(minimal_results)
        df = pd.read_csv(tmp_path / "feature_rankings.csv")
        assert df.iloc[0]["rank"] == 1

    def test_selected_features_json(self, minimal_results, tmp_path):
        gen = FeatureReportGenerator(tmp_path)
        gen.generate_all(minimal_results)
        p = tmp_path / "selected_features_top25.json"
        assert p.exists()
        data = json.loads(p.read_text())
        assert "features" in data

    def test_selected_all_json(self, minimal_results, tmp_path):
        gen = FeatureReportGenerator(tmp_path)
        gen.generate_all(minimal_results)
        p = tmp_path / "selected_features_all.json"
        assert p.exists()
        data = json.loads(p.read_text())
        assert set(data["features"]) == {"h1_rsi", "m15_ema"}

    def test_leakage_report_md(self, minimal_results, tmp_path):
        gen = FeatureReportGenerator(tmp_path)
        gen.generate_all(minimal_results)
        assert (tmp_path / "leakage_report.md").exists()


# =============================================================================
# 20. FeatureQualityAnalyzer — integration (no target)
# =============================================================================


class TestFeatureQualityAnalyzerNoTarget:
    def test_runs_without_target(self):
        df  = _feature_df(_make_df())
        cfg = {
            "skip_boruta": True, "skip_rfe": True,
            "skip_shap": True,   "skip_permutation": True,
            "skip_stability": True, "skip_vif": True,
        }
        results = FeatureQualityAnalyzer(config=cfg).run(df, "TEST")
        assert isinstance(results, FeatureQualityResults)

    def test_all_features_scored(self):
        df  = _feature_df(_make_df())
        cfg = {
            "skip_boruta": True, "skip_rfe": True,
            "skip_shap": True,   "skip_permutation": True,
            "skip_stability": True,
        }
        results = FeatureQualityAnalyzer(config=cfg).run(df, "TEST")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        assert set(results.feature_scores) == set(numeric_cols)

    def test_composite_scores_range(self):
        df  = _feature_df(_make_df())
        cfg = {
            "skip_boruta": True, "skip_rfe": True,
            "skip_shap": True,   "skip_permutation": True,
            "skip_stability": True,
        }
        results = FeatureQualityAnalyzer(config=cfg).run(df, "TEST")
        for fs in results.feature_scores.values():
            assert 0.0 <= fs.composite_score <= 100.0

    def test_constant_feature_flagged(self):
        df  = _feature_df(_make_df())
        cfg = {
            "skip_boruta": True, "skip_rfe": True,
            "skip_shap": True,   "skip_permutation": True,
            "skip_stability": True,
        }
        results = FeatureQualityAnalyzer(config=cfg).run(df, "TEST")
        if "constant_feat" in results.feature_scores:
            fs = results.feature_scores["constant_feat"]
            assert "constant" in fs.flags

    def test_ranks_assigned(self):
        df  = _feature_df(_make_df())
        cfg = {
            "skip_boruta": True, "skip_rfe": True,
            "skip_shap": True,   "skip_permutation": True,
            "skip_stability": True,
        }
        results = FeatureQualityAnalyzer(config=cfg).run(df, "TEST")
        ranks = [fs.rank for fs in results.feature_scores.values()]
        assert sorted(ranks) == list(range(1, len(ranks) + 1))


# =============================================================================
# 21. FeatureQualityAnalyzer — integration (with target)
# =============================================================================


class TestFeatureQualityAnalyzerWithTarget:
    def test_runs_with_target_col(self):
        df  = _make_df()
        cfg = {
            "skip_boruta": True, "skip_rfe": True,
            "skip_shap": True,   "skip_permutation": True,
            "skip_stability": True, "skip_vif": True,
        }
        results = FeatureQualityAnalyzer(config=cfg).run(
            df, "TEST", target_col="label_direction"
        )
        assert results.importance_report is not None
        assert results.mi_report is not None

    def test_leakage_detected_with_target(self):
        df  = _make_df()
        cfg = {
            "skip_boruta": True, "skip_rfe": True,
            "skip_shap": True,   "skip_permutation": True,
            "skip_stability": True, "skip_vif": True,
        }
        results = FeatureQualityAnalyzer(config=cfg).run(
            df, "TEST", target_col="label_direction"
        )
        assert results.leakage_report is not None
        assert "future_return" in results.leakage_report.flagged_features

    def test_selection_result_populated(self):
        df  = _make_df()
        cfg = {
            "skip_boruta": True, "skip_rfe": True,
            "skip_shap": True,   "skip_permutation": True,
            "skip_stability": True, "skip_vif": True,
            "feature_selection_strategy": "union",
        }
        results = FeatureQualityAnalyzer(config=cfg).run(
            df, "TEST", target_col="label_direction"
        )
        assert results.selection_result is not None
        assert len(results.selection_result.selected_features) > 0


# =============================================================================
# 22. FeatureQualityPipeline
# =============================================================================


class TestFeatureQualityPipeline:
    def test_pipeline_run(self, tmp_path):
        df  = _make_df()
        cfg = {
            "skip_boruta": True, "skip_rfe": True,
            "skip_shap": True,   "skip_permutation": True,
            "skip_stability": True, "skip_vif": True,
        }
        pipeline = FeatureQualityPipeline(
            output_dir=tmp_path,
            config=cfg,
        )
        results = pipeline.run(
            df, "TEST", target_col="label_direction", write_reports=True
        )
        assert isinstance(results, FeatureQualityResults)
        assert (tmp_path / "feature_quality_report.md").exists()
        assert (tmp_path / "feature_importance.csv").exists()
        assert (tmp_path / "feature_rankings.csv").exists()

    def test_pipeline_no_reports(self, tmp_path):
        df  = _make_df()
        cfg = {
            "skip_boruta": True, "skip_rfe": True,
            "skip_shap": True,   "skip_permutation": True,
            "skip_stability": True, "skip_vif": True,
        }
        pipeline = FeatureQualityPipeline(output_dir=tmp_path, config=cfg)
        results  = pipeline.run(df, "TEST", write_reports=False)
        assert isinstance(results, FeatureQualityResults)
        # Reports should NOT exist
        assert not (tmp_path / "feature_quality_report.md").exists()

    def test_pipeline_without_feature_store_run_for_symbol_raises(self):
        pipeline = FeatureQualityPipeline()
        with pytest.raises(RuntimeError):
            pipeline.run_for_symbol("EURUSD")


# =============================================================================
# 23. Read-only guarantee
# =============================================================================


class TestReadOnly:
    def test_original_df_not_modified(self):
        df   = _make_df()
        orig = df.copy()
        cfg  = {
            "skip_boruta": True, "skip_rfe": True,
            "skip_shap": True,   "skip_permutation": True,
            "skip_stability": True, "skip_vif": True,
        }
        FeatureQualityAnalyzer(config=cfg).run(
            df, "TEST", target_col="label_direction"
        )
        pd.testing.assert_frame_equal(df, orig)


# =============================================================================
# 24. Performance
# =============================================================================


class TestPerformance:
    def test_quality_modules_fast(self):
        """Core quality modules (no ML) should complete in < 10 s on 5 k rows."""
        import time
        rng = np.random.default_rng(0)
        n   = 5_000
        idx = pd.date_range("2020-01-01", periods=n, freq="15min", tz="UTC")
        df  = pd.DataFrame(
            rng.random((n, 50)),
            index=idx,
            columns=[f"h1_feat_{i:03d}" for i in range(50)],
        )
        cfg = {
            "skip_boruta": True, "skip_rfe": True,
            "skip_shap": True,   "skip_permutation": True,
            "skip_stability": True, "skip_vif": True,
            "skip_correlation": True,
        }
        t0      = time.perf_counter()
        results = FeatureQualityAnalyzer(config=cfg).run(df, "PERF")
        elapsed = time.perf_counter() - t0
        assert elapsed < 10.0, f"Core quality modules took {elapsed:.2f}s"
        assert len(results.feature_scores) == 50
