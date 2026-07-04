"""
Tests for src/labels — Label Generation Engine.

Synthetic OHLCV data (300 bars) with controlled trend phases is used.
All model-heavy parameters are reduced (max_bars=10, atr_period=5) for speed.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ── Shared fixture ────────────────────────────────────────────────────────────

def _make_ohlcv(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """Synthetic hourly OHLCV with uptrend → flat → downtrend."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-01", periods=n, freq="h")
    third = n // 3

    log_rets = np.concatenate([
        rng.normal(+4e-4, 1e-3, third),          # up
        rng.normal(0.0,   1e-3, third),           # flat
        rng.normal(-4e-4, 1e-3, n - 2 * third),  # down
    ])
    close = 1.1000 * np.exp(np.cumsum(log_rets))

    bar_range = rng.uniform(2e-4, 3e-3, n)
    open_  = close * np.exp(rng.normal(0, 5e-4, n))
    high   = np.maximum(close, open_) * np.exp(bar_range * 0.7)
    low    = np.minimum(close, open_) * np.exp(-bar_range * 0.3)
    volume = rng.uniform(1_000, 10_000, n)

    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


DF = _make_ohlcv()   # module-level fixture (fast)


# ── Module 1: Market Bias ─────────────────────────────────────────────────────

class TestMarketBias:
    from src.labels.market_bias import MarketBiasLabeler, MarketBiasConfig

    def _labeler(self):
        from src.labels.market_bias import MarketBiasLabeler, MarketBiasConfig
        return MarketBiasLabeler(MarketBiasConfig(horizons=[1, 3], rolling_vol_window=20))

    def test_output_rows_match_input(self):
        res = self._labeler().fit(DF)
        assert len(res.labels) == len(DF)

    def test_expected_columns_exist(self):
        res = self._labeler().fit(DF)
        for h in [1, 3]:
            for pfx in ["fwd_return_", "direction_", "bias_", "confidence_", "probability_"]:
                assert f"{pfx}{h}b" in res.labels.columns

    def test_direction_values(self):
        res = self._labeler().fit(DF)
        valid = res.labels["direction_1b"].dropna()
        assert set(valid.unique()).issubset({0.0, 1.0, 2.0})

    def test_binary_bias_values(self):
        res = self._labeler().fit(DF)
        valid = res.labels["bias_1b"].dropna()
        assert set(valid.unique()).issubset({0.0, 1.0})

    def test_nan_at_last_horizon_rows(self):
        res = self._labeler().fit(DF)
        # Last 3 rows (horizon=3) must be NaN for the 3-bar horizon
        assert res.labels["direction_3b"].iloc[-3:].isna().all()

    def test_probability_in_unit_interval(self):
        res = self._labeler().fit(DF)
        valid = res.labels["probability_1b"].dropna()
        assert (valid >= 0.0).all() and (valid <= 1.0).all()

    def test_confidence_in_unit_interval(self):
        res = self._labeler().fit(DF)
        valid = res.labels["confidence_1b"].dropna()
        assert (valid >= 0.0).all() and (valid <= 1.0).all()

    def test_horizon_stats_populated(self):
        res = self._labeler().fit(DF)
        for h in [1, 3]:
            s = res.horizon_stats[h]
            assert abs(s["bull_pct"] + s["bear_pct"] + s["neutral_pct"] - 1.0) < 1e-6

    def test_empty_df_raises(self):
        with pytest.raises(ValueError):
            self._labeler().fit(pd.DataFrame(columns=["close"]))

    def test_missing_price_col_raises(self):
        from src.labels.market_bias import MarketBiasLabeler, MarketBiasConfig
        with pytest.raises(ValueError):
            MarketBiasLabeler(MarketBiasConfig(price_col="nonexistent")).fit(DF)


# ── Module 2: Trade Outcome ───────────────────────────────────────────────────

class TestTradeOutcome:
    def _labeler(self):
        from src.labels.trade_outcome import TradeOutcomeLabeler, TradeOutcomeConfig
        return TradeOutcomeLabeler(TradeOutcomeConfig(atr_period=5, max_bars=10))

    def test_output_rows_match_input(self):
        res = self._labeler().fit(DF)
        assert len(res.labels) == len(DF)

    def test_outcome_values(self):
        res = self._labeler().fit(DF)
        for col in ("long_outcome", "short_outcome"):
            valid = res.labels[col].dropna()
            assert set(valid.unique()).issubset({0.0, 1.0, 2.0})

    def test_mfe_non_negative(self):
        res = self._labeler().fit(DF)
        assert (res.labels["long_mfe_pct"].dropna() >= 0).all()
        assert (res.labels["short_mfe_pct"].dropna() >= 0).all()

    def test_mae_non_negative(self):
        res = self._labeler().fit(DF)
        assert (res.labels["long_mae_pct"].dropna() >= 0).all()
        assert (res.labels["short_mae_pct"].dropna() >= 0).all()

    def test_nan_at_last_max_bars_rows(self):
        res = self._labeler().fit(DF)
        assert res.labels["long_outcome"].iloc[-10:].isna().all()

    def test_composite_columns_exist(self):
        res = self._labeler().fit(DF)
        for col in ("outcome", "mfe_pct", "mae_pct", "trade_duration_bars"):
            assert col in res.labels.columns

    def test_guaranteed_tp(self):
        """If high is very high for all future bars, TP must be hit first."""
        from src.labels.trade_outcome import simulate_trade, TP_FIRST
        f_high = np.full(20, 2.0)      # far above any TP
        f_low  = np.full(20, 1.001)    # just above entry → SL never hit
        entry  = 1.0
        outcome, dur, mfe, mae = simulate_trade(f_high, f_low, 1.01, 0.99, entry, 1)
        assert outcome == TP_FIRST
        assert dur == 1
        assert mfe >= 0

    def test_guaranteed_sl(self):
        """If low is very low for all future bars, SL must be hit first."""
        from src.labels.trade_outcome import simulate_trade, SL_FIRST
        f_high = np.full(20, 1.001)    # just above entry
        f_low  = np.full(20, 0.5)      # far below SL
        outcome, dur, mfe, mae = simulate_trade(f_high, f_low, 1.01, 0.99, 1.0, 1)
        assert outcome == SL_FIRST

    def test_empty_df_raises(self):
        with pytest.raises(ValueError):
            self._labeler().fit(pd.DataFrame(columns=["open", "high", "low", "close"]))

    def test_rates_sum_to_at_most_one(self):
        res = self._labeler().fit(DF)
        assert res.long_tp_rate + res.long_sl_rate <= 1.0 + 1e-9


# ── Module 3: Setup Quality ───────────────────────────────────────────────────

class TestSetupQuality:
    def _labeler(self):
        from src.labels.setup_quality import SetupQualityLabeler, SetupQualityConfig
        return SetupQualityLabeler(SetupQualityConfig(atr_period=5, max_bars=10))

    def test_output_rows_match_input(self):
        res = self._labeler().fit(DF)
        assert len(res.labels) == len(DF)

    def test_quality_values(self):
        valid = self._labeler().fit(DF).labels["setup_quality"].dropna()
        assert set(valid.unique()).issubset({0.0, 1.0, 2.0, 3.0})

    def test_score_range(self):
        valid = self._labeler().fit(DF).labels["setup_score"].dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_nan_at_last_rows(self):
        res = self._labeler().fit(DF)
        assert res.labels["setup_quality"].iloc[-10:].isna().all()

    def test_mfe_mae_ratio_positive(self):
        valid = self._labeler().fit(DF).labels["setup_mfe_mae_ratio"].dropna()
        assert (valid >= 0).all()

    def test_grade_percentages_sum_to_one(self):
        res = self._labeler().fit(DF)
        total = res.high_pct + res.medium_pct + res.low_pct + res.no_trade_pct
        assert abs(total - 1.0) < 1e-6


# ── Module 4: Entry Timing ────────────────────────────────────────────────────

class TestEntryTiming:
    def _labeler(self):
        from src.labels.entry_timing import EntryTimingLabeler, EntryTimingConfig
        return EntryTimingLabeler(EntryTimingConfig(atr_period=5, max_bars=10, window_size=5))

    def test_output_rows_match_input(self):
        res = self._labeler().fit(DF)
        assert len(res.labels) == len(DF)

    def test_signal_values(self):
        valid = self._labeler().fit(DF).labels["entry_signal"].dropna()
        assert set(valid.unique()).issubset({0.0, 1.0, 2.0})

    def test_optimal_entry_exists(self):
        # is_optimal_entry should be 1 for some rows
        is_opt = self._labeler().fit(DF).labels["is_optimal_entry"]
        assert int(is_opt.sum()) > 0

    def test_nan_at_last_rows(self):
        res = self._labeler().fit(DF)
        # lookback = max(max_bars=10, window_size=5) = 10
        assert res.labels["entry_signal"].iloc[-10:].isna().all()

    def test_signal_percentages_sum_to_one(self):
        res = self._labeler().fit(DF)
        total = res.enter_pct + res.wait_pct + res.ignore_pct
        assert abs(total - 1.0) < 1e-6


# ── Module 5: Trade Management ────────────────────────────────────────────────

class TestTradeManagement:
    def _labeler(self):
        from src.labels.trade_management import TradeManagementLabeler, TradeManagementConfig
        return TradeManagementLabeler(
            TradeManagementConfig(atr_period=5, max_bars=10)
        )

    def test_output_rows_match_input(self):
        res = self._labeler().fit(DF)
        assert len(res.labels) == len(DF)

    def test_strategy_values(self):
        valid = self._labeler().fit(DF).labels["mgmt_strategy"].dropna()
        assert set(valid.unique()).issubset({0.0, 1.0, 2.0, 3.0})

    def test_optimal_exit_positive(self):
        valid = self._labeler().fit(DF).labels["mgmt_optimal_exit_bar"].dropna()
        assert (valid >= 0).all()

    def test_max_r_finite(self):
        valid = self._labeler().fit(DF).labels["mgmt_max_r_multiple"].dropna()
        assert valid.notna().all()

    def test_nan_at_last_rows(self):
        res = self._labeler().fit(DF)
        assert res.labels["mgmt_strategy"].iloc[-10:].isna().all()


# ── Module 6: Label Validator ─────────────────────────────────────────────────

def _make_small_labels(n: int = 100) -> pd.DataFrame:
    from src.labels.market_bias import MarketBiasLabeler, MarketBiasConfig
    small_df = _make_ohlcv(n)
    return MarketBiasLabeler(MarketBiasConfig(horizons=[1])).fit(small_df).labels


class TestLabelValidator:
    def _validator(self):
        from src.labels.label_validator import LabelValidator, LabelValidatorConfig
        return LabelValidator(LabelValidatorConfig())

    def test_valid_labels_pass(self):
        lbl = _make_small_labels()
        rep = self._validator().validate(lbl)
        # No FAIL on valid data
        assert not rep.failures(), [str(i) for i in rep.failures()]

    def test_empty_labels_fail(self):
        empty = pd.DataFrame()
        rep = self._validator().validate(empty)
        assert not rep.passed

    def test_column_name_leakage_detected(self):
        lbl = _make_small_labels()
        features = lbl.copy()   # label cols in features → leakage
        rep = self._validator().validate(lbl, features)
        fail_checks = [i.check for i in rep.failures()]
        assert "column_leakage" in fail_checks

    def test_no_leakage_with_separate_features(self):
        lbl  = _make_small_labels()
        feat = pd.DataFrame({"some_feature": np.random.randn(len(lbl))}, index=lbl.index)
        rep  = self._validator().validate(lbl, feat)
        fail_checks = [i.check for i in rep.failures()]
        assert "column_leakage" not in fail_checks

    def test_index_mismatch_warning(self):
        lbl  = _make_small_labels()
        feat = pd.DataFrame({"x": np.zeros(50)})   # different index
        rep  = self._validator().validate(lbl, feat)
        sev  = [i.severity for i in rep.issues if i.check == "index_alignment"]
        assert "WARNING" in sev


# ── Module 7: Label Metadata ──────────────────────────────────────────────────

class TestLabelMetadata:
    def test_build_from_labels(self):
        from src.labels.label_metadata import LabelMeta
        lbl  = _make_small_labels()
        meta = LabelMeta.build(labels=lbl, symbol="TESTSYM", timeframe="H1")
        assert meta.symbol == "TESTSYM"
        assert meta.n_rows == len(lbl)
        assert len(meta.label_columns) == len(lbl.columns)

    def test_json_roundtrip(self):
        from src.labels.label_metadata import LabelMeta
        lbl  = _make_small_labels()
        meta = LabelMeta.build(labels=lbl, symbol="EURUSD")
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            p = Path(f.name)
        try:
            meta.to_json(p)
            meta2 = LabelMeta.from_json(p)
            assert meta2.symbol == meta.symbol
            assert meta2.n_rows  == meta.n_rows
        finally:
            p.unlink(missing_ok=True)

    def test_column_meta_dtypes(self):
        from src.labels.label_metadata import LabelMeta
        lbl  = _make_small_labels()
        meta = LabelMeta.build(labels=lbl, symbol="EURUSD")
        for cm in meta.column_meta:
            assert isinstance(cm.nan_rate, float)
            assert 0.0 <= cm.nan_rate <= 1.0


# ── Module 8: Label Pipeline (integration) ───────────────────────────────────

class TestLabelPipeline:
    def _pipeline(self, tmp_path):
        from src.labels.label_pipeline import LabelPipeline, LabelPipelineConfig
        from src.labels.trade_outcome   import TradeOutcomeConfig
        from src.labels.setup_quality   import SetupQualityConfig
        from src.labels.entry_timing    import EntryTimingConfig
        from src.labels.trade_management import TradeManagementConfig
        from src.labels.market_bias     import MarketBiasConfig

        cfg = LabelPipelineConfig(
            market_bias      = MarketBiasConfig(horizons=[1, 3], rolling_vol_window=20),
            trade_outcome    = TradeOutcomeConfig(atr_period=5, max_bars=10),
            setup_quality    = SetupQualityConfig(atr_period=5, max_bars=10),
            entry_timing     = EntryTimingConfig(atr_period=5, max_bars=10, window_size=5),
            trade_management = TradeManagementConfig(atr_period=5, max_bars=10),
        )
        return LabelPipeline(
            label_dir=tmp_path / "labels",
            report_dir=tmp_path / "reports",
            config=cfg,
        )

    def test_run_produces_labels(self, tmp_path):
        pipeline = self._pipeline(tmp_path)
        result   = pipeline.run(DF, symbol="EURUSD", write=True)
        assert isinstance(result.labels, pd.DataFrame)
        assert len(result.labels) == len(DF)

    def test_labels_have_all_models(self, tmp_path):
        result = self._pipeline(tmp_path).run(DF, symbol="EURUSD", write=False)
        cols   = result.labels.columns.tolist()
        assert any("bias_"   in c for c in cols)
        assert any("outcome" in c for c in cols)
        assert any("setup_"  in c for c in cols)
        assert any("entry_"  in c for c in cols)
        assert any("mgmt_"   in c for c in cols)

    def test_parquet_saved(self, tmp_path):
        result = self._pipeline(tmp_path).run(DF, symbol="EURUSD", write=True)
        assert result.parquet_path is not None
        assert result.parquet_path.exists()

    def test_parquet_loadable(self, tmp_path):
        result  = self._pipeline(tmp_path).run(DF, symbol="EURUSD", write=True)
        loaded  = pd.read_parquet(result.parquet_path)
        assert loaded.shape == result.labels.shape

    def test_input_df_not_mutated(self, tmp_path):
        original = DF.copy()
        self._pipeline(tmp_path).run(DF, symbol="EURUSD", write=False)
        pd.testing.assert_frame_equal(DF, original)

    def test_metadata_generated(self, tmp_path):
        result = self._pipeline(tmp_path).run(DF, symbol="EURUSD", write=True)
        assert result.metadata.symbol == "EURUSD"
        assert result.metadata.n_rows == len(DF)

    def test_reports_generated(self, tmp_path):
        result = self._pipeline(tmp_path).run(DF, symbol="EURUSD", write=True)
        assert (tmp_path / "reports" / "label_metadata.json").exists()
        assert (tmp_path / "reports" / "label_report.md").exists()

    def test_ohlcv_extraction_with_prefix(self, tmp_path):
        # Rename OHLCV cols with 'h1_' prefix
        prefixed = DF.rename(columns={
            "open": "h1_open", "high": "h1_high", "low": "h1_low",
            "close": "h1_close", "volume": "h1_volume",
        })
        from src.labels.label_pipeline import LabelPipeline, LabelPipelineConfig
        cfg = LabelPipelineConfig(ohlcv_prefix="h1_")
        pipeline = LabelPipeline(
            label_dir=tmp_path / "lbl2",
            report_dir=tmp_path / "rep2",
            config=cfg,
        )
        result = pipeline.run(prefixed, symbol="GBPUSD", write=False)
        assert len(result.labels) == len(DF)


# ── No-Leakage invariant ──────────────────────────────────────────────────────

class TestNoLeakage:
    """Verify that label columns are never present in a feature DataFrame."""

    def test_label_column_names_not_in_feature_cols(self, tmp_path):
        from src.labels.label_pipeline import LabelPipeline
        result  = LabelPipeline(
            label_dir=tmp_path / "lbl",
            report_dir=tmp_path / "rep",
        ).run(DF, symbol="EURUSD", write=False)
        label_cols   = set(result.labels.columns)
        feature_cols = set(DF.columns)
        assert label_cols.isdisjoint(feature_cols), \
            f"Overlap: {label_cols & feature_cols}"

    def test_labels_index_matches_ohlcv(self, tmp_path):
        from src.labels.label_pipeline import LabelPipeline
        result = LabelPipeline(
            label_dir=tmp_path / "lbl",
            report_dir=tmp_path / "rep",
        ).run(DF, symbol="EURUSD", write=False)
        assert result.labels.index.equals(DF.index)


# ── simulate_trade edge cases ─────────────────────────────────────────────────

class TestSimulateTradeEdgeCases:
    from src.labels.trade_outcome import simulate_trade, TIMEOUT, TP_FIRST, SL_FIRST

    def test_empty_future_bars(self):
        from src.labels.trade_outcome import simulate_trade, TIMEOUT
        out, dur, mfe, mae = simulate_trade(
            np.array([]), np.array([]), 1.01, 0.99, 1.0, 1
        )
        assert out == TIMEOUT
        assert dur == 0
        assert mfe == 0.0
        assert mae == 0.0

    def test_sl_priority_when_both_hit_same_bar(self):
        """When high hits TP and low hits SL on same bar, SL wins."""
        from src.labels.trade_outcome import simulate_trade, SL_FIRST
        f_high = np.array([1.02])   # hits TP (1.01)
        f_low  = np.array([0.98])   # hits SL (0.99)
        out, dur, mfe, mae = simulate_trade(f_high, f_low, 1.01, 0.99, 1.0, 1)
        assert out == SL_FIRST

    def test_short_tp_detection(self):
        """For a short trade, TP is hit when price goes DOWN."""
        from src.labels.trade_outcome import simulate_trade, TP_FIRST
        f_high = np.full(10, 1.001)   # barely above entry
        f_low  = np.full(10, 0.97)    # far below TP=0.99 for short
        out, dur, mfe, mae = simulate_trade(f_high, f_low, 0.99, 1.01, 1.0, -1)
        assert out == TP_FIRST

    def test_mfe_computed_only_within_trade_duration(self):
        """MFE should NOT include price action after trade ends."""
        from src.labels.trade_outcome import simulate_trade, SL_FIRST
        # SL hit at bar 2; bar 5+ has huge high (should not count)
        f_high = np.array([1.001, 1.001, 100.0, 100.0])
        f_low  = np.array([0.999, 0.98,  0.98,  0.98])  # SL=0.99 hit at bar 2
        out, dur, mfe, mae = simulate_trade(f_high, f_low, 1.02, 0.99, 1.0, 1)
        assert out == SL_FIRST
        assert dur == 2
        # MFE = max(1.001, 1.001) - 1.0 / 1.0 = 0.001 ; NOT 100-1=99
        assert mfe < 1.0


# ── compute_atr sanity ────────────────────────────────────────────────────────

class TestComputeATR:
    def test_atr_positive(self):
        from src.labels.trade_outcome import compute_atr
        atr = compute_atr(DF, period=14)
        assert (atr.dropna() > 0).all()

    def test_atr_length_equals_input(self):
        from src.labels.trade_outcome import compute_atr
        atr = compute_atr(DF, period=14)
        assert len(atr) == len(DF)

    def test_atr_nan_at_start(self):
        from src.labels.trade_outcome import compute_atr
        atr = compute_atr(DF, period=14)
        # First bar has NaN (no previous close for TR)
        assert pd.isna(atr.iloc[0])
