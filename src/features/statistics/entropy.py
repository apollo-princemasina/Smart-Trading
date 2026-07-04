"""Information entropy features: Shannon entropy and approximate entropy of returns."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..base_feature     import BaseFeature
from ..feature_metadata import FeatureMetadata
from ..feature_registry import FeatureRegistry

_ENT_LONG    = 20    # Shannon entropy long window
_ENT_SHORT   = 5     # Shannon entropy short window
_APEN_WIN    = 30    # approximate entropy window (kept small for speed)
_N_BINS      = 10    # histogram bins for Shannon entropy

_ENTROPY_COLS: list[str] = [
    "entropy",
    "rolling_entropy_5",
    "approximate_entropy",
]


def _shannon_entropy(x: np.ndarray) -> float:
    """Shannon entropy (bits) of x discretised into _N_BINS equal-width bins."""
    if len(x) < 2:
        return 0.0
    hist, _ = np.histogram(x, bins=_N_BINS)
    total   = hist.sum()
    if total == 0:
        return 0.0
    p = hist[hist > 0] / total
    return float(-np.sum(p * np.log2(p)))


def _approx_entropy(x: np.ndarray, m: int = 2, r_scale: float = 0.2) -> float:
    """Approximate entropy (ApEn) using vectorised template matching."""
    n = len(x)
    if n < m + 2:
        return 0.0
    r = r_scale * np.std(x, ddof=1)
    if r < 1e-10:
        return 0.0

    def _phi(m_: int) -> float:
        if n - m_ < 1:
            return 0.0
        # Build template matrix (n-m_, m_)
        try:
            tm = np.lib.stride_tricks.sliding_window_view(x, m_)
        except ValueError:
            return 0.0
        # Vectorised max-norm distance between all template pairs
        diff    = np.abs(tm[:, None, :] - tm[None, :, :]).max(axis=2)
        matches = (diff <= r).sum(axis=1)
        counts  = np.maximum(matches, 1)
        return float(np.log(counts / (n - m_ + 1)).mean())

    return float(abs(_phi(m) - _phi(m + 1)))


@FeatureRegistry.register
class EntropyEngine(BaseFeature):
    """Shannon entropy (20-bar, 5-bar) and approximate entropy (30-bar) of log returns."""

    name:             str       = "entropy"
    category:         str       = "statistics"
    dependencies:     list[str] = ["returns"]
    required_columns: list[str] = ["log_return"]

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        log_ret = df["log_return"].to_numpy(dtype=np.float64)
        lr_s    = pd.Series(log_ret, index=df.index)

        # ── Shannon entropy ───────────────────────────────────────────────────
        entropy = lr_s.rolling(_ENT_LONG, min_periods=4).apply(
            _shannon_entropy, raw=True).to_numpy()
        entropy = np.where(np.isnan(entropy), 0.0, entropy)

        rolling_entropy_5 = lr_s.rolling(_ENT_SHORT, min_periods=2).apply(
            _shannon_entropy, raw=True).to_numpy()
        rolling_entropy_5 = np.where(np.isnan(rolling_entropy_5), 0.0, rolling_entropy_5)

        # ── Approximate entropy (ApEn) ─────────────────────────────────────────
        approximate_entropy = lr_s.rolling(_APEN_WIN, min_periods=_APEN_WIN // 2).apply(
            _approx_entropy, raw=True).to_numpy()
        approximate_entropy = np.where(np.isnan(approximate_entropy), 0.0, approximate_entropy)

        out = pd.DataFrame(index=df.index)
        out["entropy"]              = entropy
        out["rolling_entropy_5"]    = rolling_entropy_5
        out["approximate_entropy"]  = approximate_entropy
        return out.astype(np.float64)

    def metadata(self) -> FeatureMetadata:
        return FeatureMetadata(
            name        = self.name,
            category    = self.category,
            description = (
                "Shannon entropy (bits) of log_return histogram: 20-bar and 5-bar. "
                "Approximate entropy (ApEn, m=2, r=0.2σ) on 30-bar window. "
                "High entropy ≈ random/unpredictable; low entropy ≈ structured. "
                "3 float64 columns."
            ),
            dependencies     = self.dependencies,
            required_columns = self.required_columns,
            output_columns   = _ENTROPY_COLS,
            version    = "1.0.0",
            author     = "Smart Trading Team",
            complexity = "high",
            tags       = ["entropy", "information", "approximate_entropy", "statistics"],
        )
