"""Core multi-timeframe feature fusion using merge_asof."""

from __future__ import annotations

import logging

import pandas as pd

from .timeframe_mapper  import TimeframeMapper
from .feature_alignment import FeatureAligner
from .fusion_validator  import FusionValidator, ValidationResult

logger = logging.getLogger(__name__)


class FeatureFusion:
    """
    Fuses feature DataFrames from multiple timeframes onto the M15 base index.

    Each higher-timeframe DataFrame is aligned with
    ``merge_asof(direction='backward')`` so that every M15 bar receives only
    *completed* HTF candle features — zero look-ahead bias.

    Column naming
    -------------
    Every column is prefixed by its timeframe:
    ``weekly_*``, ``daily_*``, ``h4_*``, ``h1_*``, ``m15_*``.

    Internal ``__bar_open_*`` columns are added during alignment for
    post-fusion validation and stripped from the final output by default.
    """

    BASE_TF   = "M15"
    HTF_ORDER = ["W", "D", "H4", "H1"]   # high → low

    def __init__(self, validate: bool = True, strict: bool = False):
        self._aligner   = FeatureAligner()
        self._validator = FusionValidator()
        self._validate  = validate
        self._strict    = strict

    # ─────────────────────────────────────────────────────────────────────────

    def fuse(
        self,
        timeframe_dfs: dict[str, pd.DataFrame],
        drop_internal: bool = True,
    ) -> tuple[pd.DataFrame, ValidationResult]:
        """
        Fuse all timeframe DataFrames onto the M15 base index.

        Args:
            timeframe_dfs:  ``{tf_string: DataFrame}`` — must include ``"M15"``.
            drop_internal:  Strip ``__bar_open_*`` metadata from the output.

        Returns:
            ``(fused_df, validation_result)``
        """
        # Normalise all keys
        normalised: dict[str, pd.DataFrame] = {
            TimeframeMapper.normalise(k): v for k, v in timeframe_dfs.items()
        }

        base_canon = TimeframeMapper.normalise(self.BASE_TF)
        if base_canon not in normalised:
            raise KeyError(f"'{self.BASE_TF}' timeframe is required for fusion")

        m15_df    = normalised[base_canon]
        m15_index = m15_df.index

        # ── Pre-fusion validation ─────────────────────────────────────────────
        val_result = ValidationResult()
        if self._validate:
            val_result = self._validator.validate_all(
                normalised, base_canon, strict=self._strict
            )

        # ── 1. Prefix M15 columns (no index shift needed) ────────────────────
        parts: list[pd.DataFrame] = [self._aligner.prefix_base(m15_df)]

        # ── 2. Align each HTF onto M15 ────────────────────────────────────────
        for tf in self.HTF_ORDER:
            canon = TimeframeMapper.normalise(tf)
            if canon not in normalised:
                logger.debug("Timeframe %s not provided — skipping", tf)
                continue
            logger.debug("Aligning %s → M15", tf)
            aligned = self._aligner.align(normalised[canon], canon, m15_index)
            parts.append(aligned)

        # ── 3. Concatenate all parts ──────────────────────────────────────────
        fused = pd.concat(parts, axis=1)
        fused.index = m15_index

        # ── 4. Post-fusion validation ─────────────────────────────────────────
        if self._validate:
            val_result.merge(self._validator.validate_no_lookahead(fused))
            col_groups = {
                tf: [
                    c for c in fused.columns
                    if c.startswith(TimeframeMapper.prefix(tf) + "_")
                ]
                for tf in self.HTF_ORDER + [self.BASE_TF]
                if TimeframeMapper.is_valid(tf)
            }
            val_result.merge(self._validator.validate_no_duplicates(col_groups))

        # ── 5. Optionally strip internal metadata ────────────────────────────
        if drop_internal:
            fused = FeatureAligner.drop_internal_columns(fused)

        logger.info(
            "Fusion complete: %d rows × %d columns", len(fused), fused.shape[1]
        )
        return fused, val_result

    def describe(self, fused_df: pd.DataFrame) -> dict:
        """Return a structural summary of the fused DataFrame."""
        prefix_counts: dict[str, int] = {}
        for col in fused_df.columns:
            pfx = col.split("_")[0] if "_" in col else "other"
            prefix_counts[pfx] = prefix_counts.get(pfx, 0) + 1
        return {
            "n_rows":         len(fused_df),
            "n_cols":         fused_df.shape[1],
            "start":          str(fused_df.index[0])  if len(fused_df) else None,
            "end":            str(fused_df.index[-1]) if len(fused_df) else None,
            "cols_by_prefix": prefix_counts,
            "nan_pct":        self._validator.validate_completeness(fused_df),
        }
