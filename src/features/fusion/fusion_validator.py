"""Validation suite for multi-timeframe feature fusion."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .timeframe_mapper  import TimeframeMapper
from .feature_alignment import _BAR_OPEN_PREFIX


@dataclass
class ValidationResult:
    """Aggregated errors, warnings, and stats from fusion validation."""

    is_valid:  bool      = True
    errors:    list[str] = field(default_factory=list)
    warnings:  list[str] = field(default_factory=list)
    stats:     dict      = field(default_factory=dict)

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)
        self.is_valid = False

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)

    def merge(self, other: ValidationResult) -> ValidationResult:
        """Absorb *other* in-place and return self."""
        self.errors   += other.errors
        self.warnings += other.warnings
        self.stats.update(other.stats)
        if not other.is_valid:
            self.is_valid = False
        return self

    def raise_if_invalid(self) -> None:
        if not self.is_valid:
            bullet = "\n  • ".join(self.errors)
            raise ValueError(f"Fusion validation failed:\n  • {bullet}")


class FusionValidator:
    """Validates timeframe DataFrames before and after fusion."""

    # ── Pre-fusion checks ─────────────────────────────────────────────────────

    def validate_timezone(self, df: pd.DataFrame, name: str) -> ValidationResult:
        """Verify the DataFrame index is timezone-aware (UTC preferred)."""
        result = ValidationResult()
        if not isinstance(df.index, pd.DatetimeIndex):
            result.add_error(f"[{name}] Index is not DatetimeIndex")
            return result
        if df.index.tz is None:
            result.add_error(f"[{name}] Index is timezone-naive — UTC required")
        elif str(df.index.tz) not in ("UTC", "Etc/UTC"):
            result.add_warning(
                f"[{name}] Index timezone is {df.index.tz!s}, expected UTC"
            )
        return result

    def validate_monotonic(self, df: pd.DataFrame, name: str) -> ValidationResult:
        """Verify the index is strictly monotonically increasing with no dupes."""
        result = ValidationResult()
        if not df.index.is_monotonic_increasing:
            result.add_error(f"[{name}] Index is not monotonically increasing")
        n_dup = int(df.index.duplicated().sum())
        if n_dup:
            result.add_error(f"[{name}] {n_dup} duplicate timestamp(s) in index")
        return result

    def validate_time_consistency(
        self,
        timeframe_dfs: dict[str, pd.DataFrame],
        base_tf: str = "M15",
    ) -> ValidationResult:
        """Warn when < 80 % of consecutive bars have the expected gap."""
        result = ValidationResult()
        base_canon = TimeframeMapper.normalise(base_tf)
        for tf, df in timeframe_dfs.items():
            canon = TimeframeMapper.normalise(tf)
            if canon == base_canon or len(df) < 2:
                continue
            expected = TimeframeMapper.timedelta(canon)
            diffs    = df.index[1:] - df.index[:-1]
            pct      = float((diffs == expected).sum()) / len(diffs) * 100
            result.stats[f"{canon}_time_consistency_pct"] = round(pct, 2)
            if pct < 80:
                result.add_warning(
                    f"[{canon}] Only {pct:.1f}% of bars have the expected "
                    f"{expected} gap (market holidays / weekend gaps expected)"
                )
        return result

    # ── Post-fusion checks ────────────────────────────────────────────────────

    def validate_no_lookahead(self, fused_df: pd.DataFrame) -> ValidationResult:
        """
        Detect look-ahead bias.

        For each internal ``__bar_open_{tf}`` column, verify that every M15
        bar uses an HTF bar whose close time (open + duration) is ≤ the M15
        bar's own open time.
        """
        result = ValidationResult()
        m15_ts = fused_df.index

        for col in fused_df.columns:
            if not col.startswith(_BAR_OPEN_PREFIX):
                continue
            tf_norm = col[len(_BAR_OPEN_PREFIX):]
            try:
                offset = TimeframeMapper.timedelta(tf_norm)
            except ValueError:
                continue

            bar_open   = pd.to_datetime(fused_df[col])
            valid_mask = bar_open.notna()
            if not valid_mask.any():
                continue

            avail_at   = bar_open[valid_mask] + offset
            violations = int((avail_at > m15_ts[valid_mask]).sum())

            if violations:
                result.add_error(
                    f"[look-ahead][{tf_norm}] {violations} M15 bar(s) use an "
                    f"HTF bar that had not yet closed"
                )
            else:
                result.stats[f"{tf_norm}_lookahead_violations"] = 0

        return result

    def validate_no_duplicates(
        self,
        col_groups: dict[str, list[str]],
    ) -> ValidationResult:
        """
        Detect duplicate column names across timeframe groups.

        *col_groups* should be ``{tf: [column_names]}``.
        """
        result = ValidationResult()
        seen:  set[str] = set()
        dupes: set[str] = set()
        for cols in col_groups.values():
            for c in cols:
                (dupes if c in seen else seen).add(c)
                seen.add(c)
        if dupes:
            result.add_error(
                f"Duplicate columns across timeframes: {sorted(dupes)}"
            )
        return result

    def validate_completeness(self, fused_df: pd.DataFrame) -> dict[str, float]:
        """NaN percentage per column (0.0 = complete, 100.0 = all NaN)."""
        n = len(fused_df)
        if n == 0:
            return {}
        return {
            c: float(fused_df[c].isna().sum()) / n * 100
            for c in fused_df.columns
        }

    # ── Orchestrator ──────────────────────────────────────────────────────────

    def validate_all(
        self,
        timeframe_dfs: dict[str, pd.DataFrame],
        base_tf: str = "M15",
        strict: bool = False,
    ) -> ValidationResult:
        """Run all pre-fusion checks; raise ValueError if *strict* and invalid."""
        combined = ValidationResult()
        for tf, df in timeframe_dfs.items():
            combined.merge(self.validate_timezone(df, tf))
            combined.merge(self.validate_monotonic(df, tf))
        combined.merge(self.validate_time_consistency(timeframe_dfs, base_tf))
        if strict:
            combined.raise_if_invalid()
        return combined
