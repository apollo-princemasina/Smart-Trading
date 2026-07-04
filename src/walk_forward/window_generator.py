"""
Window Generator
================
Generates a sequence of (train, validation, test) date-boundary specifications
for walk-forward cross-validation.

No data is loaded here — this module works exclusively with DatetimeIndex
timestamps and configuration, producing ``WindowSpec`` objects that the
downstream splitter uses to slice the actual dataset.

Window strategies
-----------------
rolling   : Fixed-size train window slides forward by ``step_period`` each round.
            Train size is constant; start date advances every iteration.

expanding : Anchor-based.  Train start is fixed; train end grows by
            ``step_period`` each round, so the train set expands over time.

anchored  : Single window — train from ``anchor_date`` to
            ``anchor_date + train_period``; one val and one test period follow.
            Identical to the first iteration of expanding with no step.

Period strings
--------------
"Ny"  — N calendar years
"Nm"  — N months
"Nw"  — N weeks
"Nd"  — N days

Example
-------
    idx    = dataset.index
    config = WindowConfig(
        train_period="5y", val_period="1y", test_period="1y",
        step_period="1y",  window_type="rolling",
    )
    specs  = WindowGenerator().generate(idx, config)
    # → [WindowSpec(window_number=0, train_start=..., ...), ...]
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_PERIOD_RE = re.compile(r"^(\d+)([ymwd])$", re.IGNORECASE)

_WINDOW_TYPES = frozenset({"rolling", "expanding", "anchored", "sliding"})


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class WindowSpec:
    """Date boundaries for one walk-forward window (all timestamps inclusive)."""
    window_number: int
    train_start:   pd.Timestamp
    train_end:     pd.Timestamp
    val_start:     pd.Timestamp
    val_end:       pd.Timestamp
    test_start:    pd.Timestamp
    test_end:      pd.Timestamp

    # ------------------------------------------------------------------
    def train_duration_days(self) -> float:
        return (self.train_end - self.train_start).total_seconds() / 86_400

    def val_duration_days(self) -> float:
        return (self.val_end - self.val_start).total_seconds() / 86_400

    def test_duration_days(self) -> float:
        return (self.test_end - self.test_start).total_seconds() / 86_400

    def __str__(self) -> str:
        return (
            f"Window {self.window_number:03d} | "
            f"Train [{self.train_start.date()}, {self.train_end.date()}] | "
            f"Val   [{self.val_start.date()},  {self.val_end.date()}]  | "
            f"Test  [{self.test_start.date()},  {self.test_end.date()}]"
        )


@dataclass
class WindowConfig:
    """Configuration for walk-forward window generation.

    Attributes:
        window_type:       Strategy — rolling / expanding / anchored / sliding.
        train_period:      Size of the training window (e.g. "5y", "18m").
        val_period:        Size of the validation window.
        test_period:       Size of the test window.
        step_period:       How far to advance the window each iteration.
        anchor_date:       Fixed start date for expanding/anchored windows.
                           Defaults to the first date in the dataset.
        gap_bars:          Number of bars to leave between consecutive splits
                           (purge period — avoids label overlap leakage).
        min_train_samples: Minimum rows required in the train split.
        min_val_samples:   Minimum rows required in the validation split.
        min_test_samples:  Minimum rows required in the test split.
        max_windows:       Hard cap on total generated windows (0 = unlimited).
    """
    window_type:       str           = "rolling"
    train_period:      str           = "5y"
    val_period:        str           = "1y"
    test_period:       str           = "1y"
    step_period:       str           = "1y"
    anchor_date:       Optional[str] = None     # "YYYY-MM-DD"
    gap_bars:          int           = 0
    min_train_samples: int           = 100
    min_val_samples:   int           = 50
    min_test_samples:  int           = 50
    max_windows:       int           = 0        # 0 = unlimited


# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_period(period_str: str) -> pd.DateOffset:
    """Convert a period string like "5y", "12m", "4w", "30d" to DateOffset."""
    m = _PERIOD_RE.match(period_str.strip())
    if not m:
        raise ValueError(
            f"Invalid period '{period_str}'. "
            "Use format Ny / Nm / Nw / Nd (e.g. '5y', '12m', '4w', '30d')."
        )
    n, unit = int(m.group(1)), m.group(2).lower()
    if unit == "y": return pd.DateOffset(years=n)
    if unit == "m": return pd.DateOffset(months=n)
    if unit == "w": return pd.DateOffset(weeks=n)
    if unit == "d": return pd.DateOffset(days=n)
    raise ValueError(f"Unknown period unit '{unit}'.")   # unreachable


def _last_bar_before(
    index: pd.DatetimeIndex, ts: pd.Timestamp
) -> Optional[pd.Timestamp]:
    """Return the last index timestamp strictly before *ts*."""
    mask = index < ts
    return index[mask][-1] if mask.any() else None


def _first_bar_at_or_after(
    index: pd.DatetimeIndex, ts: pd.Timestamp
) -> Optional[pd.Timestamp]:
    """Return the first index timestamp at or after *ts*."""
    mask = index >= ts
    return index[mask][0] if mask.any() else None


def _bar_pos(index: pd.DatetimeIndex, ts: pd.Timestamp) -> int:
    """Integer position of *ts* in *index*. *ts* must be in the index."""
    pos = index.get_loc(ts)
    return pos.start if isinstance(pos, slice) else int(pos)


# ── Generator ─────────────────────────────────────────────────────────────────

class WindowGenerator:
    """Generate walk-forward window boundary specifications."""

    def generate(
        self,
        index:  pd.DatetimeIndex,
        config: WindowConfig,
    ) -> list[WindowSpec]:
        """Return a list of WindowSpec objects for the given dataset index.

        Args:
            index:  The DatetimeIndex of the full dataset.
            config: Window configuration.

        Returns:
            Ordered list of WindowSpec (chronological).

        Raises:
            ValueError: If the configuration is invalid or index is too short.
        """
        self._validate_config(config, index)
        wtype = config.window_type.lower()
        if wtype in ("rolling", "sliding"):
            return self._rolling(index, config)
        if wtype == "expanding":
            return self._expanding(index, config)
        if wtype == "anchored":
            return self._anchored(index, config)
        raise ValueError(f"Unknown window_type '{config.window_type}'.")

    # ── Rolling ──────────────────────────────────────────────────────────

    def _rolling(self, index: pd.DatetimeIndex, cfg: WindowConfig) -> list[WindowSpec]:
        train_off = parse_period(cfg.train_period)
        val_off   = parse_period(cfg.val_period)
        test_off  = parse_period(cfg.test_period)
        step_off  = parse_period(cfg.step_period)
        data_start = index[0]
        specs: list[WindowSpec] = []

        n = 0
        current_train_start = data_start

        while True:
            spec = self._compute_window(
                index, current_train_start,
                current_train_start + train_off,
                val_off, test_off, n, cfg.gap_bars,
            )
            if spec is None:
                break
            if not self._meets_min_samples(index, spec, cfg):
                break

            specs.append(spec)
            logger.debug("Rolling %s", spec)
            n += 1

            if cfg.max_windows and n >= cfg.max_windows:
                break

            # Advance train_start by step
            next_start = _first_bar_at_or_after(index, current_train_start + step_off)
            if next_start is None or next_start >= index[-1]:
                break
            current_train_start = next_start

        return specs

    # ── Expanding ────────────────────────────────────────────────────────

    def _expanding(self, index: pd.DatetimeIndex, cfg: WindowConfig) -> list[WindowSpec]:
        train_off  = parse_period(cfg.train_period)
        val_off    = parse_period(cfg.val_period)
        test_off   = parse_period(cfg.test_period)
        step_off   = parse_period(cfg.step_period)
        anchor     = self._resolve_anchor(index, cfg)
        specs: list[WindowSpec] = []

        n = 0
        while True:
            # Train end target grows by n steps beyond the initial period
            train_end_target = anchor + train_off
            for _ in range(n):
                train_end_target = train_end_target + step_off

            spec = self._compute_window(
                index, anchor,
                train_end_target,
                val_off, test_off, n, cfg.gap_bars,
            )
            if spec is None:
                break
            if not self._meets_min_samples(index, spec, cfg):
                break

            specs.append(spec)
            logger.debug("Expanding %s", spec)
            n += 1

            if cfg.max_windows and n >= cfg.max_windows:
                break

        return specs

    # ── Anchored (single window) ──────────────────────────────────────────

    def _anchored(self, index: pd.DatetimeIndex, cfg: WindowConfig) -> list[WindowSpec]:
        train_off = parse_period(cfg.train_period)
        val_off   = parse_period(cfg.val_period)
        test_off  = parse_period(cfg.test_period)
        anchor    = self._resolve_anchor(index, cfg)

        spec = self._compute_window(
            index, anchor, anchor + train_off,
            val_off, test_off, 0, cfg.gap_bars,
        )
        return [spec] if spec else []

    # ── Core computation ─────────────────────────────────────────────────

    def _compute_window(
        self,
        index:              pd.DatetimeIndex,
        train_start:        pd.Timestamp,
        train_end_target:   pd.Timestamp,
        val_off:            pd.DateOffset,
        test_off:           pd.DateOffset,
        window_num:         int,
        gap_bars:           int,
    ) -> Optional[WindowSpec]:
        """
        Compute one window spec from boundary targets.
        Returns None when the window cannot be formed (insufficient data).
        """
        # Train end: last bar BEFORE train_end_target
        train_end = _last_bar_before(index, train_end_target)
        if train_end is None or train_end <= train_start:
            return None

        # Val start: first bar after train_end (+ gap)
        te_pos    = _bar_pos(index, train_end)
        vs_pos    = te_pos + 1 + gap_bars
        if vs_pos >= len(index):
            return None
        val_start = index[vs_pos]

        # Val end: last bar before val_start + val_period
        val_end = _last_bar_before(index, val_start + val_off)
        if val_end is None or val_end <= val_start:
            return None

        # Test start: first bar after val_end (+ gap)
        ve_pos     = _bar_pos(index, val_end)
        ts_pos     = ve_pos + 1 + gap_bars
        if ts_pos >= len(index):
            return None
        test_start = index[ts_pos]

        # Test end: last bar before test_start + test_period
        test_end = _last_bar_before(index, test_start + test_off)
        if test_end is None or test_end <= test_start:
            return None

        return WindowSpec(
            window_number=window_num,
            train_start=train_start,
            train_end=train_end,
            val_start=val_start,
            val_end=val_end,
            test_start=test_start,
            test_end=test_end,
        )

    # ── Helpers ───────────────────────────────────────────────────────────

    def _resolve_anchor(self, index: pd.DatetimeIndex, cfg: WindowConfig) -> pd.Timestamp:
        if cfg.anchor_date:
            ts = pd.Timestamp(cfg.anchor_date)
            snapped = _first_bar_at_or_after(index, ts)
            if snapped is None:
                raise ValueError(f"anchor_date '{cfg.anchor_date}' is beyond the dataset end.")
            return snapped
        return index[0]

    def _meets_min_samples(
        self, index: pd.DatetimeIndex, spec: WindowSpec, cfg: WindowConfig
    ) -> bool:
        train_n = int(((index >= spec.train_start) & (index <= spec.train_end)).sum())
        val_n   = int(((index >= spec.val_start)   & (index <= spec.val_end)).sum())
        test_n  = int(((index >= spec.test_start)  & (index <= spec.test_end)).sum())
        if train_n < cfg.min_train_samples:
            logger.debug("Window %d rejected: train has %d < %d samples.",
                         spec.window_number, train_n, cfg.min_train_samples)
            return False
        if val_n < cfg.min_val_samples:
            logger.debug("Window %d rejected: val has %d < %d samples.",
                         spec.window_number, val_n, cfg.min_val_samples)
            return False
        if test_n < cfg.min_test_samples:
            logger.debug("Window %d rejected: test has %d < %d samples.",
                         spec.window_number, test_n, cfg.min_test_samples)
            return False
        return True

    @staticmethod
    def _validate_config(cfg: WindowConfig, index: pd.DatetimeIndex) -> None:
        if cfg.window_type.lower() not in _WINDOW_TYPES:
            raise ValueError(
                f"Unknown window_type '{cfg.window_type}'. "
                f"Choose from: {sorted(_WINDOW_TYPES)}."
            )
        if len(index) == 0:
            raise ValueError("Dataset index is empty.")
        if not isinstance(index, pd.DatetimeIndex):
            raise TypeError("Dataset index must be a DatetimeIndex.")
        if not index.is_monotonic_increasing:
            raise ValueError("Dataset index is not monotonically increasing. Sort it first.")
        for attr in ("train_period", "val_period", "test_period", "step_period"):
            parse_period(getattr(cfg, attr))   # raises on bad string
