"""
Split Validator
===============
Validates that walk-forward splits are free of temporal leakage and meet
minimum sample requirements.

Checks
------
1. chronological_order — Each split end < next split start.
2. no_overlap          — No timestamp appears in more than one split.
3. min_train_samples   — Train split meets the configured minimum.
4. min_val_samples     — Validation split meets the configured minimum.
5. min_test_samples    — Test split meets the configured minimum.
6. no_shuffle          — Rows within each split are in ascending time order.
7. no_future_in_train  — No train timestamp is later than the earliest val/test bar.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

from .dataset_splitter import SplitResult

logger = logging.getLogger(__name__)

PASS    = "PASS"
WARNING = "WARNING"
FAIL    = "FAIL"


@dataclass
class SplitIssue:
    severity: str
    check:    str
    message:  str


@dataclass
class SplitValidationReport:
    window_number: int
    passed:        bool
    issues:        list[SplitIssue] = field(default_factory=list)

    def failures(self)  -> list[SplitIssue]:
        return [i for i in self.issues if i.severity == FAIL]

    def warnings(self) -> list[SplitIssue]:
        return [i for i in self.issues if i.severity == WARNING]

    def __str__(self) -> str:
        status = "PASSED" if self.passed else "FAILED"
        lines  = [f"SplitValidation window={self.window_number}: {status}"]
        for issue in self.issues:
            if issue.severity != PASS:
                lines.append(f"  [{issue.severity}] {issue.check}: {issue.message}")
        return "\n".join(lines)


@dataclass
class SplitValidatorConfig:
    min_train_samples: int = 100
    min_val_samples:   int = 50
    min_test_samples:  int = 50


class SplitValidator:
    """Validate that a SplitResult is temporally sound."""

    def __init__(self, config: SplitValidatorConfig | None = None) -> None:
        self.config = config or SplitValidatorConfig()

    def validate(self, result: SplitResult) -> SplitValidationReport:
        issues: list[SplitIssue] = []
        cfg = self.config
        wn  = result.window_number

        self._check_chronological_order(result, issues)
        self._check_no_overlap(result, issues)
        self._check_no_shuffle(result, issues)
        self._check_no_future_in_train(result, issues)
        self._check_min_samples(result, issues, cfg)

        passed = not any(i.severity == FAIL for i in issues)
        report = SplitValidationReport(window_number=wn, passed=passed, issues=issues)
        if not passed:
            logger.warning("Window %03d split validation FAILED: %s",
                           wn, [i.message for i in report.failures()])
        return report

    # ── Checks ────────────────────────────────────────────────────────────────

    def _check_chronological_order(self, r: SplitResult, issues: list) -> None:
        ok = True
        for (a_name, a_df), (b_name, b_df) in [
            (("train", r.train), ("val", r.validation)),
            (("val", r.validation), ("test", r.test)),
        ]:
            if a_df.empty or b_df.empty:
                continue
            if a_df.index[-1] >= b_df.index[0]:
                issues.append(SplitIssue(FAIL, "chronological_order",
                    f"{a_name} end ({a_df.index[-1]}) >= {b_name} start ({b_df.index[0]})."))
                ok = False
        if ok:
            issues.append(SplitIssue(PASS, "chronological_order",
                                     "train < val < test chronologically."))

    def _check_no_overlap(self, r: SplitResult, issues: list) -> None:
        sets = {
            "train": set(r.train.index),
            "val":   set(r.validation.index),
            "test":  set(r.test.index),
        }
        overlaps = []
        for (a, sa), (b, sb) in [
            (("train", sets["train"]), ("val",   sets["val"])),
            (("train", sets["train"]), ("test",  sets["test"])),
            (("val",   sets["val"]),   ("test",  sets["test"])),
        ]:
            inter = sa & sb
            if inter:
                overlaps.append(f"{a}∩{b}={len(inter)} rows")
        if overlaps:
            issues.append(SplitIssue(FAIL, "no_overlap", "; ".join(overlaps)))
        else:
            issues.append(SplitIssue(PASS, "no_overlap", "No overlapping rows."))

    def _check_no_shuffle(self, r: SplitResult, issues: list) -> None:
        for name, df in [("train", r.train), ("val", r.validation), ("test", r.test)]:
            if df.empty:
                continue
            if not df.index.is_monotonic_increasing:
                issues.append(SplitIssue(FAIL, "no_shuffle",
                    f"{name} split is NOT in ascending chronological order."))
                return
        issues.append(SplitIssue(PASS, "no_shuffle",
                                 "All splits are in ascending order."))

    def _check_no_future_in_train(self, r: SplitResult, issues: list) -> None:
        if r.train.empty:
            return
        val_start  = r.validation.index[0] if not r.validation.empty else None
        test_start = r.test.index[0]        if not r.test.empty        else None
        earliest_future = min(
            (ts for ts in [val_start, test_start] if ts is not None),
            default=None,
        )
        if earliest_future is None:
            return
        leaking = int((r.train.index >= earliest_future).sum())
        if leaking:
            issues.append(SplitIssue(FAIL, "no_future_in_train",
                f"{leaking} train rows have timestamps >= val/test start ({earliest_future})."))
        else:
            issues.append(SplitIssue(PASS, "no_future_in_train",
                                     "No future data in train split."))

    def _check_min_samples(
        self, r: SplitResult, issues: list, cfg: SplitValidatorConfig
    ) -> None:
        for name, df, minimum in [
            ("train", r.train,      cfg.min_train_samples),
            ("val",   r.validation, cfg.min_val_samples),
            ("test",  r.test,       cfg.min_test_samples),
        ]:
            n = len(df)
            if n < minimum:
                issues.append(SplitIssue(FAIL, f"min_{name}_samples",
                    f"{name} has {n} rows; minimum is {minimum}."))
            else:
                issues.append(SplitIssue(PASS, f"min_{name}_samples",
                    f"{name}: {n} rows ≥ {minimum}."))
