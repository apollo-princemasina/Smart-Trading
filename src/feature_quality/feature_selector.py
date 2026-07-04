"""
Aggregate feature selection: combines multiple selection methods into a
final consensus feature set.

Strategies
----------
* ``intersection`` — only features selected by ALL active methods
* ``union``        — features selected by ANY active method
* ``voting``       — features selected by ≥ *min_votes* methods
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd


SelectionStrategy = Literal["intersection", "union", "voting"]


@dataclass
class SelectionResult:
    selected_features:     list[str]
    method_votes:          dict[str, list[str]]   # method → its selected features
    vote_counts:           pd.Series              # feature → n methods that selected it
    strategy:              str
    top_n:                 dict[int, list[str]]   # {25: [...], 50: [...], ...}


class FeatureSelector:
    """
    Combine feature-importance and selection signals from multiple methods
    into a final feature set.

    Parameters
    ----------
    strategy:
        ``"intersection"`` (default), ``"union"``, or ``"voting"``.
    min_votes:
        Minimum number of methods that must select a feature (used only when
        *strategy* is ``"voting"``; default 2).
    top_n_counts:
        A list of top-N cutoffs to also export (default [25, 50, 75, 100, 150]).
    """

    def __init__(
        self,
        strategy:      SelectionStrategy = "intersection",
        min_votes:     int               = 2,
        top_n_counts:  list[int]         = None,
    ):
        self._strategy    = strategy
        self._min_votes   = min_votes
        self._top_n       = top_n_counts or [25, 50, 75, 100, 150]

    def select(
        self,
        composite_scores:     pd.Series,
        method_selections:    dict[str, list[str]] | None = None,
        exclude_features:     list[str] | None            = None,
    ) -> SelectionResult:
        """
        Determine the final selected feature set.

        Parameters
        ----------
        composite_scores:
            Per-feature composite quality score (higher = better).  Used to
            build ``top_n`` subsets regardless of selection strategy.
        method_selections:
            Optional dict of ``{method_name: [selected_features]}``.
            If None, selection falls back to top-N by composite score.
        exclude_features:
            Features to forcibly exclude (constant, leaky, …).
        """
        exclude = set(exclude_features or [])
        all_feats = [f for f in composite_scores.index if f not in exclude]

        votes = pd.Series(0, index=all_feats, dtype=int)
        method_votes: dict[str, list[str]] = {}

        if method_selections:
            for method, selected in method_selections.items():
                valid = [f for f in selected if f in votes.index]
                method_votes[method] = valid
                for f in valid:
                    votes[f] += 1

        # ── Apply strategy ────────────────────────────────────────────────────
        n_methods = len(method_votes) if method_votes else 0

        if method_votes and self._strategy == "intersection":
            selected = [f for f in all_feats if votes[f] == n_methods]
        elif method_votes and self._strategy == "voting":
            selected = [f for f in all_feats if votes[f] >= self._min_votes]
        elif method_votes and self._strategy == "union":
            selected = [f for f in all_feats if votes[f] > 0]
        else:
            # No method selections supplied → top-50 by composite score
            selected = list(
                composite_scores.reindex(all_feats)
                .sort_values(ascending=False)
                .head(50)
                .index
            )

        # Sort by composite score descending
        scored = composite_scores.reindex(selected).fillna(0).sort_values(ascending=False)
        selected = list(scored.index)

        # Top-N subsets
        top_n_dict: dict[int, list[str]] = {}
        scored_all  = composite_scores.reindex(all_feats).fillna(0).sort_values(ascending=False)
        for n in self._top_n:
            top_n_dict[n] = list(scored_all.head(n).index)

        return SelectionResult(
            selected_features  = selected,
            method_votes       = method_votes,
            vote_counts        = votes,
            strategy           = self._strategy,
            top_n              = top_n_dict,
        )

    def select_from_report(
        self,
        results: "FeatureQualityResults",  # noqa: F821
        strategy: SelectionStrategy | None = None,
    ) -> SelectionResult:
        """
        Convenience: extract selections from a :class:`FeatureQualityResults`
        object and apply the configured (or overridden) strategy.
        """
        from .feature_quality import FeatureQualityResults

        if strategy is not None:
            self._strategy = strategy

        composite = pd.Series(
            {name: s.composite_score for name, s in results.feature_scores.items()}
        )

        # Gather per-method selections
        method_sels: dict[str, list[str]] = {}
        if results.importance_report is not None:
            top_imp = list(results.importance_report.top_features[:100])
            method_sels["tree_importance"] = top_imp

        if results.shap_report is not None and results.shap_report.available:
            shap_top = list(
                results.shap_report.mean_abs_shap
                .sort_values(ascending=False)
                .head(100)
                .index
            )
            method_sels["shap"] = shap_top

        if results.mi_report is not None:
            method_sels["mutual_info"] = list(results.mi_report.top_features[:100])

        if results.boruta_report is not None:
            method_sels["boruta"] = list(results.boruta_report.accepted)

        if results.rfe_report is not None:
            method_sels["rfe"] = list(results.rfe_report.selected_features)

        exclude = list(set(
            (results.constant_report.constant_features if results.constant_report else [])
            + (results.duplicate_report.features_to_drop if results.duplicate_report else [])
            + (results.leakage_report.flagged_features if results.leakage_report else [])
        ))

        return self.select(composite, method_sels or None, exclude)
