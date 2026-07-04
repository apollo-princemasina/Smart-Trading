"""
Generate all feature quality reports:
  reports/feature_quality_report.md
  reports/feature_importance.csv
  reports/feature_rankings.csv
  reports/selected_features_top{N}.json
  reports/correlation_matrix.parquet
  reports/vif_report.csv
  reports/psi_report.csv
  reports/drift_report.csv
  reports/leakage_report.md
  reports/shap_summary.parquet
  reports/feature_clusters.json
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_NOW = lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")


class FeatureReportGenerator:
    """
    Generate all feature quality reports from a :class:`FeatureQualityResults` object.

    Parameters
    ----------
    output_dir:
        Directory where all reports are written (default ``reports/``).
    """

    def __init__(self, output_dir: str | Path = "reports"):
        self._dir = Path(output_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    # ── Public entry point ────────────────────────────────────────────────────

    def generate_all(self, results: "FeatureQualityResults") -> dict[str, Path]:  # noqa: F821
        """Generate all report artefacts. Returns dict of {name: path}."""
        paths: dict[str, Path] = {}

        paths["quality_report"]      = self._write_quality_report_md(results)
        paths["feature_importance"]  = self._write_importance_csv(results)
        paths["feature_rankings"]    = self._write_rankings_csv(results)
        paths["leakage_report"]      = self._write_leakage_report_md(results)
        paths.update(self._write_selected_features_json(results))

        if results.correlation_report is not None:
            paths["correlation_matrix"] = self._write_correlation_parquet(results)

        if results.vif_report is not None:
            paths["vif_report"] = self._write_vif_csv(results)

        if results.psi_report is not None:
            paths["psi_report"] = self._write_psi_csv(results)

        if results.drift_report is not None:
            paths["drift_report"] = self._write_drift_csv(results)

        if results.shap_report is not None and results.shap_report.available:
            paths["shap_summary"] = self._write_shap_parquet(results)

        if results.cluster_report is not None:
            paths["feature_clusters"] = self._write_clusters_json(results)

        logger.info("Generated %d report files → %s", len(paths), self._dir)
        return paths

    # ── Individual writers ────────────────────────────────────────────────────

    def _write_quality_report_md(self, r: "FeatureQualityResults") -> Path:
        lines: list[str] = [
            f"# Feature Quality Report — {r.symbol}",
            f"Generated: {_NOW()}  ",
            f"Pipeline version: {r.pipeline_version}  ",
            "",
            "## Summary",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Total features analysed | {len(r.feature_scores)} |",
        ]

        if r.constant_report:
            lines.append(f"| Constant features | {len(r.constant_report.constant_features)} |")
        if r.missing_report:
            flagged_m = len(r.missing_report.flagged_missing)
            lines.append(f"| High-missing features (> threshold) | {flagged_m} |")
        if r.duplicate_report:
            lines.append(f"| Duplicate features | {len(r.duplicate_report.features_to_drop)} |")
        if r.leakage_report:
            lines.append(f"| Leakage-flagged features | {len(r.leakage_report.flagged_features)} |")
        if r.drift_report:
            lines.append(f"| Drifted features | {len(r.drift_report.drifted_features)} |")
        if r.selection_result:
            lines.append(f"| Final selected features | {len(r.selection_result.selected_features)} |")

        lines += [
            "",
            "## Feature Rankings (top 20)",
            "",
            "| Rank | Feature | Composite | Quality | Importance | Stability | Leakage | Drift | Flags |",
            "|------|---------|-----------|---------|------------|-----------|---------|-------|-------|",
        ]

        ranked = sorted(
            r.feature_scores.values(),
            key=lambda s: s.composite_score,
            reverse=True,
        )
        for fs in ranked[:20]:
            flags = ", ".join(fs.flags) if fs.flags else "—"
            lines.append(
                f"| {fs.rank} | `{fs.name}` "
                f"| {fs.composite_score:.1f} "
                f"| {fs.quality_score:.1f} "
                f"| {fs.importance_score:.1f} "
                f"| {fs.stability_score:.1f} "
                f"| {fs.leakage_score:.1f} "
                f"| {fs.drift_score:.1f} "
                f"| {flags} |"
            )

        if r.selection_result:
            lines += [
                "",
                "## Selected Features",
                "",
                f"**Strategy**: {r.selection_result.strategy}  ",
                f"**Count**: {len(r.selection_result.selected_features)}  ",
                "",
            ]
            for f in r.selection_result.selected_features[:50]:
                lines.append(f"- `{f}`")

        path = self._dir / "feature_quality_report.md"
        path.write_text("\n".join(lines), encoding="utf-8")
        return path

    def _write_importance_csv(self, r: "FeatureQualityResults") -> Path:
        rows = []
        for name, fs in r.feature_scores.items():
            rows.append({
                "feature":           name,
                "composite_score":   fs.composite_score,
                "importance_score":  fs.importance_score,
                "quality_score":     fs.quality_score,
                "stability_score":   fs.stability_score,
                "leakage_score":     fs.leakage_score,
                "drift_score":       fs.drift_score,
                "rank":              fs.rank,
            })
        df   = pd.DataFrame(rows).sort_values("rank")
        path = self._dir / "feature_importance.csv"
        df.to_csv(path, index=False)
        return path

    def _write_rankings_csv(self, r: "FeatureQualityResults") -> Path:
        rows = []
        for name, fs in r.feature_scores.items():
            row: dict = {
                "rank":    fs.rank,
                "feature": name,
                "composite_score": fs.composite_score,
                "quality_score":   fs.quality_score,
                "importance_score": fs.importance_score,
                "stability_score":  fs.stability_score,
                "leakage_score":    fs.leakage_score,
                "drift_score":      fs.drift_score,
                "flags": "|".join(fs.flags),
            }
            # Add raw metrics if available
            if r.missing_report is not None and name in r.missing_report.missing_rates.index:
                row["missing_rate"] = round(float(r.missing_report.missing_rates[name]), 4)
            if r.vif_report is not None and name in r.vif_report.vif_scores.index:
                row["vif"] = round(float(r.vif_report.vif_scores[name]), 2)
            if r.psi_report is not None and name in r.psi_report.psi_scores.index:
                row["psi"] = round(float(r.psi_report.psi_scores[name]), 4)
            rows.append(row)

        df   = pd.DataFrame(rows).sort_values("rank")
        path = self._dir / "feature_rankings.csv"
        df.to_csv(path, index=False)
        return path

    def _write_leakage_report_md(self, r: "FeatureQualityResults") -> Path:
        lines = [
            f"# Leakage Report — {r.symbol}",
            f"Generated: {_NOW()}",
            "",
            "## Flagged Features",
            "",
        ]
        lr = r.leakage_report
        if lr is None or not lr.flagged_features:
            lines.append("No leakage detected.")
        else:
            lines += [
                "| Feature | Score | Type | Details |",
                "|---------|-------|------|---------|",
            ]
            for f in lr.flagged_features:
                score   = round(float(lr.leakage_scores.get(f, 0.0)), 3)
                ltype   = lr.leakage_types.get(f, "unknown")
                det_d   = lr.details.get(f, {})
                det_str = ", ".join(f"{k}={v}" for k, v in list(det_d.items())[:3])
                lines.append(f"| `{f}` | {score} | {ltype} | {det_str} |")

        path = self._dir / "leakage_report.md"
        path.write_text("\n".join(lines), encoding="utf-8")
        return path

    def _write_selected_features_json(self, r: "FeatureQualityResults") -> dict[str, Path]:
        paths: dict[str, Path] = {}
        if r.selection_result is None:
            return paths

        # Per top-N
        for n, feats in r.selection_result.top_n.items():
            payload = {
                "symbol":   r.symbol,
                "strategy": r.selection_result.strategy,
                "top_n":    n,
                "features": feats,
                "count":    len(feats),
            }
            fname = f"selected_features_top{n}.json"
            p     = self._dir / fname
            p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            paths[f"selected_top{n}"] = p

        # Full selected set
        all_sel = {
            "symbol":   r.symbol,
            "strategy": r.selection_result.strategy,
            "features": r.selection_result.selected_features,
            "count":    len(r.selection_result.selected_features),
        }
        p = self._dir / "selected_features_all.json"
        p.write_text(json.dumps(all_sel, indent=2), encoding="utf-8")
        paths["selected_all"] = p
        return paths

    def _write_correlation_parquet(self, r: "FeatureQualityResults") -> Path:
        path = self._dir / "correlation_matrix.parquet"
        r.correlation_report.pearson.to_parquet(path, engine="pyarrow", index=True)
        return path

    def _write_vif_csv(self, r: "FeatureQualityResults") -> Path:
        df = pd.DataFrame({
            "feature":   r.vif_report.vif_scores.index,
            "vif":       r.vif_report.vif_scores.values,
            "tolerance": r.vif_report.tolerance.values,
        })
        path = self._dir / "vif_report.csv"
        df.sort_values("vif", ascending=False).to_csv(path, index=False)
        return path

    def _write_psi_csv(self, r: "FeatureQualityResults") -> Path:
        df = pd.DataFrame({
            "feature": r.psi_report.psi_scores.index,
            "psi":     r.psi_report.psi_scores.values,
            "label":   r.psi_report.psi_labels.values,
        })
        path = self._dir / "psi_report.csv"
        df.sort_values("psi", ascending=False).to_csv(path, index=False)
        return path

    def _write_drift_csv(self, r: "FeatureQualityResults") -> Path:
        df = pd.DataFrame({
            "feature":      r.drift_report.ks_statistics.index,
            "ks_statistic": r.drift_report.ks_statistics.values,
            "ks_pvalue":    r.drift_report.ks_pvalues.values,
            "js_distance":  r.drift_report.js_distances.values,
            "psi":          r.drift_report.psi_scores.values,
            "drift_label":  r.drift_report.drift_labels.values,
        })
        path = self._dir / "drift_report.csv"
        df.sort_values("ks_statistic", ascending=False).to_csv(path, index=False)
        return path

    def _write_shap_parquet(self, r: "FeatureQualityResults") -> Path:
        path = self._dir / "shap_summary.parquet"
        shap_df = pd.DataFrame({
            "feature":       r.shap_report.feature_names,
            "mean_abs_shap": r.shap_report.mean_abs_shap.reindex(
                r.shap_report.feature_names
            ).fillna(0.0).values,
        }).sort_values("mean_abs_shap", ascending=False)
        shap_df.to_parquet(path, engine="pyarrow", index=False)
        return path

    def _write_clusters_json(self, r: "FeatureQualityResults") -> Path:
        payload = {
            "n_clusters": r.cluster_report.n_clusters,
            "clusters": {
                str(cid): {
                    "features":       members,
                    "representative": r.cluster_report.cluster_representatives.get(cid, ""),
                    "size":           r.cluster_report.cluster_sizes.get(cid, 0),
                }
                for cid, members in r.cluster_report.clusters.items()
            },
        }
        path = self._dir / "feature_clusters.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path
