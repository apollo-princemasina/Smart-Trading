"""
Validation Pipeline
===================
Top-level orchestrator that ties everything together.

Pipeline steps
--------------
1. Walk-forward validation  — evaluate every window for every model
2. Stability analysis       — window-to-window metric consistency
3. Robustness analysis      — aggregate statistics + generalization checks
4. Model ranking            — composite score across 7 dimensions
5. Acceptance decision      — Production Ready / Needs Improvement / Rejected
6. Report generation        — 6 output files

Model Acceptance Thresholds
---------------------------
Production Ready   — all required metrics above threshold AND stability ≥ 0.7
Needs Improvement  — some metrics below threshold OR stability < 0.7
Rejected           — critical metrics (F1 or directional accuracy) well below threshold

Model Ranking Score
-------------------
0.20 × generalization_score
0.20 × consistency_score (1 - CV_f1)
0.20 × trading_accuracy (directional_accuracy)
0.20 × F1 (mean across windows)
0.10 × ROC-AUC (mean across windows)
0.05 × inference_speed (1 / mean_inference_time, normalized)
0.05 × 1.0 (memory usage — placeholder)
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .robustness import RobustnessAnalyzer, RobustnessResult, GeneralizationAnalysis
from .stability  import StabilityAnalyzer, StabilityResult
from .validator  import WindowValidationResult
from .walk_forward_validator import WalkForwardValidator, WalkForwardValidationResult
from .reports import generate_all_reports

logger = logging.getLogger(__name__)
_NaN = float("nan")

# ── Acceptance status constants ────────────────────────────────────────────────
PRODUCTION_READY  = "production_ready"
NEEDS_IMPROVEMENT = "needs_improvement"
REJECTED          = "rejected"

# ── Config ─────────────────────────────────────────────────────────────────────

@dataclass
class ValidationConfig:
    """All settings for a validation pipeline run.

    Attributes:
        windows_dir:               Walk-forward window directories.
        models_dir:                Model bundle root.
        output_dir:                Where to write validation_results/.
        target_column:             Label column name.
        task_type:                 "auto", "classification", or "regression".
        model_names:               Restrict to these models. None = all discovered.
        min_accuracy:              Minimum acceptable accuracy.
        min_f1:                    Minimum acceptable F1.
        min_directional_accuracy:  Minimum acceptable directional accuracy.
        min_trading_accuracy:      Minimum acceptable trading accuracy (TP precision).
        max_variance:              Maximum acceptable CV of the primary metric.
        stability_threshold:       Minimum stability score for PRODUCTION_READY.
        overfitting_threshold:     train-test gap above which overfitting is flagged.
        skip_on_error:             Skip window/model errors instead of raising.
        report_dir:                Separate dir for reports (defaults to output_dir).
        log_path:                  Optional file log path.
        symbol:                    Instrument symbol (metadata only).
    """
    windows_dir:              Path
    models_dir:               Path
    output_dir:               Path
    target_column:            str
    task_type:                str                 = "auto"
    model_names:              Optional[list[str]] = None
    min_accuracy:             float               = 0.50
    min_f1:                   float               = 0.40
    min_directional_accuracy: float               = 0.50
    min_trading_accuracy:     float               = 0.45
    max_variance:             float               = 0.25
    stability_threshold:      float               = 0.65
    overfitting_threshold:    float               = 0.15
    skip_on_error:            bool                = True
    report_dir:               Optional[Path]      = None
    log_path:                 Optional[Path]      = None
    symbol:                   str                 = ""


# ── Per-model result ──────────────────────────────────────────────────────────

@dataclass
class ModelValidationResult:
    """Complete validation outcome for one model."""
    model_name:         str
    window_results:     list[WindowValidationResult]
    stability:          StabilityResult
    robustness:         RobustnessResult
    ranking_score:      float
    acceptance_status:  str
    acceptance_reasons: list[str]
    mean_inference_ms:  float   # mean per-sample inference time in ms
    task_type:          str


# ── Pipeline result ────────────────────────────────────────────────────────────

@dataclass
class ValidationPipelineResult:
    """Full pipeline output."""
    model_results:     list[ModelValidationResult]
    ranked_models:     list[str]              # model names, best first
    errors:            list[str]
    report_paths:      dict[str, str]
    total_time_s:      float
    n_windows:         int
    n_models:          int
    overall_summary:   dict                   # written to overall_summary.json


# ── Pipeline ──────────────────────────────────────────────────────────────────

class ValidationPipeline:
    """End-to-end walk-forward validation pipeline."""

    def run(self, config: ValidationConfig) -> ValidationPipelineResult:
        """Execute the full validation pipeline.

        Args:
            config: ValidationConfig with all settings.

        Returns:
            ValidationPipelineResult with all outcomes, ranked models, and
            report file paths.
        """
        _setup_logging(config.log_path)
        t_start = time.monotonic()
        logger.info("Starting Walk-Forward Validation Pipeline")

        config.output_dir = Path(config.output_dir)
        config.output_dir.mkdir(parents=True, exist_ok=True)

        report_dir = Path(config.report_dir) if config.report_dir else config.output_dir
        report_dir.mkdir(parents=True, exist_ok=True)

        # ── Step 1: Walk-forward evaluation ───────────────────────────────────
        wf_validator = WalkForwardValidator(skip_on_error=config.skip_on_error)
        wf_result    = wf_validator.validate(
            windows_dir   = Path(config.windows_dir),
            models_dir    = Path(config.models_dir),
            target_column = config.target_column,
            model_names   = config.model_names,
        )
        errors = list(wf_result.errors)

        if not wf_result.model_results:
            logger.warning("No model results produced — check windows_dir and models_dir")

        # ── Step 2: Detect task type ───────────────────────────────────────────
        task_type = _detect_task_type(wf_result, config.task_type)

        # ── Steps 3-5: Stability + Robustness + Acceptance per model ──────────
        model_validation_results: list[ModelValidationResult] = []
        for model_name, window_results in wf_result.model_results.items():
            try:
                mvr = self._analyze_model(
                    model_name     = model_name,
                    window_results = window_results,
                    task_type      = task_type,
                    config         = config,
                )
                model_validation_results.append(mvr)
            except Exception as exc:
                msg = f"Analysis failed for {model_name}: {exc}"
                logger.error(msg, exc_info=True)
                errors.append(msg)

        # ── Step 6: Rank models ────────────────────────────────────────────────
        ranked = sorted(
            model_validation_results,
            key=lambda m: m.ranking_score,
            reverse=True,
        )
        ranked_names = [m.model_name for m in ranked]

        # ── Step 7: Save per-window results to output_dir ─────────────────────
        self._save_window_results(wf_result, config.output_dir)

        # ── Step 8: Build overall_summary.json ────────────────────────────────
        summary = _build_summary(ranked, wf_result, task_type, config)
        summary_path = config.output_dir / "overall_summary.json"
        summary_path.write_text(
            json.dumps(summary, indent=2, default=_json_default),
            encoding="utf-8",
        )
        logger.info("Written overall_summary.json → %s", summary_path)

        # ── Step 9: Generate reports ───────────────────────────────────────────
        pipeline_result = ValidationPipelineResult(
            model_results   = ranked,
            ranked_models   = ranked_names,
            errors          = errors,
            report_paths    = {},
            total_time_s    = time.monotonic() - t_start,
            n_windows       = wf_result.n_windows,
            n_models        = wf_result.n_models,
            overall_summary = summary,
        )
        try:
            paths = generate_all_reports(pipeline_result, report_dir, task_type)
            pipeline_result.report_paths = {k: str(v) for k, v in paths.items()}
        except Exception as exc:
            logger.error("Report generation failed: %s", exc, exc_info=True)

        logger.info(
            "Validation complete: %d models, %d windows, %.1fs | ranked: %s",
            len(ranked), wf_result.n_windows, pipeline_result.total_time_s,
            ranked_names[:3],
        )
        return pipeline_result

    # ── Per-model analysis ─────────────────────────────────────────────────────

    def _analyze_model(
        self,
        model_name:     str,
        window_results: list[WindowValidationResult],
        task_type:      str,
        config:         ValidationConfig,
    ) -> ModelValidationResult:
        # Filter out failed windows for metric computation (keep them in list)
        good_results = [r for r in window_results if r.error is None]
        metric_dicts = [r.combined_metrics for r in good_results]

        # ── Stability ──────────────────────────────────────────────────────────
        stab_analyzer = StabilityAnalyzer(
            task_type = task_type,
            threshold = config.stability_threshold,
        )
        stability = stab_analyzer.analyze(metric_dicts, model_name)

        # ── Robustness ─────────────────────────────────────────────────────────
        train_histories = [r.bundle_train_metrics for r in good_results
                           if r.bundle_train_metrics]
        if len(train_histories) != len(good_results):
            train_histories = None

        rob_analyzer = RobustnessAnalyzer(
            task_type             = task_type,
            overfitting_threshold = config.overfitting_threshold,
            min_acceptable        = config.min_f1 if task_type == "classification" else 0.0,
        )
        robustness = rob_analyzer.analyze(good_results, model_name, train_histories)

        # ── Ranking score ──────────────────────────────────────────────────────
        ranking_score = _compute_ranking_score(
            stability  = stability,
            robustness = robustness,
            good_results = good_results,
            task_type  = task_type,
        )

        # ── Acceptance decision ────────────────────────────────────────────────
        status, reasons = _determine_acceptance(
            stability   = stability,
            robustness  = robustness,
            metric_dicts = metric_dicts,
            task_type   = task_type,
            config      = config,
        )

        # ── Mean inference speed ───────────────────────────────────────────────
        inf_times = [r.inference_time_s for r in good_results
                     if r.inference_time_s > 0 and r.n_test > 0]
        mean_inf_ms = _NaN
        if inf_times and any(r.n_test > 0 for r in good_results):
            ns = [r.n_test for r in good_results if r.n_test > 0]
            mean_time_per_sample = float(np.mean(
                [t / n for t, n in zip(inf_times, ns) if n > 0]
            ))
            mean_inf_ms = mean_time_per_sample * 1000

        return ModelValidationResult(
            model_name        = model_name,
            window_results    = window_results,
            stability         = stability,
            robustness        = robustness,
            ranking_score     = ranking_score,
            acceptance_status = status,
            acceptance_reasons = reasons,
            mean_inference_ms = mean_inf_ms,
            task_type         = task_type,
        )

    # ── Save per-window results ────────────────────────────────────────────────

    @staticmethod
    def _save_window_results(
        wf_result: WalkForwardValidationResult,
        output_dir: Path,
    ) -> None:
        for model_name, results in wf_result.model_results.items():
            for r in results:
                win_dir = output_dir / f"window_{r.window_number:03d}"
                win_dir.mkdir(parents=True, exist_ok=True)
                out_path = win_dir / f"{model_name}.json"
                payload  = _result_to_dict(r)
                out_path.write_text(
                    json.dumps(payload, indent=2, default=_json_default),
                    encoding="utf-8",
                )


# ── Scoring functions ──────────────────────────────────────────────────────────

def _compute_ranking_score(
    stability:    StabilityResult,
    robustness:   RobustnessResult,
    good_results: list[WindowValidationResult],
    task_type:    str,
) -> float:
    """Composite ranking score in [0, 1].

    Weights:
        0.20 generalization
        0.20 consistency (stability)
        0.20 directional / trading accuracy
        0.20 F1 (or R²)
        0.10 ROC-AUC (classification only)
        0.05 inference speed
        0.05 placeholder (memory / model size)
    """
    gen_score   = robustness.generalization.generalization_score
    cons_score  = stability.stability_score

    prim_key    = "f1" if task_type == "classification" else "r2"
    prim_stats  = robustness.metric_stats.get(prim_key, {})
    prim_mean   = float(prim_stats.get("mean", 0) or 0)

    da_stats    = robustness.metric_stats.get("directional_accuracy", {})
    da_mean     = float(da_stats.get("mean", 0) or 0) if task_type == "classification" else prim_mean

    auc_stats   = robustness.metric_stats.get("roc_auc", {})
    auc_mean    = float(auc_stats.get("mean", 0) or 0) if task_type == "classification" else 0.5

    # Inference speed score: normalize by a 10 ms/sample ceiling
    inf_times   = [r.inference_time_s / r.n_test * 1000
                   for r in good_results if r.n_test > 0 and r.inference_time_s >= 0]
    if inf_times:
        mean_ms   = float(np.mean(inf_times))
        inf_score = float(max(0.0, 1.0 - mean_ms / 10.0))
    else:
        inf_score = 0.5

    score = (
        0.20 * max(0.0, gen_score)
        + 0.20 * max(0.0, cons_score)
        + 0.20 * max(0.0, da_mean)
        + 0.20 * max(0.0, prim_mean)
        + 0.10 * max(0.0, auc_mean)
        + 0.05 * inf_score
        + 0.05 * 1.0   # memory placeholder
    )
    return round(float(score), 6)


def _determine_acceptance(
    stability:    StabilityResult,
    robustness:   RobustnessResult,
    metric_dicts: list[dict],
    task_type:    str,
    config:       ValidationConfig,
) -> tuple[str, list[str]]:
    """Return (acceptance_status, list_of_reasons)."""
    reasons: list[str] = []

    if not metric_dicts:
        return REJECTED, ["No successful window evaluations"]

    def _mean(key: str) -> float:
        vals = [float(d.get(key, _NaN)) for d in metric_dicts]
        valid = [v for v in vals if v == v]
        return float(np.mean(valid)) if valid else _NaN

    # ── Check thresholds ───────────────────────────────────────────────────────
    acc = _mean("accuracy")
    f1  = _mean("f1")
    da  = _mean("directional_accuracy")
    ta  = _mean("tp_prediction_accuracy")
    cv  = stability.metric_cvs.get("f1", _NaN) if task_type == "classification" else \
          stability.metric_cvs.get("r2", _NaN)
    r2  = _mean("r2") if task_type == "regression" else _NaN

    failures, warnings = [], []

    if task_type == "classification":
        if acc  == acc and acc  < config.min_accuracy:
            failures.append(f"accuracy={acc:.4f} < threshold={config.min_accuracy}")
        if f1   == f1  and f1   < config.min_f1:
            failures.append(f"f1={f1:.4f} < threshold={config.min_f1}")
        if da   == da  and da   < config.min_directional_accuracy:
            failures.append(f"directional_accuracy={da:.4f} < threshold={config.min_directional_accuracy}")
        if ta   == ta  and ta   < config.min_trading_accuracy:
            warnings.append(f"tp_accuracy={ta:.4f} < threshold={config.min_trading_accuracy}")
    else:
        if r2 == r2 and r2 < 0:
            failures.append(f"r2={r2:.4f} < 0 (worse than mean prediction)")

    # Variance check
    if cv == cv and cv > config.max_variance:
        warnings.append(f"primary_metric CV={cv:.4f} > max={config.max_variance} (unstable)")

    # Stability check
    if not stability.is_stable:
        warnings.append(f"stability_score={stability.stability_score:.4f} < threshold={config.stability_threshold}")

    # Generalization flags
    gen = robustness.generalization
    if gen.overfitting_detected:
        warnings.append(f"Overfitting detected (train-test gap={gen.train_test_gap:.4f})")
    if gen.underfitting_detected:
        failures.append("Underfitting detected (performance below minimum threshold)")
    if gen.performance_degradation:
        warnings.append(f"Performance degradation across windows (slope={gen.degradation_slope:.4f})")

    reasons = failures + warnings

    # ── Determine status ───────────────────────────────────────────────────────
    if failures:
        # Critical failures: check if they are catastrophic
        critical = [r for r in failures if "f1" in r or "directional_accuracy" in r
                    or "underfitting" in r.lower()]
        if critical:
            return REJECTED, reasons
        return NEEDS_IMPROVEMENT, reasons

    if warnings:
        return NEEDS_IMPROVEMENT, reasons

    return PRODUCTION_READY, reasons


# ── Helpers ───────────────────────────────────────────────────────────────────

def _detect_task_type(
    wf_result: WalkForwardValidationResult,
    config_type: str,
) -> str:
    if config_type != "auto":
        return config_type
    for results in wf_result.model_results.values():
        for r in results:
            if r.task_type in ("classification", "regression"):
                return r.task_type
    return "classification"


def _build_summary(
    ranked:    list[ModelValidationResult],
    wf_result: WalkForwardValidationResult,
    task_type: str,
    config:    ValidationConfig,
) -> dict:
    model_summaries = []
    for mvr in ranked:
        prod   = mvr.robustness.metric_stats.get("f1" if task_type == "classification" else "r2", {})
        ms = {
            "model_name":        mvr.model_name,
            "acceptance_status": mvr.acceptance_status,
            "ranking_score":     mvr.ranking_score,
            "stability_score":   mvr.stability.stability_score,
            "robustness_score":  mvr.robustness.robustness_score,
            "generalization_score": mvr.robustness.generalization.generalization_score,
            "primary_metric_mean": prod.get("mean"),
            "primary_metric_std":  prod.get("std"),
            "n_windows":         mvr.stability.n_windows,
            "mean_inference_ms": mvr.mean_inference_ms,
            "acceptance_reasons": mvr.acceptance_reasons,
        }
        model_summaries.append(ms)

    return {
        "task_type":      task_type,
        "n_windows":      wf_result.n_windows,
        "n_models":       wf_result.n_models,
        "target_column":  config.target_column,
        "symbol":         config.symbol,
        "models":         model_summaries,
        "ranked_models":  [m["model_name"] for m in model_summaries],
        "best_model":     model_summaries[0]["model_name"] if model_summaries else None,
        "thresholds": {
            "min_accuracy":             config.min_accuracy,
            "min_f1":                   config.min_f1,
            "min_directional_accuracy": config.min_directional_accuracy,
            "min_trading_accuracy":     config.min_trading_accuracy,
            "max_variance":             config.max_variance,
            "stability_threshold":      config.stability_threshold,
        },
        "errors": wf_result.errors,
    }


def _result_to_dict(r: WindowValidationResult) -> dict:
    return {
        "window_number":          r.window_number,
        "model_name":             r.model_name,
        "task_type":              r.task_type,
        "classification_metrics": r.classification_metrics,
        "regression_metrics":     r.regression_metrics,
        "trading_metrics":        _strip_non_serializeable(r.trading_metrics),
        "combined_metrics":       r.combined_metrics,
        "inference_time_s":       r.inference_time_s,
        "n_test":                 r.n_test,
        "error":                  r.error,
    }


def _strip_non_serializeable(d: dict) -> dict:
    return {k: v for k, v in d.items() if isinstance(v, (str, int, float, bool, type(None)))}


def _json_default(obj):
    import numpy as np
    if isinstance(obj, np.integer):  return int(obj)
    if isinstance(obj, np.floating): return None if obj != obj else float(obj)
    if isinstance(obj, np.ndarray):  return obj.tolist()
    return str(obj)


def _setup_logging(log_path: Optional[Path]) -> None:
    if log_path is None:
        return
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    h = logging.FileHandler(log_path, encoding="utf-8")
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    logging.getLogger("src.validation").addHandler(h)
