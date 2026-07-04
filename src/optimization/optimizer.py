"""
Optimizer
=========
Manages Optuna studies: creates, resumes, runs, and summarises them.

Key features
------------
* Bayesian optimisation via TPE sampler.
* Optional SQLite backend for crash recovery and resumption.
* Early stopping callback — stops the study when *patience* consecutive
  trials fail to improve the best value by at least *min_delta*.
* Parallel trial execution when n_jobs_trials > 1 (requires SQLite storage).
* Returns an ``OptimizationResult`` with best params, history, and timing.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

logger = logging.getLogger(__name__)


# ── Config ─────────────────────────────────────────────────────────────────────

@dataclass
class OptimizerConfig:
    """Settings for a single Optuna study.

    Attributes:
        n_trials:       Maximum number of trials.
        timeout:        Wall-clock seconds budget per study (None = unlimited).
        direction:      "maximize" (all supported metrics are higher-is-better).
        n_jobs_trials:  Parallel trials.  >1 requires SQLite storage.
        random_seed:    TPE sampler seed.
        early_stopping_patience:  Stop after this many non-improving trials.
        early_stopping_warmup:    Don't apply early stopping before this many trials.
        early_stopping_min_delta: Minimum absolute improvement to reset patience.
        use_pruning:    Whether to use MedianPruner (requires intermediate values).
        storage_dir:    Directory for the SQLite .db file.  None = in-memory.
        resume_if_exists: Load existing study from storage instead of creating new.
    """
    n_trials:                  int            = 50
    timeout:                   Optional[float]= None
    direction:                 str            = "maximize"
    n_jobs_trials:             int            = 1
    random_seed:               int            = 42
    early_stopping_patience:   int            = 20
    early_stopping_warmup:     int            = 10
    early_stopping_min_delta:  float          = 1e-4
    use_pruning:               bool           = False
    storage_dir:               Optional[Path] = None
    resume_if_exists:          bool           = True


# ── Result ─────────────────────────────────────────────────────────────────────

@dataclass
class OptimizationResult:
    """Outcome of a single Optuna study."""
    study_name:           str
    model_name:           str
    metric:               str
    direction:            str
    best_value:           float
    best_params:          dict
    n_trials_completed:   int
    optimization_time_s:  float
    trial_history:        list[dict]  = field(default_factory=list)
    storage_path:         Optional[str] = None

    @property
    def best_trial_number(self) -> Optional[int]:
        if not self.trial_history:
            return None
        key = "value"
        try:
            candidates = [t for t in self.trial_history if t.get(key) is not None]
            if self.direction == "maximize":
                best = max(candidates, key=lambda t: t[key])
            else:
                best = min(candidates, key=lambda t: t[key])
            return best.get("number")
        except (ValueError, TypeError):
            return None


# ── Early stopping callback ───────────────────────────────────────────────────

class EarlyStoppingCallback:
    """Stop the study when validation score stops improving."""

    def __init__(
        self,
        warmup:    int   = 10,
        patience:  int   = 20,
        min_delta: float = 1e-4,
    ) -> None:
        self._warmup    = warmup
        self._patience  = patience
        self._min_delta = min_delta
        self._best:     Optional[float] = None
        self._no_improve = 0

    def __call__(self, study: optuna.Study, trial: optuna.FrozenTrial) -> None:
        completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if len(completed) < self._warmup:
            return

        current_best = study.best_value
        if self._best is None:
            self._best = current_best
            return

        if (current_best - self._best) > self._min_delta:
            self._best       = current_best
            self._no_improve = 0
        else:
            self._no_improve += 1

        if self._no_improve >= self._patience:
            logger.info(
                "Early stopping: no improvement for %d trials (best=%.5f).",
                self._no_improve, self._best,
            )
            study.stop()

    def reset(self) -> None:
        self._best       = None
        self._no_improve = 0


# ── Optimizer ─────────────────────────────────────────────────────────────────

class Optimizer:
    """Creates and runs Optuna studies."""

    def optimize(
        self,
        objective:   callable,
        model_name:  str,
        metric:      str,
        window_number: int,
        config:      OptimizerConfig,
    ) -> OptimizationResult:
        """Run an Optuna study and return the result.

        Args:
            objective:     Callable ``(trial) -> float``.
            model_name:    For logging and study naming.
            metric:        Metric name (for naming and result storage).
            window_number: Walk-forward window index (for study naming).
            config:        Optimizer configuration.

        Returns:
            OptimizationResult with best params and trial history.
        """
        study_name   = f"{model_name}_w{window_number:03d}_{metric}"
        storage, storage_path = self._build_storage(config, study_name)

        pruner  = (optuna.pruners.MedianPruner(n_startup_trials=5)
                   if config.use_pruning else optuna.pruners.NopPruner())
        sampler = optuna.samplers.TPESampler(seed=config.random_seed)

        study = optuna.create_study(
            study_name      = study_name,
            storage         = storage,
            direction       = config.direction,
            sampler         = sampler,
            pruner          = pruner,
            load_if_exists  = config.resume_if_exists,
        )

        early_stop = EarlyStoppingCallback(
            warmup    = config.early_stopping_warmup,
            patience  = config.early_stopping_patience,
            min_delta = config.early_stopping_min_delta,
        )

        t0 = time.monotonic()
        study.optimize(
            objective,
            n_trials  = config.n_trials,
            timeout   = config.timeout,
            n_jobs    = config.n_jobs_trials,
            callbacks = [early_stop],
            show_progress_bar = False,
        )
        elapsed = time.monotonic() - t0

        history = [
            {
                "number": t.number,
                "value":  t.value,
                "state":  t.state.name,
                "params": t.params,
            }
            for t in study.trials
        ]

        completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

        try:
            best_value  = study.best_value
            best_params = study.best_params
        except ValueError:
            best_value  = float("nan")
            best_params = {}

        logger.info(
            "Study '%s': %d/%d trials | best_%s=%.5f | %.1fs",
            study_name, len(completed), config.n_trials,
            metric, best_value, elapsed,
        )

        return OptimizationResult(
            study_name          = study_name,
            model_name          = model_name,
            metric              = metric,
            direction           = config.direction,
            best_value          = best_value,
            best_params         = best_params,
            n_trials_completed  = len(completed),
            optimization_time_s = elapsed,
            trial_history       = history,
            storage_path        = storage_path,
        )

    # ── Private ────────────────────────────────────────────────────────────────

    @staticmethod
    def _build_storage(
        config: OptimizerConfig, study_name: str
    ) -> tuple[Optional[str], Optional[str]]:
        """Return (storage_url, storage_path_str) for the study."""
        if config.storage_dir is None:
            return None, None
        storage_dir = Path(config.storage_dir)
        storage_dir.mkdir(parents=True, exist_ok=True)
        db_path    = storage_dir / "optuna_studies.db"
        storage_url = f"sqlite:///{db_path}"
        return storage_url, str(db_path)
