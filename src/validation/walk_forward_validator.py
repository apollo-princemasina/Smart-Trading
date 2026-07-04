"""
Walk-Forward Validator
======================
Orchestrates per-window validation across all discovered windows for every
available model bundle.

Bundle Discovery
----------------
Expected directory layout (from the optimization pipeline)::

    models_dir/
      window_001/
        xgboost/bundle/
        lightgbm/bundle/
        random_forest/bundle/
        ...
      window_002/
        ...
      best_model/               ← optional single best bundle

windows_dir/
  window_001/
    train.parquet
    validation.parquet
    test.parquet
    metadata.json
  window_002/
    ...

Each window's test split is evaluated against the corresponding per-window
bundles.  Every window remains completely independent.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

from .validator import WindowValidationResult, WindowValidator

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardValidationResult:
    """All per-window results for every model."""
    # {model_name: [WindowValidationResult, ...]}  ordered by window number
    model_results: dict[str, list[WindowValidationResult]] = field(default_factory=dict)
    errors:        list[str]                                = field(default_factory=list)
    n_windows:     int                                      = 0
    n_models:      int                                      = 0


class WalkForwardValidator:
    """Run walk-forward validation across all windows and all models.

    Attributes:
        skip_on_error: If True, log errors and continue.  If False, raise.
    """

    def __init__(self, skip_on_error: bool = True) -> None:
        self.skip_on_error  = skip_on_error
        self._validator     = WindowValidator()

    def validate(
        self,
        windows_dir:   Path,
        models_dir:    Path,
        target_column: str,
        model_names:   Optional[list[str]] = None,
    ) -> WalkForwardValidationResult:
        """Run the full walk-forward validation.

        Args:
            windows_dir:   Root of the walk-forward window directories.
            models_dir:    Root of the model bundle directories.
            target_column: Label column name.
            model_names:   Restrict to these model names.  None = all discovered.

        Returns:
            WalkForwardValidationResult grouped by model.
        """
        windows_dir = Path(windows_dir)
        models_dir  = Path(models_dir)

        window_dirs = self._discover_windows(windows_dir)
        logger.info("Discovered %d walk-forward windows in %s", len(window_dirs), windows_dir)

        if not window_dirs:
            return WalkForwardValidationResult(
                errors=[f"No window directories found in {windows_dir}"]
            )

        # Discover all models that have bundles (use first window as reference)
        if model_names is None:
            model_names = self._discover_models(models_dir, window_dirs[0])
        logger.info("Validating %d models: %s", len(model_names), model_names)

        model_results: dict[str, list[WindowValidationResult]] = {m: [] for m in model_names}
        errors: list[str] = []

        for win_dir in window_dirs:
            wnum = _parse_window_number(win_dir)
            test_df = self._load_test_split(win_dir)
            if test_df is None:
                msg = f"Could not load test split for window {wnum}"
                logger.error(msg)
                errors.append(msg)
                continue

            for model_name in model_names:
                bundle_dir = models_dir / f"window_{wnum:03d}" / model_name / "bundle"
                if not bundle_dir.exists():
                    msg = f"Bundle not found: {bundle_dir}"
                    logger.warning(msg)
                    errors.append(msg)
                    # Append an error result so windows stay aligned
                    model_results[model_name].append(
                        WindowValidationResult(
                            window_number=wnum,
                            model_name=model_name,
                            task_type="unknown",
                            error=msg,
                            bundle_dir=bundle_dir,
                        )
                    )
                    continue

                try:
                    result = self._validator.validate(
                        bundle_dir    = bundle_dir,
                        test_df       = test_df,
                        target_column = target_column,
                        window_number = wnum,
                    )
                    # Propagate error results so callers can detect them
                    if result.error is not None:
                        errors.append(f"Window {wnum} / {model_name}: {result.error}")
                    model_results[model_name].append(result)
                except Exception as exc:
                    msg = f"Window {wnum} / {model_name}: {exc}"
                    logger.error(msg, exc_info=True)
                    errors.append(msg)
                    if not self.skip_on_error:
                        raise
                    model_results[model_name].append(
                        WindowValidationResult(
                            window_number=wnum,
                            model_name=model_name,
                            task_type="unknown",
                            error=msg,
                            bundle_dir=bundle_dir,
                        )
                    )

        return WalkForwardValidationResult(
            model_results = model_results,
            errors        = errors,
            n_windows     = len(window_dirs),
            n_models      = len(model_names),
        )

    # ── Static helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _discover_windows(windows_dir: Path) -> list[Path]:
        if not windows_dir.exists():
            return []
        return sorted(
            p for p in windows_dir.iterdir()
            if p.is_dir() and p.name.startswith("window_")
        )

    @staticmethod
    def _discover_models(models_dir: Path, first_window_dir: Path) -> list[str]:
        """Return model names found in first window's models subdirectory."""
        wnum = _parse_window_number(first_window_dir)
        win_model_dir = models_dir / f"window_{wnum:03d}"
        if not win_model_dir.exists():
            return []
        return sorted(
            p.name for p in win_model_dir.iterdir()
            if p.is_dir() and (p / "bundle").exists()
        )

    @staticmethod
    def _load_test_split(win_dir: Path) -> Optional[pd.DataFrame]:
        test_path = win_dir / "test.parquet"
        if not test_path.exists():
            return None
        try:
            return pd.read_parquet(test_path)
        except Exception as exc:
            logger.error("Could not read %s: %s", test_path, exc)
            return None


def _parse_window_number(win_dir: Path) -> int:
    try:
        return int(win_dir.name.split("_")[-1])
    except ValueError:
        return 0
