"""Run the trained model bundle and return class predictions.

Typical usage
-------------
    from src.inference.predictor import predict

    preds, probas = predict(feature_df)
    # preds:  np.ndarray of int  (0=SELL, 1=HOLD, 2=BUY)
    # probas: np.ndarray shape (n_rows, 3)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_BASE_DIR   = Path(__file__).resolve().parents[2]
_BUNDLE_DIR = _BASE_DIR / "models" / "best_model"

CLASS_NAMES = {0: "SELL", 1: "HOLD", 2: "BUY"}


def predict(
    features:   pd.DataFrame,
    bundle_dir: Optional[Path] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict signal class and probabilities for a feature DataFrame.

    Parameters
    ----------
    features : pd.DataFrame
        Output of ``build_inference_features()``.
        Must contain all columns listed in ``feature_order.json``.
    bundle_dir : Path, optional
        Directory containing the model bundle (``model.joblib``,
        ``preprocessing.joblib``, ``feature_order.json``, etc.).
        Defaults to ``models/best_model/``.

    Returns
    -------
    predictions : np.ndarray, shape (n_rows,), dtype int
        Class labels: 0 = SELL, 1 = HOLD, 2 = BUY.
    probabilities : np.ndarray, shape (n_rows, 3)
        Softmax class probabilities in [SELL, HOLD, BUY] column order.
    """
    from src.optimization.artifact_manager import InferencePipeline

    bdir = Path(bundle_dir) if bundle_dir is not None else _BUNDLE_DIR
    pipe = InferencePipeline(bdir)

    # Ensure all 247 expected feature columns are present.
    # Missing columns (e.g. HTF columns when no htf_dfs were provided, or
    # skipped feature generators) are filled with NaN — XGBoost handles NaN
    # natively via learned split directions from training.
    missing = [c for c in pipe._feature_order if c not in features.columns]
    if missing:
        logger.warning(
            "predict: %d feature columns absent — filling with NaN "
            "(XGBoost will use learned NaN-branch directions). "
            "Pass htf_dfs to build_inference_features() for full accuracy.",
            len(missing),
        )
        features = features.copy()
        for col in missing:
            features[col] = np.nan

    preds  = pipe.predict(features)
    probas = pipe.predict_proba(features)

    if probas is None:
        # Model does not support predict_proba — construct one-hot fallback
        probas = np.zeros((len(preds), 3), dtype=float)
        for i, p in enumerate(preds):
            probas[i, int(p)] = 1.0

    logger.info(
        "predict: %d bars  bundle=%s  SELL=%d HOLD=%d BUY=%d",
        len(preds), bdir.name,
        (preds == 0).sum(), (preds == 1).sum(), (preds == 2).sum(),
    )
    return preds, probas
