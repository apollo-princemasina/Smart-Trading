"""Feature Engine — wraps src.inference.build_inference_features.

Runs in an executor so the synchronous feature pipeline doesn't block
the async event loop.
"""
from __future__ import annotations

import asyncio
from functools import partial
from typing import Optional

import pandas as pd
from loguru import logger

from src.api.core.config import settings


class FeatureEngine:

    async def build(
        self,
        m15_df:  pd.DataFrame,
        htf_dfs: Optional[dict[str, pd.DataFrame]] = None,
    ) -> pd.DataFrame:
        """Build the 247-feature dataset from raw OHLCV dataframes.

        Runs the synchronous FeaturePipeline in a thread executor so it
        doesn't block the async event loop.
        """
        if m15_df.empty:
            raise ValueError("M15 DataFrame is empty — cannot build features")

        loop = asyncio.get_event_loop()
        feature_df = await loop.run_in_executor(
            None,
            partial(self._build_sync, m15_df, htf_dfs),
        )
        logger.debug(
            "Features built: {} rows x {} cols",
            len(feature_df), len(feature_df.columns),
        )
        return feature_df

    @staticmethod
    def _build_sync(
        m15_df:  pd.DataFrame,
        htf_dfs: Optional[dict[str, pd.DataFrame]],
    ) -> pd.DataFrame:
        from src.inference.feature_builder import build_inference_features
        return build_inference_features(
            m15_df,
            htf_dfs,
            symbol=settings.MODEL_SYMBOL,
            spread_fill=settings.INFERENCE_SPREAD_FILL,
        )
