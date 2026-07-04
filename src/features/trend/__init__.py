"""trend — directional trend and bias feature generators.

Planned features
----------------
- EMA stack (9, 21, 50, 200 EMA) — bullish / bearish alignment
- Price position relative to key MAs (pct distance)
- Higher-timeframe trend bias (H1, H4, D1 direction encoded as -1/0/+1)
- Trend slope (angle of EMA over N bars)
- Consecutive closes above/below EMA (persistence)
- LuxAlgo Smart Money Concepts trend signals

Each future generator inherits from BaseFeature and is decorated with
@FeatureRegistry.register to self-register with the pipeline.
"""

from ._placeholder import TrendPlaceholder

__all__ = ["TrendPlaceholder"]
