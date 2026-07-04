"""volatility — market volatility feature generators.

Planned features
----------------
- ATR (14, 20): raw and as % of price
- Bollinger Bands: width, %B position, squeeze detection
- Historical volatility: rolling log-return standard deviation
- Garman-Klass volatility estimator (OHLC-based)
- Volatility regime: low / normal / high (rolling percentile)
- Intraday range: (high - low) / ATR

Each future generator inherits from BaseFeature and is decorated with
@FeatureRegistry.register to self-register with the pipeline.
"""

from ._placeholder import VolatilityPlaceholder

__all__ = ["VolatilityPlaceholder"]
