"""momentum — oscillator and momentum feature generators.

Planned features
----------------
- RSI (14, 21): raw, smoothed, divergence flag
- MACD (12/26/9): line, signal, histogram, zero-cross
- Stochastic (%K, %D, overbought/oversold)
- ADX: trend strength, +DI / -DI
- Rate of Change (ROC): 5, 10, 20, 50 bar
- Z-score normalised close (mean-reversion signal)
- Commodity Channel Index (CCI)

Each future generator inherits from BaseFeature and is decorated with
@FeatureRegistry.register to self-register with the pipeline.
"""

from ._placeholder import MomentumPlaceholder

__all__ = ["MomentumPlaceholder"]
