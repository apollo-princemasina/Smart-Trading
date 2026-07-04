"""technical — Traditional Technical Indicator Engine.

Implemented
-----------
MovingAveragesEngine (moving_averages)
    EMA (9/20/50/100/200), SMA (20/50/100), WMA(20), HMA(20),
    ema_slope (5-bar % change of EMA20), ema_cross (sign of EMA20 - EMA50).
    12 float64 columns.  Zero dependencies.

MomentumEngine (momentum)
    RSI-14, Stochastic %K/%D, MACD/Signal/Histogram, CCI-20,
    Williams %R-14, ROC-12, Price Momentum-10, TSI(25,13).
    11 float64 columns.  Zero dependencies.

TrendEngine (trend)
    ADX-14, +DI-14, -DI-14, Aroon Up/Down/Oscillator-25, Parabolic SAR.
    7 float64 columns.  Zero dependencies.

VolatilityEngine (volatility)
    ATR-14, Normalized ATR, Bollinger Bands-20 (upper/lower/width/%B),
    Keltner Channels (EMA20 ± 1.5×ATR), Donchian Channels-20,
    Chaikin Volatility.  11 float64 columns.  Zero dependencies.

OscillatorsEngine (oscillators)
    VWAP (daily-reset), VWMA-20, OBV, CMF-20, MFI-14,
    Accumulation/Distribution, Force Index, EOM-14.
    8 float64 columns.  Zero dependencies.

TechnicalEngine (technical)
    Cross-indicator composite features: price_vs_ema200, price_vs_vwap,
    macd_normalized (by ATR), rsi_stoch_divergence, trend_strength
    (ADX × DI direction sign).  5 float64 columns.
    Depends on: moving_averages, momentum, trend, volatility, oscillators.

Each generator inherits from BaseFeature and is decorated with
@FeatureRegistry.register to self-register with the pipeline.
"""

from ._placeholder      import TechnicalPlaceholder
from .moving_averages   import MovingAveragesEngine
from .momentum          import MomentumEngine
from .trend             import TrendEngine
from .volatility        import VolatilityEngine
from .oscillators       import OscillatorsEngine
from .technical_engine  import TechnicalEngine

__all__ = [
    "TechnicalPlaceholder",
    "MovingAveragesEngine",
    "MomentumEngine",
    "TrendEngine",
    "VolatilityEngine",
    "OscillatorsEngine",
    "TechnicalEngine",
]
