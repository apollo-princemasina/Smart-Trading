"""statistics — Statistical & Market Microstructure Feature Engine.

Implemented
-----------
ReturnsEngine (returns)
    Log return, simple return, rolling log-return sums (5/20 bars), 1-bar
    forward return.  5 float64 columns.  Zero dependencies.

CandleStatisticsEngine (candle_statistics)
    Body/wick anatomy, doji/marubozu scores, inside/outside bars, consecutive
    run lengths, rolling directional counts.  21 float64 columns.

RollingStatisticsEngine (rolling_statistics)
    20-bar rolling mean, median, variance, std, min, max, Q25, Q75, MAD
    on close.  9 float64 columns.  Zero dependencies.

DistributionEngine (distribution)
    Rolling skewness/kurtosis of log_return; z-score, percentile rank,
    min-max normalised price, ordinal price rank.  6 float64 columns.
    Depends on: returns.

MomentumStatisticsEngine (momentum_stats)
    Price velocity, acceleration, deceleration, rolling momentum sums,
    lag-1 autocorrelation, same-sign fraction.  7 float64 columns.
    Depends on: returns.

VolatilityStatisticsEngine (volatility_stats)
    Realised/historical volatility, expansion/compression, ATR ratio,
    rolling ATR mean, volatility regime.  7 float64 columns.
    Depends on: returns, volatility.

EntropyEngine (entropy)
    Shannon entropy (20-bar, 5-bar) and approximate entropy (30-bar)
    of log_return.  3 float64 columns.  Depends on: returns.

MarketMicrostructureEngine (market_microstructure)
    Efficiency ratio, Hurst exponent, fractal dimension, market noise,
    directional efficiency, price smoothness, mean-reversion score,
    trend score.  8 float64 columns.  Depends on: returns.

StatisticalEngine (statistics)
    Cross-module composites: return_vol_ratio, trend_quality, noise_ratio,
    price_efficiency, regime_consistency.  5 float64 columns.
    Depends on: all 8 sub-engines above.

Total output: 71 float64 columns across 9 engines.

Each generator inherits from BaseFeature and is decorated with
@FeatureRegistry.register to self-register with the pipeline.
"""

from ._placeholder         import StatisticsPlaceholder
from .returns              import ReturnsEngine
from .candle_statistics    import CandleStatisticsEngine
from .rolling_statistics   import RollingStatisticsEngine
from .distribution         import DistributionEngine
from .momentum             import MomentumStatisticsEngine
from .volatility           import VolatilityStatisticsEngine
from .entropy              import EntropyEngine
from .market_microstructure import MarketMicrostructureEngine
from .statistical_engine   import StatisticalEngine

__all__ = [
    "StatisticsPlaceholder",
    "ReturnsEngine",
    "CandleStatisticsEngine",
    "RollingStatisticsEngine",
    "DistributionEngine",
    "MomentumStatisticsEngine",
    "VolatilityStatisticsEngine",
    "EntropyEngine",
    "MarketMicrostructureEngine",
    "StatisticalEngine",
]
