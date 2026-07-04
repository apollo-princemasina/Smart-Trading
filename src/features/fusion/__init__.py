"""fusion — Multi-Timeframe Feature Fusion Engine.

Implemented
-----------
TimeframeMapper
    Canonical timeframe metadata: prefix, Timedelta, rank, hierarchy.
    Aliases: ``"4H"`` → ``"H4"``, ``"1D"`` → ``"D"``, etc.

FeatureAligner
    Aligns one higher-timeframe DataFrame onto a base (M15) DatetimeIndex
    using ``pd.merge_asof(direction='backward')``.  Guarantees zero
    look-ahead bias by shifting each HTF bar index forward by one full period
    (``available_at = bar_open + duration``) before joining.

ValidationResult / FusionValidator
    Pre-fusion checks (timezone, monotonicity, time consistency) and
    post-fusion checks (look-ahead detection, duplicate column names,
    completeness reporting).

FeatureFusion
    Orchestrates alignment of W/D/H4/H1 onto M15 and concatenates all
    prefixed columns into a single DataFrame.

FusionEngine
    End-to-end orchestrator supporting multi-symbol, incremental updates,
    fingerprint-based Parquet caching, and parallel execution.
    Saves to ``data/features/{symbol}/feature_dataset_fused.parquet``.
"""

from .timeframe_mapper  import TimeframeMapper
from .feature_alignment import FeatureAligner
from .fusion_validator  import FusionValidator, ValidationResult
from .feature_fusion    import FeatureFusion
from .fusion_engine     import FusionEngine

__all__ = [
    "TimeframeMapper",
    "FeatureAligner",
    "FusionValidator",
    "ValidationResult",
    "FeatureFusion",
    "FusionEngine",
]
