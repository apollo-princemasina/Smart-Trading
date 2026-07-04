"""Feature engineering package for the Smart Trading ICT + ML platform.

Package structure
-----------------
src/features/
    base_feature.py       — BaseFeature abstract contract
    feature_registry.py   — FeatureRegistry (singleton class registry)
    feature_metadata.py   — FeatureMetadata dataclass
    feature_validator.py  — FeatureValidator + FeatureValidationReport
    feature_utils.py      — Shared utility functions
    feature_pipeline.py   — FeaturePipeline orchestrator

    market_structure/     — BOS, CHoCH, Order Blocks, FVGs, Swings
    liquidity/            — Equal H/L, Liquidity Pools, Sweeps
    sessions/             — London, NY, Asia session markers
    trend/                — EMA Stack, HTF Bias, Trend Direction
    volatility/           — ATR, Bollinger Bands, Historical Volatility
    momentum/             — RSI, MACD, Stochastic, ADX
    volume/               — Delta Volume, Volume Profile, CVD
    labels/               — Triple Barrier, Binary Direction, RR Labels

    # Legacy stubs (not yet migrated to BaseFeature):
    ict/                  — ICT indicator stubs
    statistics/           — Statistical indicator stubs
    traditional/          — Traditional indicator stubs

Auto-registration
-----------------
Importing this package triggers the import of every category subpackage.
Each category ``__init__.py`` imports its ``_placeholder.py`` module, which
applies ``@FeatureRegistry.register`` to the placeholder class.

After ``import src.features``, ``FeatureRegistry.all_features()`` will contain
8 placeholder generators — one per category.
"""

from .base_feature      import BaseFeature
from .feature_registry  import FeatureRegistry
from .feature_metadata  import FeatureMetadata
from .feature_validator import FeatureValidator, FeatureValidationReport, PipelineValidationSummary
from .feature_pipeline  import FeaturePipeline
from .feature_utils     import (
    align_to_base,
    cache_path,
    check_required_columns,
    constant_columns,
    data_fingerprint,
    drop_input_columns,
    has_infinite_values,
    load_from_cache,
    load_parquet,
    merge_features,
    prefix_columns,
    save_parquet,
    save_to_cache,
    timer,
)

# ── Auto-register all category placeholders ───────────────────────────────────
# Importing each subpackage fires the @FeatureRegistry.register decorators
# defined in that package's _placeholder.py module.
from . import market_structure
from . import liquidity
from . import sessions
from . import trend
from . import volatility
from . import momentum
from . import volume
from . import labels
from . import technical
from . import statistics
from . import fusion

__all__ = [
    # Core framework
    "BaseFeature",
    "FeatureRegistry",
    "FeatureMetadata",
    "FeatureValidator",
    "FeatureValidationReport",
    "PipelineValidationSummary",
    "FeaturePipeline",
    # Utilities
    "align_to_base",
    "cache_path",
    "check_required_columns",
    "constant_columns",
    "data_fingerprint",
    "drop_input_columns",
    "has_infinite_values",
    "load_from_cache",
    "load_parquet",
    "merge_features",
    "prefix_columns",
    "save_parquet",
    "save_to_cache",
    "timer",
    # Category packages (re-exported for convenience)
    "market_structure",
    "liquidity",
    "sessions",
    "trend",
    "volatility",
    "momentum",
    "volume",
    "labels",
    "technical",
    "statistics",
    "fusion",
]
