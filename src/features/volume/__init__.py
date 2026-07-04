"""volume — volume analysis feature generators.

Implemented
-----------
PremiumDiscountDeltaVolumeEngine (premium_discount_delta_volume)
    BigBeluga-style delta volume paired with the ICT Premium/Discount
    framework.  Computes SR-period (50 bar) and Macro-period (200 bar)
    delta volumes, zone-attributed volumes, and volume regime signals.
    Outputs 20 ML-ready float64 columns.
    Depends on: premium_discount.

Planned
-------
- CumulativeVolumeDeltaEngine (cvd)          — running net delta
- VolumeProfileEngine (volume_profile)       — POC, VAH, VAL per session
- OnBalanceVolumeEngine (obv)                — trend confirmation
- VolumeZScoreEngine (volume_zscore)         — relative volume
- TickVolumeRatioEngine (tick_volume_ratio)  — M15 / H1 tick ratio

Legacy stubs (delta_volume.py, volume_profile.py) will be migrated to
inherit from BaseFeature in a future iteration.

Each new generator inherits from BaseFeature and is decorated with
@FeatureRegistry.register to self-register with the pipeline.
"""

from ._placeholder                      import VolumePlaceholder
from .premium_discount_delta_volume     import PremiumDiscountDeltaVolumeEngine

__all__ = [
    "VolumePlaceholder",
    "PremiumDiscountDeltaVolumeEngine",
]
