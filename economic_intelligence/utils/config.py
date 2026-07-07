"""EIE runtime configuration — all values overridable via environment variables."""
from __future__ import annotations

import os


class EIEConfig:
    # Scheduler poll interval for the EIE cycle (seconds)
    EIE_CYCLE_SECONDS: int = int(os.getenv("EIE_CYCLE_SECONDS", "60"))

    # Decay: minimum remaining influence to keep an event "active" (0-100)
    EIE_ACTIVE_THRESHOLD: float = float(os.getenv("EIE_ACTIVE_THRESHOLD", "5.0"))

    # Execution risk: lookahead window for upcoming HIGH events (minutes)
    EIE_RISK_LOOKAHEAD_MIN: int = int(os.getenv("EIE_RISK_LOOKAHEAD_MIN", "120"))

    # Execution risk: lookback window for recent releases (minutes)
    EIE_RISK_LOOKBACK_MIN: int = int(os.getenv("EIE_RISK_LOOKBACK_MIN", "60"))

    # Circuit breaker: open after N consecutive EIE failures
    EIE_CIRCUIT_BREAKER_THRESHOLD: int = int(os.getenv("EIE_CIRCUIT_BREAKER_THRESHOLD", "5"))

    # How many processed reports to keep in memory per currency
    EIE_MAX_REPORTS_PER_CURRENCY: int = int(os.getenv("EIE_MAX_REPORTS_PER_CURRENCY", "50"))

    # Surprise engine: minimum |forecast| to compute pct_surprise (avoids division near zero)
    EIE_MIN_FORECAST_ABS: float = float(os.getenv("EIE_MIN_FORECAST_ABS", "0.001"))


eie_config = EIEConfig()
