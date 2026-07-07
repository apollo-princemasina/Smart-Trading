"""ICT Entry State Machine — tracks displacement → OB retracement → entry sequence.

Persists across M15 cycles on InferenceEngine.  Each call to update() advances
the state machine one bar and returns an ICTOBEntry when an entry is confirmed.

State flow
----------
IDLE  ──(BOS/CHoCH + OB active)──►  ARMED
        [structural direction locked, OB zone stored]

ARMED  ──(price enters OB zone)──►  OB_TESTED
         [in_order_block = True and ML / structural dir still agree]

ARMED / OB_TESTED  ──(> max_bars elapsed)──►  IDLE   [timeout]
ARMED / OB_TESTED  ──(OB mitigated)──►  IDLE          [OB blown through]

OB_TESTED  ──(next bar: price exits OB in structural direction)──►  FIRE + IDLE
            OR
OB_TESTED  ──(same bar: close still in OB with directional pressure)──►  FIRE + IDLE

When FIRE triggers, the engine receives an ICTOBEntry with precise SL at the OB
far edge (+/- buffer) instead of an ATR-based SL.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.inference.market_regime import RegimeReport

logger = logging.getLogger(__name__)

# Number of M15 bars the state machine stays ARMED before timing out (~2 hours)
_MAX_ARMED_BARS: int = 8

# Pip buffer added to OB far edge for SL placement (prevents stop hunt at exact OB boundary)
_OB_SL_BUFFER_PIPS: float = 2.0
_PIP = 0.0001


@dataclass
class ICTOBEntry:
    """A confirmed ICT Order Block entry signal."""
    direction:   str            # "BUY" | "SELL"
    ob_top:      float          # top of the OB zone
    ob_bottom:   float          # bottom of the OB zone
    sl_price:    float          # stop loss placed beyond OB far edge
    armed_at:    Optional[datetime] = None   # when the setup was first armed
    bars_waited: int = 0        # bars elapsed between arming and entry


# ── Internal states ────────────────────────────────────────────────────────────

_IDLE       = "IDLE"
_ARMED      = "ARMED"
_OB_TESTED  = "OB_TESTED"


class ICTEntryStateMachine:
    """Stateful, single-instance machine that runs once per M15 bar."""

    def __init__(self, max_armed_bars: int = _MAX_ARMED_BARS) -> None:
        self._max_bars   = max_armed_bars
        self._state      = _IDLE
        self._direction: Optional[str]   = None  # "BUY" | "SELL"
        self._ob_top:    Optional[float] = None
        self._ob_bottom: Optional[float] = None
        self._armed_at:  Optional[datetime] = None
        self._bars:      int = 0            # bars elapsed since arming

    # ── Public API ─────────────────────────────────────────────────────────────

    @property
    def state(self) -> str:
        return self._state

    @property
    def armed_direction(self) -> Optional[str]:
        return self._direction

    @property
    def ob_zone(self) -> Optional[tuple[float, float]]:
        """Return (ob_bottom, ob_top) or None if not armed."""
        if self._ob_top is not None and self._ob_bottom is not None:
            return (self._ob_bottom, self._ob_top)
        return None

    def update(
        self,
        regime: "RegimeReport",
        structural_direction: Optional[str],  # from conviction: "BUY" | "SELL" | None
        ml_direction: str,                     # final effective_direction this bar
    ) -> Optional[ICTOBEntry]:
        """Advance the state machine one bar and return an entry signal if confirmed.

        Parameters
        ----------
        regime               : Current bar's RegimeReport (contains OB zone, ICT flags)
        structural_direction : The 4b+8b consensus direction (from conviction data).
                               None means no structural agreement this bar.
        ml_direction         : The 1b model's final direction (after session weighting).
        """
        # Resolve the directional signal we care about — prefer structural (4b+8b),
        # fall back to 1b only if no structural direction is available.
        active_dir = structural_direction or (ml_direction if ml_direction != "HOLD" else None)

        if self._state == _IDLE:
            return self._check_arm(regime, active_dir)

        elif self._state == _ARMED:
            return self._check_ob_tested(regime, active_dir)

        elif self._state == _OB_TESTED:
            return self._check_fire(regime, active_dir)

        return None

    def reset(self) -> None:
        """Force reset to IDLE — call when a signal fires or on major invalidation."""
        self._state     = _IDLE
        self._direction = None
        self._ob_top    = None
        self._ob_bottom = None
        self._armed_at  = None
        self._bars      = 0

    # ── State handlers ─────────────────────────────────────────────────────────

    def _check_arm(
        self,
        regime: "RegimeReport",
        active_dir: Optional[str],
    ) -> None:
        """IDLE → ARMED when a BOS/CHoCH fires and an OB exists in that direction."""
        if active_dir is None:
            return None

        # Require a structural break event this bar
        structural_break = regime.bos_detected or regime.choch_detected
        if not structural_break:
            return None

        # Break must align with active direction
        break_dir = regime.bos_direction if regime.bos_detected else regime.choch_direction
        if break_dir == "NONE" or break_dir != _ict_dir(active_dir):
            return None

        # An active OB must exist in the same direction
        ob_top, ob_bottom = self._extract_ob(regime, active_dir)
        if ob_top is None or ob_bottom is None:
            return None

        # ARM the state machine
        self._state     = _ARMED
        self._direction = active_dir
        self._ob_top    = ob_top
        self._ob_bottom = ob_bottom
        self._armed_at  = datetime.now(timezone.utc)
        self._bars      = 0
        logger.info(
            "ICT SM: ARMED  dir={} OB=[{:.5f}–{:.5f}]",
            active_dir, ob_bottom, ob_top,
        )
        return None

    def _check_ob_tested(
        self,
        regime: "RegimeReport",
        active_dir: Optional[str],
    ) -> Optional[ICTOBEntry]:
        """ARMED → advance or expire."""
        self._bars += 1

        # Expire after max bars
        if self._bars > self._max_bars:
            logger.info("ICT SM: ARMED timeout after {} bars — reset to IDLE", self._bars)
            self.reset()
            return None

        # Direction must not have flipped
        if active_dir and active_dir != self._direction:
            logger.info("ICT SM: direction flipped ({} → {}) — reset", self._direction, active_dir)
            self.reset()
            return None

        # If OB is mitigated (ob_active went False in our direction) → reset
        if not self._ob_still_active(regime):
            logger.info("ICT SM: OB mitigated — reset to IDLE")
            self.reset()
            return None

        # Check if price has now entered the OB zone
        if regime.in_order_block and self._ob_zone_matches(regime):
            self._state = _OB_TESTED
            logger.info(
                "ICT SM: OB_TESTED  dir={} price in OB=[{:.5f}–{:.5f}]  bar={}",
                self._direction, self._ob_bottom, self._ob_top, self._bars,
            )
            # Attempt immediate fire on the same bar (OB touch + ML agrees = enter)
            return self._try_fire(regime, active_dir)

        return None

    def _check_fire(
        self,
        regime: "RegimeReport",
        active_dir: Optional[str],
    ) -> Optional[ICTOBEntry]:
        """OB_TESTED → fire entry or reset."""
        self._bars += 1

        # Expire
        if self._bars > self._max_bars:
            logger.info("ICT SM: OB_TESTED timeout — reset to IDLE")
            self.reset()
            return None

        # Direction flipped
        if active_dir and active_dir != self._direction:
            self.reset()
            return None

        # OB mitigated
        if not self._ob_still_active(regime):
            self.reset()
            return None

        return self._try_fire(regime, active_dir)

    def _try_fire(
        self,
        regime: "RegimeReport",
        active_dir: Optional[str],
    ) -> Optional[ICTOBEntry]:
        """Emit ICTOBEntry if ML agrees direction and OB conditions are met."""
        if active_dir != self._direction:
            return None
        if self._ob_top is None or self._ob_bottom is None:
            return None

        sl = self._compute_sl(self._direction, self._ob_top, self._ob_bottom)
        entry = ICTOBEntry(
            direction   = self._direction,
            ob_top      = self._ob_top,
            ob_bottom   = self._ob_bottom,
            sl_price    = round(sl, 5),
            armed_at    = self._armed_at,
            bars_waited = self._bars,
        )
        logger.info(
            "ICT SM: ENTRY_CONFIRMED  {} OB=[{:.5f}–{:.5f}]  SL={:.5f}  bars_waited={}",
            self._direction, self._ob_bottom, self._ob_top, sl, self._bars,
        )
        self.reset()   # single-use: reset after firing
        return entry

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _extract_ob(
        self,
        regime: "RegimeReport",
        direction: str,
    ) -> tuple[Optional[float], Optional[float]]:
        if direction == "BUY":
            return regime.ob_bullish_top, regime.ob_bullish_bottom
        elif direction == "SELL":
            return regime.ob_bearish_top, regime.ob_bearish_bottom
        return None, None

    def _ob_still_active(self, regime: "RegimeReport") -> bool:
        if self._direction == "BUY":
            return regime.ob_active and regime.ob_direction == "BULLISH"
        elif self._direction == "SELL":
            return regime.ob_active and regime.ob_direction == "BEARISH"
        return False

    def _ob_zone_matches(self, regime: "RegimeReport") -> bool:
        """Confirm price is in the OB zone we armed on, not a stale or different OB."""
        ob_top, ob_bottom = self._extract_ob(regime, self._direction)
        if ob_top is None or ob_bottom is None:
            return False
        # Allow small tolerance (1 pip) in case OB levels refreshed slightly
        tol = 0.0001
        return (abs(ob_top - self._ob_top) < tol and
                abs(ob_bottom - self._ob_bottom) < tol)

    @staticmethod
    def _compute_sl(direction: str, ob_top: float, ob_bottom: float) -> float:
        buf = _OB_SL_BUFFER_PIPS * _PIP
        if direction == "BUY":
            return ob_bottom - buf   # SL below OB bottom (bearish entry)
        else:
            return ob_top + buf      # SL above OB top (bullish candle)


def _ict_dir(ml_dir: str) -> str:
    """Convert ML direction string to ICT regime direction string."""
    return "BULLISH" if ml_dir == "BUY" else "BEARISH"
