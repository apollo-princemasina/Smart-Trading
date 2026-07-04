"""Outcome Tracker — evaluates whether TP or SL was hit for open predictions."""
from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger
from sqlalchemy.ext.asyncio import async_sessionmaker

if TYPE_CHECKING:
    from src.services.rolling_buffer import RollingBufferManager


class OutcomeTracker:

    def __init__(
        self,
        session_factory: async_sessionmaker,
        rolling_buffer:  "RollingBufferManager",
    ) -> None:
        self._factory = session_factory
        self._buffer  = rolling_buffer

    async def evaluate_pending(self) -> int:
        """Check all pending predictions against the current M15 buffer.

        Returns the number of predictions resolved this cycle.
        """
        from src.database.models.prediction        import Prediction
        from src.database.models.outcome            import PredictionOutcome
        from src.database.repositories.prediction_repo import PredictionRepository
        from sqlalchemy.ext.asyncio import AsyncSession

        candles = self._buffer.get_candles("M15")
        if not candles:
            return 0

        resolved = 0
        async with self._factory() as session:
            repo   = PredictionRepository(session)
            pending = await repo.get_pending_outcomes()

            for pred in pending:
                outcome = self._check(pred, candles)
                if outcome["outcome"] == "PENDING":
                    continue

                record = PredictionOutcome(
                    prediction_id = pred.id,
                    outcome       = outcome["outcome"],
                    exit_price    = outcome.get("exit_price"),
                    pnl_pips      = outcome.get("pnl_pips"),
                    bars_to_exit  = outcome.get("bars_to_exit"),
                )
                session.add(record)
                resolved += 1
                logger.info(
                    "Outcome resolved  pred={}  {}  pnl={} pips",
                    pred.id[:8], outcome["outcome"], outcome.get("pnl_pips"),
                )

            if resolved:
                await session.commit()

        return resolved

    @staticmethod
    def _check(pred, candles: list[dict]) -> dict:
        """Walk forward through M15 candles to see if TP or SL was hit."""
        if pred.tp_price is None or pred.sl_price is None:
            return {"outcome": "EXPIRED"}

        # Find candles after the signal time
        signal_ts = pred.signal_time
        relevant  = [c for c in candles if c.get("timestamp") and c["timestamp"] > signal_ts]

        for i, candle in enumerate(relevant):
            high  = float(candle.get("high",  0))
            low   = float(candle.get("low",   0))
            close = float(candle.get("close", 0))

            if pred.direction == "BUY":
                if high >= pred.tp_price:
                    pnl = round((pred.tp_price - pred.close) / 0.0001, 1)
                    return {"outcome": "TP_HIT", "exit_price": pred.tp_price,
                            "pnl_pips": pnl, "bars_to_exit": i + 1}
                if low <= pred.sl_price:
                    pnl = round((pred.sl_price - pred.close) / 0.0001, 1)
                    return {"outcome": "SL_HIT", "exit_price": pred.sl_price,
                            "pnl_pips": pnl, "bars_to_exit": i + 1}
            else:  # SELL
                if low <= pred.tp_price:
                    pnl = round((pred.close - pred.tp_price) / 0.0001, 1)
                    return {"outcome": "TP_HIT", "exit_price": pred.tp_price,
                            "pnl_pips": pnl, "bars_to_exit": i + 1}
                if high >= pred.sl_price:
                    pnl = round((pred.close - pred.sl_price) / 0.0001, 1)
                    return {"outcome": "SL_HIT", "exit_price": pred.sl_price,
                            "pnl_pips": pnl, "bars_to_exit": i + 1}

            # Max hold: 48 M15 bars (12 hours)
            if i >= 47:
                return {"outcome": "EXPIRED", "exit_price": close,
                        "pnl_pips": round((close - pred.close) / 0.0001 *
                                          (1 if pred.direction == "BUY" else -1), 1),
                        "bars_to_exit": i + 1}

        return {"outcome": "PENDING"}
