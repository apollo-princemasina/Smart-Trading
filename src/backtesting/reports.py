"""
Reports
=======
Generates 8 output files from a completed backtest:

  1. backtest_report.md         — human-readable summary
  2. trade_log.csv              — every closed trade
  3. equity_curve.csv           — bar-by-bar equity
  4. performance_summary.csv    — key metrics table
  5. risk_report.md             — risk/drawdown analysis
  6. trade_statistics.csv       — exit-reason breakdown
  7. monthly_returns.csv        — monthly P&L
  8. yearly_returns.csv         — yearly P&L
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd


def create_report(metrics: dict) -> str:
    """Thin compatibility shim — returns a one-line summary string."""
    if not metrics:
        return "No trades executed."
    net = metrics.get("net_profit", 0.0)
    wr  = metrics.get("win_rate", 0.0)
    return f"Net profit: {net:.2f} | Win rate: {wr:.2%}"


class BacktestReporter:
    """Write all 8 report files to output_dir."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        trades:         list,
        metrics:        dict,
        equity_df:      pd.DataFrame,
        config_summary: dict,
    ) -> dict[str, Path]:
        """Generate all reports.  Returns {report_name: path}."""
        paths: dict[str, Path] = {}
        closed = [t for t in trades if getattr(t, "status", None) == "closed"]

        paths["backtest_report"]     = self._write_backtest_report(metrics, config_summary)
        paths["trade_log"]           = self._write_trade_log(closed)
        paths["equity_curve"]        = self._write_equity_curve(equity_df)
        paths["performance_summary"] = self._write_performance_summary(metrics)
        paths["risk_report"]         = self._write_risk_report(metrics)
        paths["trade_statistics"]    = self._write_trade_statistics(closed, metrics)
        paths["monthly_returns"]     = self._write_period_returns(equity_df, period="ME")
        paths["yearly_returns"]      = self._write_period_returns(
            equity_df, period="YE", name="yearly_returns"
        )
        return paths

    # ── 1. Backtest report ────────────────────────────────────────────────────

    def _write_backtest_report(self, m: dict, cfg: dict) -> Path:
        p = self.output_dir / "backtest_report.md"
        lines = [
            "# Backtest Report",
            f"\nGenerated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
            "\n## Configuration",
        ]
        for k, v in cfg.items():
            lines.append(f"- **{k}**: {v}")

        lines += ["\n## Performance Summary", "| Metric | Value |", "|--------|-------|"]
        display = [
            ("Net Profit",       "net_profit",      ".4f"),
            ("Gross Profit",     "gross_profit",    ".4f"),
            ("Gross Loss",       "gross_loss",      ".4f"),
            ("Profit Factor",    "profit_factor",   ".4f"),
            ("Expectancy",       "expectancy",      ".4f"),
            ("Win Rate",         "win_rate",        ".2%"),
            ("# Trades",         "n_trades",        ""),
            ("# Winners",        "n_winners",       ""),
            ("# Losers",         "n_losers",        ""),
            ("Sharpe Ratio",     "sharpe_ratio",    ".4f"),
            ("Sortino Ratio",    "sortino_ratio",   ".4f"),
            ("Calmar Ratio",     "calmar_ratio",    ".4f"),
            ("Omega Ratio",      "omega_ratio",     ".4f"),
            ("Max Drawdown",     "max_drawdown",    ".4f"),
            ("Max Drawdown %",   "max_drawdown_pct",".2%"),
            ("Recovery Factor",  "recovery_factor", ".4f"),
            ("Ulcer Index",      "ulcer_index",     ".6f"),
        ]
        for label, key, fmt in display:
            val = m.get(key)
            if val is None:
                lines.append(f"| {label} | N/A |")
            else:
                try:
                    formatted = format(val, fmt) if fmt else str(val)
                except (ValueError, TypeError):
                    formatted = str(val)
                lines.append(f"| {label} | {formatted} |")

        p.write_text("\n".join(lines), encoding="utf-8")
        return p

    # ── 2. Trade log ──────────────────────────────────────────────────────────

    def _write_trade_log(self, trades: list) -> Path:
        p = self.output_dir / "trade_log.csv"
        if not trades:
            pd.DataFrame().to_csv(p, index=False)
            return p
        rows = []
        for t in trades:
            rows.append({
                "trade_id":         t.trade_id,
                "direction":        t.direction,
                "signal_time":      t.signal_time,
                "entry_time":       t.entry_time,
                "exit_time":        t.exit_time,
                "entry_price":      t.entry_price,
                "exit_price":       t.exit_price,
                "stop_loss":        t.stop_loss,
                "take_profit":      t.take_profit,
                "lot_size":         t.lot_size,
                "commission":       t.commission,
                "spread_cost":      t.spread_cost,
                "slippage_cost":    t.slippage_cost,
                "gross_profit":     t.gross_profit,
                "net_profit":       t.net_profit,
                "profit_pips":      t.profit_pips,
                "exit_reason":      t.exit_reason,
                "holding_bars":     t.holding_bars,
                "confidence":       t.confidence,
                "prediction_class": t.prediction_class,
                "session":          t.session,
                "atr_at_entry":     t.atr_at_entry,
            })
        pd.DataFrame(rows).to_csv(p, index=False)
        return p

    # ── 3. Equity curve ───────────────────────────────────────────────────────

    def _write_equity_curve(self, equity_df: pd.DataFrame) -> Path:
        p = self.output_dir / "equity_curve.csv"
        equity_df.to_csv(p)
        return p

    # ── 4. Performance summary ────────────────────────────────────────────────

    def _write_performance_summary(self, m: dict) -> Path:
        p = self.output_dir / "performance_summary.csv"
        rows = [{"metric": k, "value": v} for k, v in m.items() if k != "exit_breakdown"]
        pd.DataFrame(rows).to_csv(p, index=False)
        return p

    # ── 5. Risk report ────────────────────────────────────────────────────────

    def _write_risk_report(self, m: dict) -> Path:
        p = self.output_dir / "risk_report.md"
        lines = [
            "# Risk Report",
            f"\nGenerated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
            "\n## Drawdown",
        ]
        for key, label in [
            ("max_drawdown",     "Max Drawdown (USD)"),
            ("max_drawdown_pct", "Max Drawdown (%)"),
            ("recovery_factor",  "Recovery Factor"),
            ("ulcer_index",      "Ulcer Index"),
        ]:
            lines.append(f"- **{label}**: {m.get(key)}")

        lines.append("\n## Return Risk Ratios")
        for key, label in [
            ("sharpe_ratio",  "Sharpe"),
            ("sortino_ratio", "Sortino"),
            ("calmar_ratio",  "Calmar"),
            ("omega_ratio",   "Omega"),
        ]:
            lines.append(f"- **{label}**: {m.get(key)}")

        p.write_text("\n".join(lines), encoding="utf-8")
        return p

    # ── 6. Trade statistics ───────────────────────────────────────────────────

    def _write_trade_statistics(self, trades: list, m: dict) -> Path:
        p = self.output_dir / "trade_statistics.csv"
        breakdown = m.get("exit_breakdown") or {}
        rows = [{"exit_reason": k, "count": v} for k, v in breakdown.items()]
        pd.DataFrame(rows).to_csv(p, index=False)
        return p

    # ── 7 & 8. Period returns ─────────────────────────────────────────────────

    def _write_period_returns(
        self,
        equity_df: pd.DataFrame,
        period:    str = "ME",
        name:      str = "monthly_returns",
    ) -> Path:
        p = self.output_dir / f"{name}.csv"
        if equity_df.empty or "equity" not in equity_df.columns:
            pd.DataFrame().to_csv(p, index=False)
            return p

        eq = equity_df["equity"].dropna()
        try:
            resampled = eq.resample(period).agg(["first", "last"])
        except Exception:
            pd.DataFrame().to_csv(p, index=False)
            return p

        resampled.columns   = ["period_start", "period_end"]
        resampled["return_usd"] = resampled["period_end"] - resampled["period_start"]
        resampled["return_pct"] = (
            resampled["return_usd"] / resampled["period_start"] * 100.0
        ).round(4)
        resampled.to_csv(p)
        return p
