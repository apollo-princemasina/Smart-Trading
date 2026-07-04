"""MetaTrader 5 historical OHLCV candle downloader.

Provides MT5Downloader, a class that wraps the MetaTrader5 Python API to
download, validate, and persist historical candle data as Parquet files.

Requirements:
    - MetaTrader 5 terminal must be installed and running on Windows.
    - Python MetaTrader5 package: pip install MetaTrader5
    - Credentials in .env: MT5_LOGIN, MT5_PASSWORD, MT5_SERVER

Supported timeframes: W1 | D1 | H4 | H1 | M15
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

try:
    import MetaTrader5 as mt5  # type: ignore[import-untyped]
except ImportError as exc:
    raise ImportError(
        "MetaTrader5 package is not installed. "
        "Run: pip install MetaTrader5"
    ) from exc

from config.settings import RAW_DATA_DIR, INGESTION_LOG_PATH

# ── Module logger ─────────────────────────────────────────────────────────────

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    _h = logging.FileHandler(INGESTION_LOG_PATH, encoding="utf-8")
    _h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    logger.addHandler(_h)

# ── Constants ─────────────────────────────────────────────────────────────────

# Maps the project's timeframe strings to MT5 TIMEFRAME_* constants.
TIMEFRAME_MAP: dict[str, int] = {
    "W1":  mt5.TIMEFRAME_W1,
    "D1":  mt5.TIMEFRAME_D1,
    "H4":  mt5.TIMEFRAME_H4,
    "H1":  mt5.TIMEFRAME_H1,
    "M15": mt5.TIMEFRAME_M15,
}

# Canonical column order returned by download().
OHLCV_COLUMNS: list[str] = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "tick_volume",
    "spread",
    "real_volume",
]


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class MT5Config:
    """MT5 terminal connection configuration."""
    login:    int
    password: str
    server:   str
    timeout:  int = 60_000   # milliseconds; increase on slow connections


@dataclass
class DownloadResult:
    """Summary of a single symbol/timeframe download operation."""
    symbol:    str
    timeframe: str
    rows:      int
    path:      Path
    success:   bool
    error:     str = field(default="")


# ── Downloader ────────────────────────────────────────────────────────────────

class MT5Downloader:
    """
    Download historical OHLCV candle data from a MetaTrader 5 terminal.

    Lifecycle
    ---------
    1. Construct with an MT5Config.
    2. Call initialize() to connect to the running MT5 terminal.
    3. Call login() to authenticate.
    4. Call download() one or more times.
    5. Call save() to persist results.
    6. Always call shutdown() in a finally block.

    The downloader is intentionally designed to be extended with additional
    symbols (GBPUSD, USDJPY, XAUUSD, etc.) without any code changes — simply
    pass a different symbol string to download().

    Example
    -------
    >>> cfg = MT5Config(login=12345678, password="secret", server="ICMarkets-Demo")
    >>> dl  = MT5Downloader(cfg)
    >>> dl.initialize()
    >>> dl.login()
    >>> df  = dl.download("EURUSD", "H1", datetime(2017, 1, 1), datetime(2025, 12, 31))
    >>> path = dl.save(df, "EURUSD", "H1", 2017, 2025)
    >>> dl.shutdown()
    """

    def __init__(self, config: MT5Config) -> None:
        self._config      = config
        self._initialized = False

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def initialize(self) -> None:
        """
        Connect to the MT5 terminal.

        Raises:
            RuntimeError: Terminal not running or initialize() returned False.
        """
        logger.info("Initializing MT5 terminal...")
        success = mt5.initialize(timeout=self._config.timeout)
        if not success:
            code, msg = mt5.last_error()
            raise RuntimeError(
                f"mt5.initialize() failed [{code}]: {msg}. "
                "Ensure the MetaTrader 5 terminal is open."
            )

        self._initialized = True
        info = mt5.terminal_info()
        logger.info(
            "MT5 initialized | build=%s path=%s",
            getattr(info, "build", "?"),
            getattr(info, "data_path", "?"),
        )

    def login(self) -> None:
        """
        Authenticate with the MT5 trading account.

        Raises:
            RuntimeError: initialize() has not been called, or login failed.
        """
        if not self._initialized:
            raise RuntimeError("Call initialize() before login().")

        logger.info(
            "Logging in | login=%s server=%s",
            self._config.login, self._config.server,
        )
        authorized = mt5.login(
            login    = self._config.login,
            password = self._config.password,
            server   = self._config.server,
        )
        if not authorized:
            code, msg = mt5.last_error()
            raise RuntimeError(
                f"mt5.login() failed [{code}]: {msg}. "
                "Check MT5_LOGIN, MT5_PASSWORD, and MT5_SERVER in your .env file."
            )

        info = mt5.account_info()
        logger.info(
            "Login successful | name=%s broker=%s currency=%s balance=%.2f",
            getattr(info, "name",     "?"),
            getattr(info, "company",  "?"),
            getattr(info, "currency", "?"),
            getattr(info, "balance",  0.0),
        )

    def shutdown(self) -> None:
        """Disconnect from the MT5 terminal."""
        if self._initialized:
            mt5.shutdown()
            self._initialized = False
            logger.info("MT5 shutdown complete.")

    # ── Download ──────────────────────────────────────────────────────────────

    def download(
        self,
        symbol:     str,
        timeframe:  str,
        start_date: datetime,
        end_date:   datetime,
    ) -> pd.DataFrame:
        """
        Download historical OHLCV candles using mt5.copy_rates_range().

        Args:
            symbol:     MT5 symbol name, e.g. "EURUSD", "GBPUSD", "XAUUSD".
            timeframe:  One of W1 | D1 | H4 | H1 | M15.
            start_date: Inclusive start datetime. May be UTC-aware or naive UTC.
            end_date:   Inclusive end datetime. May be UTC-aware or naive UTC.

        Returns:
            DataFrame with columns (in order):
                timestamp   – datetime64[ns, UTC]
                open        – float64
                high        – float64
                low         – float64
                close       – float64
                tick_volume – int64
                spread      – int64
                real_volume – int64

        Raises:
            ValueError:  Unknown timeframe or symbol unavailable in MT5.
            RuntimeError: No data returned or MT5 error.
        """
        symbol    = symbol.upper()
        timeframe = timeframe.upper()

        self._validate_state()
        self._validate_timeframe(timeframe)
        self._ensure_symbol_selected(symbol)

        # MT5 copy_rates_range() expects naive UTC datetimes (no tzinfo)
        start_naive = _strip_tz(start_date)
        end_naive   = _strip_tz(end_date)

        logger.info(
            "Download started | symbol=%s timeframe=%s from=%s to=%s",
            symbol, timeframe,
            start_naive.strftime("%Y-%m-%d"),
            end_naive.strftime("%Y-%m-%d"),
        )

        rates = mt5.copy_rates_range(
            symbol,
            TIMEFRAME_MAP[timeframe],
            start_naive,
            end_naive,
        )

        if rates is None or len(rates) == 0:
            code, msg = mt5.last_error()
            raise RuntimeError(
                f"No data returned for {symbol} {timeframe} "
                f"[{code}]: {msg or 'verify the symbol name and date range'}."
            )

        df = self._rates_to_dataframe(rates)

        logger.info(
            "Download completed | symbol=%s timeframe=%s rows=%d",
            symbol, timeframe, len(df),
        )
        return df

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(
        self,
        df:         pd.DataFrame,
        symbol:     str,
        timeframe:  str,
        start_year: int,
        end_year:   int,
        output_dir: Optional[Path] = None,
    ) -> Path:
        """
        Persist a candle DataFrame to Parquet via PyArrow.

        Output path structure:
            <output_dir>/<SYMBOL>/<TIMEFRAME>/<SYMBOL>_<TF>_<start>_<end>.parquet

        Args:
            df:         DataFrame produced by download().
            symbol:     Instrument symbol, e.g. "EURUSD".
            timeframe:  Timeframe string, e.g. "H1".
            start_year: First year in the dataset (embedded in filename).
            end_year:   Last year in the dataset (embedded in filename).
            output_dir: Override for RAW_DATA_DIR (useful in tests).

        Returns:
            Absolute path to the saved Parquet file.

        Raises:
            OSError: Disk write failure.
        """
        root      = output_dir or RAW_DATA_DIR
        symbol    = symbol.upper()
        timeframe = timeframe.upper()
        directory = root / symbol / timeframe
        directory.mkdir(parents=True, exist_ok=True)

        path = directory / f"{symbol}_{timeframe}_{start_year}_{end_year}.parquet"
        try:
            df.to_parquet(path, index=False, engine="pyarrow")
        except OSError as exc:
            logger.error("Parquet write failed | path=%s error=%s", path, exc)
            raise

        logger.info("File saved | path=%s rows=%d", path, len(df))
        return path

    # ── Private helpers ───────────────────────────────────────────────────────

    def _validate_state(self) -> None:
        if not self._initialized:
            raise RuntimeError(
                "MT5Downloader is not connected. "
                "Call initialize() and login() first."
            )

    @staticmethod
    def _validate_timeframe(timeframe: str) -> None:
        if timeframe not in TIMEFRAME_MAP:
            raise ValueError(
                f"Unsupported timeframe '{timeframe}'. "
                f"Supported: {sorted(TIMEFRAME_MAP)}"
            )

    @staticmethod
    def _ensure_symbol_selected(symbol: str) -> None:
        """Make the symbol visible in MarketWatch so MT5 can serve its history."""
        if not mt5.symbol_select(symbol, True):
            code, msg = mt5.last_error()
            raise ValueError(
                f"Symbol '{symbol}' is unavailable in MT5 [{code}]: {msg}. "
                "Add it to MarketWatch in your broker's terminal."
            )

    @staticmethod
    def _rates_to_dataframe(rates) -> pd.DataFrame:
        """Convert mt5.copy_rates_range() output to a clean DataFrame."""
        df = pd.DataFrame(rates)

        # 'time' is a Unix timestamp in seconds → convert to UTC datetime
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df.rename(columns={"time": "timestamp"})

        # Enforce canonical column order; drop any MT5-internal extras
        return df[OHLCV_COLUMNS].copy()

    # ── Utility ───────────────────────────────────────────────────────────────

    @staticmethod
    def granularity_timedelta(timeframe: str) -> pd.Timedelta:
        """Return the candle duration for a given timeframe string."""
        mapping: dict[str, pd.Timedelta] = {
            "W1":  pd.Timedelta(weeks=1),
            "D1":  pd.Timedelta(days=1),
            "H4":  pd.Timedelta(hours=4),
            "H1":  pd.Timedelta(hours=1),
            "M15": pd.Timedelta(minutes=15),
        }
        try:
            return mapping[timeframe.upper()]
        except KeyError:
            raise ValueError(f"Unknown timeframe '{timeframe}'. Supported: {sorted(mapping)}")

    # ── Context-manager support ───────────────────────────────────────────────

    def __enter__(self) -> "MT5Downloader":
        self.initialize()
        self.login()
        return self

    def __exit__(self, *_) -> None:
        self.shutdown()


# ── Utility ───────────────────────────────────────────────────────────────────

def _strip_tz(dt: datetime) -> datetime:
    """Return a naive datetime assumed to be UTC (MT5 expects naive UTC)."""
    return dt.replace(tzinfo=None) if dt.tzinfo is not None else dt
