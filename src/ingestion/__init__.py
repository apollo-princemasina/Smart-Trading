"""Ingestion package — MetaTrader 5 market data pipeline."""

from .mt5_downloader import MT5Config, MT5Downloader, DownloadResult
from .update_data import MT5DataUpdater
from .validate_data import ValidationReport, validate_dataframe

__all__ = [
    "MT5Config",
    "MT5Downloader",
    "MT5DataUpdater",
    "DownloadResult",
    "ValidationReport",
    "validate_dataframe",
]
