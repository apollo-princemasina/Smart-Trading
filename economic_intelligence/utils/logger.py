"""EIE logger — thin wrapper so all EIE modules share the same logger name."""
from loguru import logger as _logger

logger = _logger.bind(module="eie")
