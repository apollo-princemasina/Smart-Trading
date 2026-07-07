import sys
from loguru import logger

logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>ff_connector</cyan> | {message}",
    level="INFO",
)
logger.add(
    "logs/ff_connector.log",
    rotation="10 MB",
    retention="7 days",
    level="DEBUG",
)

__all__ = ["logger"]
