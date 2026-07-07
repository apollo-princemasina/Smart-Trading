import asyncio
import random
import functools
from typing import Callable, Type
from .logger import logger
from .config import settings


def async_retry(
    max_retries: int = settings.MAX_RETRIES,
    initial_backoff: float = settings.INITIAL_BACKOFF_S,
    multiplier: float = settings.BACKOFF_MULTIPLIER,
    max_backoff: float = settings.MAX_BACKOFF_S,
    exceptions: tuple[Type[Exception], ...] = (Exception,),
) -> Callable:
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            delay = initial_backoff
            for attempt in range(1, max_retries + 2):
                try:
                    return await fn(*args, **kwargs)
                except exceptions as exc:
                    if attempt > max_retries:
                        raise
                    jitter = random.uniform(-2, 2)
                    wait = min(delay + jitter, max_backoff)
                    logger.warning(
                        f"{fn.__name__} failed (attempt {attempt}/{max_retries}): {exc} — retry in {wait:.1f}s"
                    )
                    await asyncio.sleep(wait)
                    delay *= multiplier
        return wrapper
    return decorator
