from ...cache.memory_cache import connector_cache
from ...utils.logger import logger


async def run_speeches_job() -> None:
    """Log speech event summary derived from the cached calendar."""
    try:
        cache = await connector_cache.get_calendar("thisweek")
        speeches = [e for e in cache.events if e.is_speech]
        logger.info(f"Speeches this week: {len(speeches)}")
    except Exception as exc:
        logger.warning(f"speeches_job: {exc}")
