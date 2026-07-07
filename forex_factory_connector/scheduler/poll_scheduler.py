from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

from .jobs.calendar_job  import run_calendar_job
from .jobs.speeches_job  import run_speeches_job
from .jobs.news_job      import run_news_job
from .jobs.sentiment_job import run_sentiment_job
from .health_reporter    import health
from ..utils.config      import settings
from ..utils.logger      import logger

# Job registry: (job_id, coroutine_fn, kwargs, poll_interval_s)
_JOB_REGISTRY = [
    ("calendar:thisweek", run_calendar_job, {"week": "thisweek"}, settings.CALENDAR_POLL_SECONDS),
    ("calendar:nextweek", run_calendar_job, {"week": "nextweek"}, settings.CALENDAR_POLL_SECONDS),
    ("calendar:lastweek", run_calendar_job, {"week": "lastweek"}, settings.CALENDAR_POLL_SECONDS),
    ("speeches",          run_speeches_job,  {},                   settings.SPEECHES_POLL_SECONDS),
    ("news",              run_news_job,       {},                   settings.NEWS_POLL_SECONDS),
    ("sentiment",         run_sentiment_job,  {},                   settings.SENTIMENT_POLL_SECONDS),
]


def build_scheduler() -> AsyncIOScheduler:
    scheduler = AsyncIOScheduler(timezone="UTC")

    for job_id, fn, kwargs, interval_s in _JOB_REGISTRY:
        health.register_job(job_id, interval_s)
        scheduler.add_job(
            fn,
            trigger=IntervalTrigger(seconds=interval_s),
            kwargs=kwargs,
            id=job_id,
            name=job_id,
            max_instances=1,
            coalesce=True,
            misfire_grace_time=30,
        )

    logger.info(
        "Scheduler built: calendar={}s, speeches={}s, news={}s, sentiment={}s",
        settings.CALENDAR_POLL_SECONDS,
        settings.SPEECHES_POLL_SECONDS,
        settings.NEWS_POLL_SECONDS,
        settings.SENTIMENT_POLL_SECONDS,
    )
    return scheduler
