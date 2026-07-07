import pytest
from ..scheduler import build_scheduler


def test_build_scheduler_registers_jobs():
    scheduler = build_scheduler()
    job_ids = {job.id for job in scheduler.get_jobs()}
    assert "calendar_thisweek" in job_ids
    assert "calendar_nextweek" in job_ids
    assert "calendar_lastweek" in job_ids
    assert "speeches" in job_ids
