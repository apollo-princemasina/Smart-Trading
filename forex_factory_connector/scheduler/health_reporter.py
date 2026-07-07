from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from ..utils.config import settings
from ..utils.logger import logger

SCHEMA_VERSION = "1.0.0"
_CIRCUIT_THRESHOLD = settings.CIRCUIT_BREAKER_THRESHOLD


@dataclass
class JobMetrics:
    job_id:          str
    poll_interval_s: int

    last_success:  Optional[datetime] = None
    last_failure:  Optional[datetime] = None
    success_count: int  = 0
    failure_count: int  = 0
    retry_count:   int  = 0
    circuit_open:  bool = False

    # Rolling window of the last 20 response times (milliseconds)
    _response_times: deque = field(default_factory=lambda: deque(maxlen=20))

    def record_success(self, response_time_ms: float) -> None:
        self.last_success   = datetime.now(timezone.utc)
        self.success_count += 1
        self.circuit_open   = False
        self._response_times.append(round(response_time_ms, 1))

    def record_failure(self, exc: Exception) -> None:
        self.last_failure   = datetime.now(timezone.utc)
        self.failure_count += 1
        if self.failure_count >= _CIRCUIT_THRESHOLD and not self.circuit_open:
            self.circuit_open = True
            logger.error(f"[{self.job_id}] circuit breaker OPEN after {self.failure_count} failures")

    def record_retry(self) -> None:
        self.retry_count += 1

    @property
    def avg_response_ms(self) -> Optional[float]:
        if not self._response_times:
            return None
        return round(sum(self._response_times) / len(self._response_times), 1)

    @property
    def last_response_ms(self) -> Optional[float]:
        return self._response_times[-1] if self._response_times else None

    @property
    def status(self) -> str:
        if self.circuit_open:
            return "down"
        if self.last_success is None and self.failure_count > 0:
            return "not_started"
        if self.last_success is None:
            return "initializing"
        if self.failure_count > 0 and (
            self.last_failure and self.last_success and
            self.last_failure > self.last_success
        ):
            return "degraded"
        return "ok"


class HealthReporter:
    def __init__(self) -> None:
        self._jobs: dict[str, JobMetrics] = {}

    def register_job(self, job_id: str, poll_interval_s: int) -> None:
        self._jobs[job_id] = JobMetrics(job_id=job_id, poll_interval_s=poll_interval_s)

    def record_success(self, job_id: str, response_time_ms: float = 0.0) -> None:
        if job_id not in self._jobs:
            return
        self._jobs[job_id].record_success(response_time_ms)
        logger.debug(f"[{job_id}] success ({response_time_ms:.0f} ms)")

    def record_failure(self, job_id: str, exc: Exception) -> None:
        if job_id not in self._jobs:
            return
        self._jobs[job_id].record_failure(exc)
        logger.warning(f"[{job_id}] failure #{self._jobs[job_id].failure_count}: {exc}")

    def record_retry(self, job_id: str) -> None:
        if job_id in self._jobs:
            self._jobs[job_id].record_retry()

    def is_open(self, job_id: str) -> bool:
        m = self._jobs.get(job_id)
        return m.circuit_open if m else False

    def get_job(self, job_id: str) -> Optional[JobMetrics]:
        return self._jobs.get(job_id)

    def all_jobs(self) -> dict[str, JobMetrics]:
        return dict(self._jobs)

    @property
    def overall_status(self) -> str:
        if not self._jobs:
            return "initializing"
        statuses = {j.status for j in self._jobs.values()}
        if statuses == {"ok"}:
            return "ok"
        if "down" in statuses:
            return "degraded"
        if statuses <= {"initializing", "not_started"}:
            return "initializing"
        return "degraded"


# Module-level singleton
health = HealthReporter()
