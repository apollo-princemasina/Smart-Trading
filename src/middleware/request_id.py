"""RequestID middleware — injects X-Request-ID and X-Process-Time on every response."""
from __future__ import annotations

import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


class RequestIDMiddleware(BaseHTTPMiddleware):
    """
    Attaches a unique request ID to every HTTP request and response.

    - Reads X-Request-ID from incoming headers; generates a UUID4 if absent.
    - Adds X-Request-ID and X-Process-Time to the response.
    - Binds the ID to the loguru context so all log lines for this request
      include the correlation ID.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())

        # Make the ID available to downstream code via request.state
        request.state.request_id = request_id

        start = time.monotonic()
        response: Response = await call_next(request)
        elapsed_ms = (time.monotonic() - start) * 1000

        response.headers["X-Request-ID"]   = request_id
        response.headers["X-Process-Time"] = f"{elapsed_ms:.2f}ms"
        return response
