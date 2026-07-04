"""Custom exceptions and FastAPI exception handlers."""
from __future__ import annotations

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from loguru import logger


# ── Domain exceptions ─────────────────────────────────────────────────────────

class MFIPError(Exception):
    """Base class for all MFIP application errors."""
    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR
    error_code:  str = "INTERNAL_ERROR"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class BufferNotReadyError(MFIPError):
    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    error_code  = "BUFFER_NOT_READY"


class ModelNotLoadedError(MFIPError):
    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    error_code  = "MODEL_NOT_LOADED"


class InferenceError(MFIPError):
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    error_code  = "INFERENCE_FAILED"


class PredictionNotFoundError(MFIPError):
    status_code = status.HTTP_404_NOT_FOUND
    error_code  = "PREDICTION_NOT_FOUND"


class TwelveDataError(MFIPError):
    status_code = status.HTTP_502_BAD_GATEWAY
    error_code  = "TWELVE_DATA_ERROR"


# ── Handler registration ──────────────────────────────────────────────────────

def register_exception_handlers(app: FastAPI) -> None:
    """Attach all exception handlers to the FastAPI app."""

    @app.exception_handler(MFIPError)
    async def mfip_error_handler(request: Request, exc: MFIPError) -> JSONResponse:
        logger.warning("{} — {} {}: {}", exc.error_code, request.method, request.url.path, exc.message)
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": exc.error_code, "message": exc.message},
        )

    @app.exception_handler(Exception)
    async def generic_error_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled error — {} {}", request.method, request.url.path)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": "INTERNAL_ERROR", "message": "An unexpected error occurred."},
        )
