"""FastAPI application entry point for the private model-serving process."""

from __future__ import annotations

import hmac
import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from server.api.v1.endpoints import router as v1_router
from server.api.v1.batch_infer import router as batch_infer_router  # daily pipeline
from server.config import (
    ALLOW_EXPERIMENTAL_RELEASE,
    AUTH_REQUIRED,
    CORS_ORIGINS,
    ENVIRONMENT,
    HOST,
    LOG_LEVEL,
    MODEL_SERVER_API_KEY,
    PORT,
    SERVICE_NAME,
    SERVICE_VERSION,
)
from server.contracts import ErrorDetail, ErrorResponse
from server.core.catalog import model_catalog
from server.core.model_loader import model_manager
from server.errors import ServiceError


logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(SERVICE_NAME)


def _error_response(
    *,
    status_code: int,
    code: str,
    message: str,
    request_id: str,
    retryable: bool,
    details=None,
) -> JSONResponse:
    payload = ErrorResponse(
        error=ErrorDetail(
            code=code,
            message=message,
            request_id=request_id,
            retryable=retryable,
            details=details,
        )
    )
    return JSONResponse(status_code=status_code, content=payload.model_dump(mode="json"))


@asynccontextmanager
async def lifespan(_: FastAPI):
    logger.info(
        "service_started environment=%s catalog=%s models=%s",
        ENVIRONMENT,
        getattr(model_catalog, "catalog_version", "v1"),
        len(getattr(model_catalog, "models", {})),
    )
    yield
    model_manager.clear_cache()
    logger.info("service_stopped")


app = FastAPI(
    title="Myanmar Agricultural Model Serving API",
    description=(
        "Model inference microservice for 40 agricultural indicators and composite features."
    ),
    version=SERVICE_VERSION,
    lifespan=lifespan,
    docs_url=None if ENVIRONMENT == "production" else "/docs",
    redoc_url=None,
)

if CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(CORS_ORIGINS),
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "X-Internal-API-Key", "X-Request-ID"],
    )


@app.middleware("http")
async def security_and_request_context(request: Request, call_next):
    supplied_request_id = request.headers.get("x-request-id", "")
    request_id = (
        supplied_request_id
        if 0 < len(supplied_request_id) <= 128
        else str(uuid.uuid4())
    )
    request.state.request_id = request_id

    public_paths = {"/api/v1/live", "/api/v1/ready", "/docs", "/openapi.json"}
    if AUTH_REQUIRED and request.url.path not in public_paths:
        supplied_key = request.headers.get("x-internal-api-key", "")
        if not supplied_key or not hmac.compare_digest(supplied_key, MODEL_SERVER_API_KEY):
            return _error_response(
                status_code=401,
                code="UNAUTHORIZED",
                message="valid service credentials are required",
                request_id=request_id,
                retryable=False,
            )

    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Cache-Control"] = "no-store"
    return response


@app.exception_handler(ServiceError)
async def handle_service_error(request: Request, exc: ServiceError):
    return _error_response(
        status_code=exc.status_code,
        code=exc.code,
        message=exc.safe_message,
        request_id=str(request.state.request_id),
        retryable=exc.retryable,
        details=exc.details,
    )


@app.exception_handler(RequestValidationError)
async def handle_validation_error(request: Request, exc: RequestValidationError):
    details = [
        {"path": ".".join(str(part) for part in error["loc"]), "message": error["msg"]}
        for error in exc.errors()
    ]
    return _error_response(
        status_code=422,
        code="REQUEST_VALIDATION_FAILED",
        message="request did not match the API contract",
        request_id=str(request.state.request_id),
        retryable=False,
        details=details,
    )


@app.exception_handler(Exception)
async def handle_unexpected_error(request: Request, exc: Exception):
    logger.exception("unhandled_error request_id=%s", getattr(request.state, "request_id", "unknown"))
    return _error_response(
        status_code=500,
        code="INTERNAL_ERROR",
        message="the model server encountered an internal error",
        request_id=str(getattr(request.state, "request_id", "unknown")),
        retryable=False,
    )


app.include_router(v1_router)
# Daily-pipeline batch inference (internal use only, requires API key)
app.include_router(batch_infer_router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server.main:app", host=HOST, port=PORT, reload=ENVIRONMENT == "development")
