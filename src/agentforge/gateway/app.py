from __future__ import annotations

import uuid
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import text

from agentforge.config import get_settings
from agentforge.db import create_all, get_engine
from agentforge.gateway.api import router as api_router
from agentforge.gateway.web import router as web_router
from agentforge.logging import configure_logging, get_logger
from agentforge.runtime.queue import close_redis, get_redis
from agentforge.security import generate_csrf_token
from agentforge.services.auth import bootstrap_default_admin
from agentforge.training.audit import PolicyViolation

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level)
    settings.ensure_directories()
    if settings.is_test or settings.database_url.startswith("sqlite"):
        await create_all()
    await bootstrap_default_admin()
    logger.info("gateway_started", environment=settings.env)
    yield
    await close_redis()
    logger.info("gateway_stopped")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="AgentForge API",
        version="0.1.0",
        description="Asynchronous multi-agent runtime platform",
        docs_url="/docs" if settings.expose_docs else None,
        redoc_url="/redoc" if settings.expose_docs else None,
        openapi_url="/openapi.json" if settings.expose_docs else None,
        lifespan=lifespan,
    )
    static_dir = settings.web_dir / "static"
    static_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        csrf_token = request.cookies.get("agentforge_csrf") or generate_csrf_token()
        request.state.request_id = request_id
        request.state.csrf_token = csrf_token
        structlog.contextvars.bind_contextvars(request_id=request_id)
        try:
            response = await call_next(request)
        finally:
            structlog.contextvars.clear_contextvars()
        response.headers["X-Request-ID"] = request_id
        if request.cookies.get("agentforge_csrf") != csrf_token:
            response.set_cookie(
                "agentforge_csrf",
                csrf_token,
                httponly=False,
                samesite="lax",
                secure=settings.is_production,
                max_age=12 * 60 * 60,
            )
        return response

    @app.exception_handler(Exception)
    async def unhandled_exception(request: Request, exc: Exception):
        logger.exception("unhandled_request_error", path=request.url.path)
        return JSONResponse(
            status_code=500,
            content={
                "type": "about:blank",
                "title": "Internal Server Error",
                "status": 500,
                "detail": "The request could not be completed.",
                "instance": request.url.path,
            },
        )

    @app.exception_handler(PolicyViolation)
    async def policy_violation(request: Request, exc: PolicyViolation):
        return JSONResponse(
            status_code=409,
            content={
                "type": "about:blank",
                "title": "Policy Violation",
                "status": 409,
                "detail": exc.reason,
                "reason_code": exc.reason_code,
                "instance": request.url.path,
            },
        )

    @app.get("/health/live", tags=["health"])
    async def live():
        return {"status": "ok"}

    @app.get("/health/ready", tags=["health"])
    async def ready():
        engine = get_engine()
        async with engine.connect() as connection:
            await connection.execute(text("SELECT 1"))
        await get_redis().ping()
        return {"status": "ready"}

    app.include_router(api_router)
    app.include_router(web_router)
    return app
