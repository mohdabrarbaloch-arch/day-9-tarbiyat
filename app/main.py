"""FastAPI application factory."""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import auth, datasets, runs
from app.core.config import get_settings
from app.core.ratelimit import RateLimiter
from app.db.base import init_db
from app.schemas import HealthOut

logger = logging.getLogger("tarbiyat")

VERSION = "1.0.0"
STATIC_DIR = Path(__file__).parent / "static"


class SecurityHeadersMiddleware:
    """Helmet-equivalent for a page that owns its own UI.

    The CSP allows no remote script or style origins. Everything the frontend needs is
    bundled, so there is no reason to permit a third-party origin — and permitting one
    "just in case" is exactly how a demo turns into an injection target.
    """

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                headers = dict(message.get("headers", []))
                h = message.setdefault("headers", [])
                additions = {
                    b"x-content-type-options": b"nosniff",
                    b"x-frame-options": b"DENY",
                    b"referrer-policy": b"no-referrer",
                    b"permissions-policy": b"geolocation=(), microphone=(), camera=()",
                    b"content-security-policy": (
                        b"default-src 'self'; "
                        b"script-src 'self'; "
                        b"style-src 'self' 'unsafe-inline'; "
                        b"img-src 'self' data:; "
                        b"connect-src 'self'; "
                        b"font-src 'self'; "
                        b"object-src 'none'; "
                        b"base-uri 'none'; "
                        b"frame-ancestors 'none'"
                    ),
                }
                for key, value in additions.items():
                    if key not in headers:
                        h.append((key, value))
            await send(message)

        await self.app(scope, receive, send_wrapper)


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="Tarbiyat — ML Training Playground",
        description=(
            "Upload a CSV, configure a neural network, and watch it actually train. "
            "Every gradient is computed by hand; there is no framework underneath."
        ),
        version=VERSION,
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
    )

    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type"],
    )

    app.state.limiter = RateLimiter()
    app.state.settings = settings

    app.include_router(auth.router)
    app.include_router(datasets.router)
    app.include_router(runs.router)

    @app.get("/api/health", response_model=HealthOut, tags=["meta"])
    def health() -> HealthOut:
        return HealthOut(status="ok", version=VERSION, environment=settings.environment)

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        # Log the detail server-side, return nothing revealing to the client.
        logger.exception("Unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Something went wrong on our side. The error was logged."},
        )

    if STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

        @app.get("/", include_in_schema=False)
        def index() -> FileResponse:
            return FileResponse(STATIC_DIR / "index.html")

    @app.on_event("startup")
    def _startup() -> None:
        init_db()

    return app


app = create_app()
