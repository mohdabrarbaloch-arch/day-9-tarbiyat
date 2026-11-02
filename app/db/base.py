"""SQLAlchemy declarative base and engine/session wiring."""

from __future__ import annotations

import os
from collections.abc import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.core.config import get_settings


class Base(DeclarativeBase):
    """Common base for every ORM model."""


def _resolve_database_url(raw: str) -> str:
    """Vercel's filesystem is read-only apart from /tmp, so relocate SQLite there."""
    if raw.startswith("sqlite") and os.getenv("VERCEL"):
        return "sqlite:////tmp/tarbiyat.db"
    return raw


def _build_engine():
    settings = get_settings()
    url = _resolve_database_url(settings.database_url)
    connect_args: dict[str, object] = {}
    if url.startswith("sqlite"):
        # FastAPI may touch the connection from a worker thread.
        connect_args["check_same_thread"] = False
    return create_engine(url, connect_args=connect_args, pool_pre_ping=True, future=True)


engine = _build_engine()
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def rebind(url: str) -> None:
    """Point the module at a different database.

    Used by the test suite so each test gets its own file. Rebinding the engine is safe
    here because `Base.metadata` and the mapped classes are shared and stay valid — the
    models are registered on the metadata once, at import, and never depend on which
    engine is current.
    """
    global engine, SessionLocal
    connect_args: dict[str, object] = {}
    if url.startswith("sqlite"):
        connect_args["check_same_thread"] = False
    engine = create_engine(url, connect_args=connect_args, pool_pre_ping=True, future=True)
    SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def get_db() -> Iterator[Session]:
    """FastAPI dependency yielding a scoped session that always closes."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def init_db() -> None:
    """Create tables. Alembic handles migrations in a real deployment; this is for dev."""
    from app import models  # noqa: F401  (import registers the mappers)

    Base.metadata.create_all(bind=engine)
