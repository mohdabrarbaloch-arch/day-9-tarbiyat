"""Shared test fixtures: an isolated database and a TestClient per test."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Must be set before any app module reads settings.
os.environ.setdefault("SECRET_KEY", "test-secret-that-is-not-used-in-production")
os.environ.setdefault("ENVIRONMENT", "test")
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("RATE_LIMIT_AUTH", "1000/minute")
os.environ.setdefault("RATE_LIMIT_WRITE", "1000/minute")


@pytest.fixture()
def client(tmp_path):
    """A TestClient backed by a throwaway SQLite file, isolated per test.

    The engine is rebound rather than the modules reloaded. Reloading `app.db.base`
    builds a *second* `Base`, which orphans every already-imported model and makes
    `create_all` silently produce an empty database.
    """
    from fastapi.testclient import TestClient

    from app import main as main_module
    from app.db import base as base_module

    base_module.rebind(f"sqlite:///{tmp_path / 'test.db'}")
    base_module.Base.metadata.create_all(bind=base_module.engine)

    application = main_module.create_app()
    application.state.limiter.reset()

    with TestClient(application) as test_client:
        yield test_client


@pytest.fixture()
def auth_headers(client):
    response = client.post(
        "/api/auth/register",
        json={
            "email": "tester@example.com",
            "password": "correct-horse-9",
            "display_name": "Tester",
        },
    )
    assert response.status_code == 201, response.text
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


@pytest.fixture()
def iris_csv_text() -> str:
    """Two well-separated clusters with two features — easy to learn, easy to verify."""
    rows = ["x,y,label"]
    for i in range(20):
        rows.append(f"{i * 0.1:.2f},{i * 0.1 + 0.05:.2f},low")
        rows.append(f"{5 + i * 0.1:.2f},{5 + i * 0.1 + 0.05:.2f},high")
    return "\n".join(rows) + "\n"
