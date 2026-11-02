"""ORM models: users, datasets, runs."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


def _utcnow() -> datetime:
    return datetime.now(UTC)


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    display_name: Mapped[str] = mapped_column(String(120), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    datasets: Mapped[list[Dataset]] = relationship(back_populates="owner")
    runs: Mapped[list[TrainingRun]] = relationship(back_populates="owner")


class Dataset(Base):
    __tablename__ = "datasets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    owner_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    name: Mapped[str] = mapped_column(String(200))
    target: Mapped[str] = mapped_column(String(120))
    task: Mapped[str] = mapped_column(String(20))
    fingerprint: Mapped[str] = mapped_column(String(32), index=True)
    row_count: Mapped[int] = mapped_column(Integer)
    column_count: Mapped[int] = mapped_column(Integer)
    # The raw CSV is kept so a run can be reproduced exactly. Small by construction
    # (the upload limit is a few MB), and never served to the client verbatim.
    csv_body: Mapped[str] = mapped_column(Text)
    profile_json: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    owner: Mapped[User] = relationship(back_populates="datasets")
    runs: Mapped[list[TrainingRun]] = relationship(back_populates="dataset")

    @property
    def profile(self) -> list[dict[str, Any]]:
        return json.loads(self.profile_json)


class TrainingRun(Base):
    __tablename__ = "runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    owner_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id"), index=True)
    label: Mapped[str] = mapped_column(String(200), default="")
    status: Mapped[str] = mapped_column(String(20), default="queued", index=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    config_json: Mapped[str] = mapped_column(Text)
    loss: Mapped[str] = mapped_column(String(10))
    test_fraction: Mapped[float] = mapped_column(Float, default=0.2)

    history_json: Mapped[str] = mapped_column(Text, default="[]")
    metrics_json: Mapped[str] = mapped_column(Text, default="{}")
    weights_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    # The encoding learned at training time. Prediction reuses it verbatim so a new row
    # is scaled and one-hot encoded exactly like the training rows were.
    encoding_json: Mapped[str | None] = mapped_column(Text, nullable=True)

    param_count: Mapped[int] = mapped_column(Integer, default=0)
    train_rows: Mapped[int] = mapped_column(Integer, default=0)
    test_rows: Mapped[int] = mapped_column(Integer, default=0)
    duration_ms: Mapped[int] = mapped_column(Integer, default=0)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    owner: Mapped[User] = relationship(back_populates="runs")
    dataset: Mapped[Dataset] = relationship(back_populates="runs")

    @property
    def config(self) -> dict[str, Any]:
        return json.loads(self.config_json)

    @property
    def history(self) -> list[dict[str, Any]]:
        return json.loads(self.history_json)

    @property
    def metrics(self) -> dict[str, Any]:
        return json.loads(self.metrics_json)
