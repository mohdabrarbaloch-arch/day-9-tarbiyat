"""Pydantic request/response models. Validation lives here, not in the routers."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator

# --------------------------------------------------------------------------------------
# Auth
# --------------------------------------------------------------------------------------


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=72)
    display_name: str = Field(default="", max_length=120)

    @field_validator("password")
    @classmethod
    def _strength(cls, value: str) -> str:
        if value.isdigit():
            raise ValueError("Password must not be all digits")
        if len(value.encode("utf-8")) > 72:
            raise ValueError("Password must be 72 bytes or fewer")
        return value


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=72)


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    email: str
    display_name: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserOut


# --------------------------------------------------------------------------------------
# Datasets
# --------------------------------------------------------------------------------------


class ColumnOut(BaseModel):
    name: str
    kind: str
    missing: int = 0
    mean: float | None = None
    std: float | None = None
    min: float | None = None
    max: float | None = None
    categories: list[str] = Field(default_factory=list)


class DatasetOut(BaseModel):
    id: int
    name: str
    target: str
    task: str
    fingerprint: str
    row_count: int
    column_count: int
    created_at: str
    profile: list[ColumnOut] = Field(default_factory=list)


class DatasetPreview(BaseModel):
    columns: list[str]
    rows: list[list[str]]


# --------------------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------------------


class TrainRequest(BaseModel):
    dataset_id: int = Field(gt=0)
    label: str = Field(default="", max_length=200)
    hidden_layers: list[int] = Field(default_factory=lambda: [16], min_length=1, max_length=4)
    activation: Literal["relu", "tanh", "sigmoid"] = "relu"
    init: Literal["he", "xavier", "normal"] = "he"
    optimizer: Literal["sgd", "momentum", "adam"] = "adam"
    loss: Literal["mse", "bce", "cce"] | None = None
    learning_rate: float = Field(default=0.05, gt=0, le=2.0)
    epochs: int = Field(default=120, ge=1, le=400)
    batch_size: int = Field(default=16, ge=1, le=512)
    l2: float = Field(default=0.0, ge=0, le=0.5)
    test_fraction: float = Field(default=0.2, ge=0.05, le=0.5)
    early_stopping: bool = False
    patience: int = Field(default=12, ge=2, le=100)
    seed: int = Field(default=42, ge=0, le=2**31 - 1)

    @field_validator("hidden_layers")
    @classmethod
    def _widths(cls, value: list[int]) -> list[int]:
        if any(w < 1 or w > 256 for w in value):
            raise ValueError("Each hidden layer width must be between 1 and 256")
        return value


class EpochOut(BaseModel):
    epoch: int
    train_loss: float
    train_accuracy: float | None = None
    val_loss: float | None = None
    val_accuracy: float | None = None


class ClassMetricOut(BaseModel):
    label: str
    precision: float
    recall: float
    f1: float
    support: float


class RunOut(BaseModel):
    id: int
    dataset_id: int
    label: str
    status: str
    error: str | None = None
    loss: str
    config: dict
    param_count: int
    train_rows: int
    test_rows: int
    duration_ms: int
    created_at: str
    history: list[EpochOut] = Field(default_factory=list)
    metrics: dict = Field(default_factory=dict)


class RunSummary(BaseModel):
    id: int
    dataset_id: int
    dataset_name: str = ""
    label: str
    status: str
    loss: str
    score: float | None = None
    score_kind: str = ""
    epochs_run: int = 0
    param_count: int
    duration_ms: int
    created_at: str


class PredictRequest(BaseModel):
    rows: list[dict[str, float | str]]


class PredictionOut(BaseModel):
    index: int
    predicted: str | float
    confidence: float | None = None
    probabilities: dict[str, float] = Field(default_factory=dict)


class PredictResponse(BaseModel):
    task: str
    predictions: list[PredictionOut]


class HealthOut(BaseModel):
    status: str
    version: str
    environment: str
