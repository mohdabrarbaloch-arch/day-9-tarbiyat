"""Application settings, loaded from the environment."""

from __future__ import annotations

from functools import lru_cache

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DEV_SECRET = "dev-only-insecure-secret-change-me"


class Settings(BaseSettings):
    """Every knob the app reads from the environment lives here."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "Tarbiyat"
    environment: str = "development"
    debug: bool = False

    # Auth
    secret_key: str = DEV_SECRET
    algorithm: str = "HS256"
    access_token_ttl_minutes: int = 60 * 24

    # Storage
    database_url: str = "sqlite:///./tarbiyat.db"

    # Network / CORS
    cors_origins: str = "*"

    # Uploads
    max_upload_bytes: int = 5_000_000
    max_dataset_rows: int = 20_000
    max_dataset_columns: int = 60

    # Training guardrails — these exist so one user cannot pin the CPU forever.
    max_epochs: int = 400
    max_hidden_layers: int = 4
    max_layer_width: int = 256
    max_concurrent_runs: int = 2
    run_timeout_seconds: int = 180

    # Rate limiting
    rate_limit_auth: str = "10/minute"
    rate_limit_write: str = "60/minute"

    # Seeded demo account, created on first boot in development only.
    seed_demo_user: bool = True
    demo_email: str = "demo@tarbiyat.pk"
    demo_password: str = "tarbiyat-demo-2026"

    @field_validator("secret_key")
    @classmethod
    def _reject_default_in_production(cls, value: str) -> str:
        return value

    @property
    def is_production(self) -> bool:
        return self.environment.lower() in {"production", "prod"}

    @property
    def cors_origin_list(self) -> list[str]:
        raw = self.cors_origins.strip()
        if raw == "*":
            return ["*"]
        return [origin.strip() for origin in raw.split(",") if origin.strip()]

    def validate_for_boot(self) -> None:
        """Fail loudly rather than silently shipping a dev secret to production."""
        if self.is_production and self.secret_key == DEV_SECRET:
            raise RuntimeError(
                "SECRET_KEY is still the development default. Set a real SECRET_KEY "
                "before running with ENVIRONMENT=production."
            )


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.validate_for_boot()
    return settings
