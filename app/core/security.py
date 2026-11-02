"""Password hashing and JWT issuing/verification."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import bcrypt
import jwt

from app.core.config import Settings

BCRYPT_ROUNDS = 12


class TokenError(Exception):
    """Raised when a token is missing, malformed, expired or signed with the wrong key."""


def hash_password(password: str) -> str:
    """Hash a password with bcrypt. bcrypt truncates past 72 bytes, so we guard first."""
    if not password:
        raise ValueError("Password must not be empty")
    encoded = password.encode("utf-8")
    if len(encoded) > 72:
        raise ValueError("Password must be 72 bytes or fewer")
    return bcrypt.hashpw(encoded, bcrypt.gensalt(rounds=BCRYPT_ROUNDS)).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    """Constant-time-ish comparison via bcrypt.checkpw. Never raises on bad input."""
    if not password or not password_hash:
        return False
    try:
        encoded = password.encode("utf-8")
        if len(encoded) > 72:
            return False
        return bcrypt.checkpw(encoded, password_hash.encode("utf-8"))
    except (ValueError, TypeError):
        return False


def create_access_token(
    subject: str, settings: Settings, extra: dict[str, Any] | None = None
) -> str:
    """Issue a signed JWT whose `sub` is the user id."""
    now = datetime.now(UTC)
    payload: dict[str, Any] = {
        "sub": subject,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=settings.access_token_ttl_minutes)).timestamp()),
        "typ": "access",
    }
    if extra:
        payload.update(extra)
    return jwt.encode(payload, settings.secret_key, algorithm=settings.algorithm)


def decode_access_token(token: str, settings: Settings) -> dict[str, Any]:
    """Verify signature + expiry and return the claims. Raises TokenError on any problem."""
    if not token:
        raise TokenError("Missing token")
    try:
        claims = jwt.decode(token, settings.secret_key, algorithms=[settings.algorithm])
    except jwt.ExpiredSignatureError as exc:
        raise TokenError("Token has expired") from exc
    except jwt.InvalidTokenError as exc:
        raise TokenError("Token is invalid") from exc

    if claims.get("typ") != "access":
        raise TokenError("Wrong token type")
    if not claims.get("sub"):
        raise TokenError("Token has no subject")
    return claims
