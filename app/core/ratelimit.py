"""A small fixed-window rate limiter.

Scope and honesty: this keeps its counters in process memory. That is correct for a
single worker and wrong for a multi-worker deployment, where each worker would enforce
its own copy of the quota. `docs/setup.md` says so and points at Redis for production.
Guessing that it "just works" under uvicorn --workers 4 is how rate limits quietly stop
rate-limiting.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque

from fastapi import HTTPException, Request, status

WINDOW_SECONDS = 60.0


class RateLimiter:
    """Per-key sliding-window limiter with an LRU-ish sweep to bound memory."""

    def __init__(self, max_keys: int = 20_000) -> None:
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()
        self._max_keys = max_keys
        self._last_sweep = time.monotonic()

    @staticmethod
    def _parse(spec: str) -> tuple[int, float]:
        """Parse '10/minute' or '120/hour' into (count, window_seconds)."""
        try:
            count_part, unit = spec.strip().split("/")
            count = int(count_part)
        except (ValueError, AttributeError) as exc:
            raise ValueError(f"Bad rate limit spec: {spec!r}") from exc
        windows = {"second": 1.0, "minute": 60.0, "hour": 3600.0, "day": 86400.0}
        if unit not in windows:
            raise ValueError(f"Unknown rate limit unit: {unit!r}")
        if count < 1:
            raise ValueError("Rate limit count must be >= 1")
        return count, windows[unit]

    @staticmethod
    def client_key(request: Request) -> str:
        """Identify the caller. Behind a proxy this must be X-Forwarded-For, not the socket."""
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    def _sweep(self, now: float) -> None:
        if len(self._hits) < self._max_keys:
            return
        if now - self._last_sweep < WINDOW_SECONDS:
            return
        self._last_sweep = now
        for key in [k for k, v in self._hits.items() if not v or v[-1] < now - 3600]:
            self._hits.pop(key, None)

    def check(self, request: Request, spec: str) -> None:
        """Raise 429 if the caller has exceeded `spec`. Otherwise record the hit."""
        limit, window = self._parse(spec)
        key = f"{self.client_key(request)}:{request.url.path}"
        now = time.monotonic()

        with self._lock:
            self._sweep(now)
            bucket = self._hits[key]
            while bucket and bucket[0] < now - window:
                bucket.popleft()
            if len(bucket) >= limit:
                retry_after = int(window - (now - bucket[0])) + 1
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=f"Too many requests. Try again in {retry_after}s.",
                    headers={"Retry-After": str(retry_after)},
                )
            bucket.append(now)

    def reset(self) -> None:
        """Clear all counters. Used by tests so one case cannot poison the next."""
        with self._lock:
            self._hits.clear()
