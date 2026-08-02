"""Process-wide throttling and retry/backoff for yfinance calls.

Yahoo Finance's rate limit is undocumented and enforced per-IP with no
published quota or reset window (see yfinance's ``YFRateLimitError`` — it
carries no ``Retry-After`` and the client doesn't parse one). Two things
make it easy to trip even at modest volume: firing requests back-to-back
with no spacing, and running several requests concurrently (e.g. a
thread-pooled scan). This module gives every yfinance call in the app a
single shared choke point so scans stay polite regardless of which code
path triggers them.

Usage:
    from .yf_throttle import throttled_call

    info = throttled_call(lambda: yf.Ticker(ticker).info)
"""
from __future__ import annotations

import logging
import random
import threading
import time
from typing import Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Minimum spacing enforced between the *start* of consecutive yfinance
# requests, process-wide (all threads share this clock). 0.5s -> at most
# ~2 req/s, which is conservative enough to avoid tripping Yahoo's limiter
# at the volumes this app scans (hundreds to low thousands of tickers).
MIN_REQUEST_INTERVAL_SECONDS = 0.5

MAX_RETRIES = 3
# Backoff on a 429 specifically: this is a much longer pause than the
# regular request spacing above, since a 429 means the limiter has already
# been tripped and hammering it again immediately just extends the block.
BACKOFF_BASE_SECONDS = 5.0
BACKOFF_MAX_SECONDS = 60.0


class _RateLimiter:
    """Thread-safe minimum-interval gate shared by every caller."""

    def __init__(self, min_interval: float):
        self._min_interval = min_interval
        self._lock = threading.Lock()
        self._next_allowed_at = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            sleep_for = self._next_allowed_at - now
            if sleep_for > 0:
                time.sleep(sleep_for)
                now = time.monotonic()
            self._next_allowed_at = now + self._min_interval


_limiter = _RateLimiter(MIN_REQUEST_INTERVAL_SECONDS)


def _is_rate_limit_error(exc: Exception) -> bool:
    if type(exc).__name__ == "YFRateLimitError":
        return True
    text = str(exc).lower()
    return "too many requests" in text or "rate limit" in text or " 429" in text


def throttled_call(fn: Callable[[], T], *, context: str = "") -> T:
    """Run ``fn`` (a zero-arg callable making one yfinance request) behind
    the shared rate limiter, retrying with backoff if Yahoo returns a 429.

    Any other exception from ``fn`` propagates immediately — callers already
    handle per-ticker failures (missing data, bad symbol, etc.) themselves.
    """
    last_exc: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        _limiter.wait()
        try:
            return fn()
        except Exception as e:
            if not _is_rate_limit_error(e):
                raise
            last_exc = e
            if attempt == MAX_RETRIES:
                break
            backoff = min(BACKOFF_BASE_SECONDS * (2 ** (attempt - 1)), BACKOFF_MAX_SECONDS)
            backoff += random.uniform(0, backoff * 0.25)  # jitter
            logger.warning(
                f"Yahoo rate limit hit{' for ' + context if context else ''} "
                f"(attempt {attempt}/{MAX_RETRIES}); backing off {backoff:.1f}s"
            )
            time.sleep(backoff)
    assert last_exc is not None
    raise last_exc
