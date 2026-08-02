"""Tests for the shared yfinance rate-limit throttle/backoff."""
import time

import pytest

from src.data import yf_throttle as T


@pytest.fixture(autouse=True)
def _fast_backoff(monkeypatch):
    """Backoff sleeps would make retry tests slow; shrink them but keep the
    shape of the logic (still multiple real, distinct sleep calls)."""
    monkeypatch.setattr(T, "BACKOFF_BASE_SECONDS", 0.01)
    monkeypatch.setattr(T, "BACKOFF_MAX_SECONDS", 0.05)


@pytest.fixture(autouse=True)
def _fresh_limiter(monkeypatch):
    """Each test gets its own limiter so timing from other tests can't leak
    in via the shared module-level instance's ``_next_allowed_at``."""
    monkeypatch.setattr(T, "_limiter", T._RateLimiter(T.MIN_REQUEST_INTERVAL_SECONDS))


def test_successful_call_returns_value():
    assert T.throttled_call(lambda: 42) == 42


def test_non_rate_limit_error_propagates_immediately():
    calls = []

    def _boom():
        calls.append(1)
        raise ValueError("No data found for symbol")

    with pytest.raises(ValueError):
        T.throttled_call(_boom)
    assert len(calls) == 1  # no retry for a non-rate-limit error


def test_retries_on_rate_limit_error_then_succeeds(monkeypatch):
    monkeypatch.setattr(T, "MIN_REQUEST_INTERVAL_SECONDS", 0)
    monkeypatch.setattr(T, "_limiter", T._RateLimiter(0))

    calls = {"n": 0}

    def _flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise Exception("Too Many Requests. Rate limited. Try after a while.")
        return "ok"

    result = T.throttled_call(_flaky)
    assert result == "ok"
    assert calls["n"] == 3


def test_gives_up_after_max_retries():
    calls = {"n": 0}

    def _always_limited():
        calls["n"] += 1
        raise Exception("Too Many Requests. Rate limited. Try after a while.")

    with pytest.raises(Exception, match="Too Many Requests"):
        T.throttled_call(_always_limited)
    assert calls["n"] == T.MAX_RETRIES


def test_recognises_named_yfrate_limit_error():
    class YFRateLimitError(Exception):
        pass

    calls = {"n": 0}

    def _fn():
        calls["n"] += 1
        raise YFRateLimitError("whatever message")

    with pytest.raises(YFRateLimitError):
        T.throttled_call(_fn)
    assert calls["n"] == T.MAX_RETRIES  # treated as rate-limit -> retried


def test_limiter_enforces_minimum_spacing():
    limiter = T._RateLimiter(min_interval=0.05)
    start = time.monotonic()
    limiter.wait()
    limiter.wait()
    limiter.wait()
    elapsed = time.monotonic() - start
    # Three calls at 0.05s spacing -> at least ~0.1s between the 1st and 3rd.
    assert elapsed >= 0.09


def test_limiter_is_thread_safe_under_concurrent_callers():
    import threading

    limiter = T._RateLimiter(min_interval=0.02)
    call_times = []
    lock = threading.Lock()

    def _worker():
        limiter.wait()
        with lock:
            call_times.append(time.monotonic())

    threads = [threading.Thread(target=_worker) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    call_times.sort()
    gaps = [b - a for a, b in zip(call_times, call_times[1:])]
    # Every consecutive pair must respect the minimum interval, even though
    # calls were issued concurrently from separate threads.
    assert all(g >= 0.018 for g in gaps)  # small tolerance for scheduling jitter
