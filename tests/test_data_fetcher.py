"""Tests for DataFetcher index resolution and the unknown-key fallback."""
import logging

import pytest

from src.data.data_fetcher import DataFetcher


@pytest.fixture(scope="module")
def fetcher():
    return DataFetcher()


def test_sp500_returns_us_tickers(fetcher):
    tickers = fetcher.get_index_tickers("SP500")
    assert tickers
    assert "AAPL" in tickers


@pytest.mark.parametrize("key,suffix", [
    ("INDIA_NIFTY50", ".NS"),
    ("INDIA_SENSEX", ".BO"),
    ("BRAZIL_BOVESPA", ".SA"),
    ("SAUDI_TASI", ".SR"),
    ("TURKEY_BIST100", ".IS"),
    ("ASX200", ".AX"),
])
def test_market_returns_local_exchange_tickers(fetcher, key, suffix):
    """Non-US markets must return their own exchange-suffixed tickers,
    not a substituted US list."""
    tickers = fetcher.get_index_tickers(key)
    assert tickers
    assert all(t.endswith(suffix) for t in tickers), (
        f"{key} returned tickers without the {suffix} suffix: {tickers[:5]}"
    )
    # And they must not be the US fallback.
    assert "AAPL" not in tickers


def test_unknown_key_warns_and_falls_back(fetcher, caplog):
    with caplog.at_level(logging.WARNING, logger="src.data.data_fetcher"):
        tickers = fetcher.get_index_tickers("NOT_A_REAL_INDEX")
    assert tickers, "fallback should still return a usable list"
    assert "Unknown index" in caplog.text
    # The warning should help the developer by naming the bad key.
    assert "NOT_A_REAL_INDEX" in caplog.text


def test_legacy_sp500_helper_still_works(fetcher):
    assert fetcher.get_sp500_tickers() == fetcher.get_index_tickers("SP500")


def test_sp500_fetches_full_live_list(fetcher, monkeypatch):
    """When a live source is reachable, the full ~500 constituents are used
    (and class-suffix tickers are normalised, e.g. BRK.B -> BRK-B)."""
    import requests

    symbols = [f"SYM{i}" for i in range(500)]
    symbols[0] = "BRK.B"
    csv_text = "Symbol,Name\n" + "\n".join(f"{s},Example Corp" for s in symbols)

    class _Resp:
        def __init__(self, text, ok=True):
            self.text = text
            self._ok = ok

        def raise_for_status(self):
            if not self._ok:
                raise RuntimeError("blocked")

    def fake_get(url, *args, **kwargs):
        # Simulate Wikipedia being blocked; the CSV secondary source succeeds.
        if "wikipedia" in url:
            return _Resp("", ok=False)
        return _Resp(csv_text)

    monkeypatch.setattr(requests, "get", fake_get)
    tickers = fetcher._get_sp500_tickers()
    assert len(tickers) >= 400
    assert "BRK-B" in tickers
    assert "BRK.B" not in tickers


def test_sp500_falls_back_when_live_sources_fail(fetcher, monkeypatch):
    """If every live source fails, we still return the curated ~90 majors."""
    import requests

    def fail_get(*args, **kwargs):
        raise RuntimeError("no network")

    monkeypatch.setattr(requests, "get", fail_get)
    tickers = fetcher._get_sp500_tickers()
    assert "AAPL" in tickers
    assert 50 <= len(tickers) <= 130  # the curated fallback, not the full index
