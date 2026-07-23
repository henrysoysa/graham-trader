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
