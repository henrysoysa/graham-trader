"""Tests for the shared stock-universe configuration.

These guard the bug that made the app 'always select the US': every market
label must map to a real DataFetcher index key, and every page draws from this
one source of truth.
"""
import logging

import pytest

from src.universe import (
    UNIVERSE_OPTIONS,
    UNIVERSE_MAPPING,
    is_separator,
    resolve_index_key,
)
from src.data.data_fetcher import DataFetcher


def _selectable_market_options():
    """Options that should resolve to a real index (not separators/custom)."""
    return [
        o for o in UNIVERSE_OPTIONS
        if not is_separator(o) and o != "Custom Tickers"
    ]


def test_every_market_option_is_mapped():
    """No selectable market may be missing from the mapping."""
    unmapped = [o for o in _selectable_market_options() if o not in UNIVERSE_MAPPING]
    assert unmapped == [], f"Market options with no index key: {unmapped}"


def test_mapping_has_no_orphans():
    """Every mapping entry must correspond to a visible dropdown option."""
    orphans = [label for label in UNIVERSE_MAPPING if label not in UNIVERSE_OPTIONS]
    assert orphans == [], f"Mapping labels not shown in the dropdown: {orphans}"


def test_resolve_index_key_behaviour():
    assert resolve_index_key("S&P 500 (USA)") == "SP500"
    assert resolve_index_key("India - NIFTY 50") == "INDIA_NIFTY50"
    # Non-market rows resolve to None (caller handles them explicitly).
    assert resolve_index_key("Custom Tickers") is None
    assert resolve_index_key("--- Developed Markets (Large Cap) ---") is None


def test_multiple_non_us_markets_are_offered():
    """Guard against silently regressing to a US-only list."""
    keys = set(UNIVERSE_MAPPING.values())
    for non_us in ("INDIA_NIFTY50", "BRAZIL_BOVESPA", "SAUDI_TASI", "TURKEY_BIST100"):
        assert non_us in keys


def test_every_mapped_key_resolves_without_fallback(caplog):
    """Each mapped key must be understood by the DataFetcher.

    If a key is unknown, get_index_tickers logs an 'Unknown index' warning and
    silently returns the S&P 500. Asserting the warning never fires for a
    mapped key is exactly what prevents the 'always US' class of bug.
    """
    fetcher = DataFetcher()
    for label, key in UNIVERSE_MAPPING.items():
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="src.data.data_fetcher"):
            tickers = fetcher.get_index_tickers(key)
        assert tickers, f"{label} ({key}) returned no tickers"
        assert "Unknown index" not in caplog.text, (
            f"{label} ({key}) fell through to the US fallback"
        )
