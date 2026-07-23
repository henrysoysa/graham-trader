"""Tests for the Graham Number math and the screening / margin-of-safety gate."""
import math

import numpy as np
import pytest

from src.data.data_fetcher import DataFetcher
from src.screening.graham_criteria import GrahamScreener
from conftest import StubFetcher


CRITERIA = {
    "defensive": {
        "min_earnings_stability": 10,
        "min_dividend_history": 10,
        "min_earnings_growth": 0.33,
        "max_pe_ratio": 25,
        "max_pb_ratio": 4.0,
        "min_current_ratio": 1.2,
        "max_debt_to_current_assets": 1.5,
        "margin_of_safety": 0.20,
    },
    "enterprising": {
        "min_earnings_stability": 5,
        "max_pe_ratio": 35,
        "max_pb_ratio": 5.0,
        "min_current_ratio": 1.0,
        "margin_of_safety": 0.15,
    },
}


# --------------------------------------------------------------------------
# Graham Number math
# --------------------------------------------------------------------------

def test_graham_number_and_margin_of_safety(monkeypatch):
    d = DataFetcher()
    monkeypatch.setattr(d, "get_key_metrics", lambda t: {
        "eps": 5.0, "book_value_per_share": 20.0, "current_price": 30.0,
    })
    res = d.calculate_graham_number("X")
    # sqrt(22.5 * 5 * 20) = sqrt(2250) = 47.434...
    assert res["graham_number"] == pytest.approx(math.sqrt(2250), rel=1e-6)
    assert res["margin_of_safety"] == pytest.approx((res["graham_number"] - 30) / res["graham_number"], rel=1e-6)
    assert bool(res["is_undervalued"]) is True  # ~36.7% discount > 33%


def test_graham_number_undefined_for_negative_inputs(monkeypatch):
    d = DataFetcher()
    monkeypatch.setattr(d, "get_key_metrics", lambda t: {
        "eps": -1.0, "book_value_per_share": 20.0, "current_price": 30.0,
    })
    res = d.calculate_graham_number("X")
    assert res["graham_number"] == 0
    # MoS must be NaN (not 0) so it is never read as "priced at fair value".
    assert math.isnan(res["margin_of_safety"])
    assert bool(res["is_undervalued"]) is False


# --------------------------------------------------------------------------
# Defensive screening
# --------------------------------------------------------------------------

def _deep_value_metrics():
    return {
        "current_price": 30.0, "market_cap": 5_000_000_000, "eps": 5.0,
        "pe_ratio": 10.0, "pb_ratio": 1.2, "current_ratio": 2.0,
        "debt_to_current_assets": 0.5, "dividend_yield": 0.03,
    }


def _graham(graham_number, mos):
    return {"graham_number": graham_number, "margin_of_safety": mos}


def test_defensive_deep_value_passes():
    fetcher = StubFetcher(_deep_value_metrics(), _graham(47.43, 0.37))
    screener = GrahamScreener(fetcher, CRITERIA)
    res = screener._evaluate_defensive("X")
    assert res["margin_of_safety_check"] is True
    assert res["total_score"] == 9
    assert res["passes_screening"] is True


def test_defensive_overvalued_fails_on_margin_of_safety_gate():
    """Strong ratios but trading ABOVE intrinsic value must NOT pass.

    This is the regression guard for the margin-of-safety gate: without it,
    an expensive stock could pass on the other criteria alone.
    """
    fetcher = StubFetcher(_deep_value_metrics(), _graham(25.0, -0.20))
    screener = GrahamScreener(fetcher, CRITERIA)
    res = screener._evaluate_defensive("X")
    assert res["margin_of_safety_check"] is False
    assert res["total_score"] >= 6            # other criteria still pass
    assert res["passes_screening"] is False   # ...but the gate blocks it


def test_defensive_nan_margin_of_safety_does_not_pass():
    fetcher = StubFetcher(_deep_value_metrics(), _graham(0, float("nan")))
    screener = GrahamScreener(fetcher, CRITERIA)
    res = screener._evaluate_defensive("X")
    assert res["margin_of_safety_check"] is False
    assert res["passes_screening"] is False


def test_screen_skips_stocks_without_price():
    fetcher = StubFetcher({"current_price": 0}, _graham(0, float("nan")))
    screener = GrahamScreener(fetcher, CRITERIA)
    assert screener._evaluate_defensive("X") is None
    # And the public API returns an empty frame rather than raising.
    df = screener.screen_defensive(["X"])
    assert df.empty


# --------------------------------------------------------------------------
# Enterprising screening
# --------------------------------------------------------------------------

def test_enterprising_requires_margin_of_safety():
    metrics = {
        "current_price": 30.0, "market_cap": 2_000_000_000, "eps": 5.0,
        "pe_ratio": 12.0, "pb_ratio": 2.0, "current_ratio": 1.5,
    }
    # Passes everything except the discount requirement.
    fetcher = StubFetcher(metrics, _graham(25.0, -0.20))
    screener = GrahamScreener(fetcher, CRITERIA)
    res = screener._evaluate_enterprising("X")
    assert res["margin_of_safety_check"] is False
    assert res["passes_screening"] is False

    # Now give it a real discount -> it should pass.
    fetcher2 = StubFetcher(metrics, _graham(47.43, 0.37))
    res2 = GrahamScreener(fetcher2, CRITERIA)._evaluate_enterprising("X")
    assert res2["margin_of_safety_check"] is True
    assert res2["passes_screening"] is True
