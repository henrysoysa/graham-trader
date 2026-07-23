"""Tests for the Graham Number math and the screening / margin-of-safety gate."""
import math

import numpy as np
import pytest

import config
from src.data.data_fetcher import DataFetcher
from src.screening.graham_criteria import GrahamScreener
from conftest import StubFetcher


def test_default_criteria_come_from_config():
    """No-config screener must use the same thresholds config.py exposes,
    so strict/modern values can't silently diverge."""
    screener = GrahamScreener(DataFetcher())
    assert screener.criteria["defensive"]["max_pe_ratio"] == config.GRAHAM_CRITERIA["defensive"]["max_pe_ratio"]
    assert screener.criteria["defensive"]["max_pb_ratio"] == config.GRAHAM_CRITERIA["defensive"]["max_pb_ratio"]
    assert screener.criteria == config.GRAHAM_CRITERIA


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
        "earnings_growth": 0.10,
    }


# A clean, steadily-growing, all-positive earnings history.
GROWING_EPS = [3.0, 3.5, 4.0, 4.5, 5.0]


def _graham(graham_number, mos):
    return {"graham_number": graham_number, "margin_of_safety": mos}


def test_defensive_deep_value_passes():
    fetcher = StubFetcher(_deep_value_metrics(), _graham(47.43, 0.37), GROWING_EPS)
    screener = GrahamScreener(fetcher, CRITERIA)
    res = screener._evaluate_defensive("X")
    assert res["margin_of_safety_check"] is True
    assert res["earnings_stability_check"] is True
    assert res["earnings_growth_check"] is True
    assert res["total_score"] == 10           # now a 10-criterion scorecard
    assert res["max_score"] == 10
    assert res["passes_screening"] is True


def test_defensive_overvalued_fails_on_margin_of_safety_gate():
    """Strong ratios but trading ABOVE intrinsic value must NOT pass.

    This is the regression guard for the margin-of-safety gate: without it,
    an expensive stock could pass on the other criteria alone.
    """
    fetcher = StubFetcher(_deep_value_metrics(), _graham(25.0, -0.20), GROWING_EPS)
    screener = GrahamScreener(fetcher, CRITERIA)
    res = screener._evaluate_defensive("X")
    assert res["margin_of_safety_check"] is False
    assert res["total_score"] >= 7            # other criteria still pass
    assert res["passes_screening"] is False   # ...but the gate blocks it


def test_defensive_nan_margin_of_safety_does_not_pass():
    fetcher = StubFetcher(_deep_value_metrics(), _graham(0, float("nan")), GROWING_EPS)
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
# Earnings-stability check (multi-year, no annual losses)
# --------------------------------------------------------------------------

def _screener(earnings_history):
    fetcher = StubFetcher(_deep_value_metrics(), _graham(47.43, 0.37), earnings_history)
    return GrahamScreener(fetcher, CRITERIA)


def test_stability_all_positive_history_passes():
    assert _screener([1.0, 2.0, 3.0])._earnings_stability_ok("X", 10, 3.0) is True


def test_stability_fails_with_a_loss_year():
    # A single negative year breaks stability even if the latest year is positive.
    assert _screener([2.0, -0.5, 3.0])._earnings_stability_ok("X", 10, 3.0) is False


def test_stability_falls_back_to_trailing_eps_when_no_history():
    s = _screener(None)
    assert s._earnings_stability_ok("X", 10, 4.0) is True
    assert s._earnings_stability_ok("X", 10, -1.0) is False


# --------------------------------------------------------------------------
# Earnings-growth check (was previously dead config)
# --------------------------------------------------------------------------

def test_growth_strong_history_passes():
    # ~13%/yr, comfortably above the ~2.9%/yr annualised 33%-over-10y target.
    assert _screener([3.0, 3.5, 4.0, 4.5, 5.0])._earnings_growth_ok("X", 0.33, 0.0) is True


def test_growth_flat_history_fails():
    assert _screener([5.0, 5.0, 5.0])._earnings_growth_ok("X", 0.33, 0.0) is False


def test_growth_declining_history_fails():
    assert _screener([6.0, 5.0, 4.0])._earnings_growth_ok("X", 0.33, 0.0) is False


def test_growth_falls_back_to_trailing_growth_when_history_short():
    s = _screener([5.0])          # only one point -> use trailing figure
    assert s._earnings_growth_ok("X", 0.33, 0.08) is True
    assert s._earnings_growth_ok("X", 0.33, -0.02) is False


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
