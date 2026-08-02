"""Tests for the per-stock Graham backtesting logic (pure functions)."""
import math

import numpy as np
import pandas as pd
import pytest

from src.backtesting.graham_backtest import (
    benchmark_for_ticker,
    compute_graham_signal_series,
    GrahamBacktester,
)


# --------------------------------------------------------------------------
# Benchmark resolution by exchange suffix
# --------------------------------------------------------------------------

@pytest.mark.parametrize("ticker,symbol,matched", [
    ("AAPL", "^GSPC", True),          # US, no suffix
    ("RELIANCE.NS", "^NSEI", True),   # India NSE
    ("SHEL.L", "^FTSE", True),        # UK
    ("BHP.AX", "^AXJO", True),        # Australia
    ("7203.T", "^N225", True),        # Japan
    ("PETR4.SA", "^BVSP", True),      # Brazil
    ("RY.TO", "^GSPTSE", True),       # Canada
    ("2222.SR", "^TASI.SR", True),    # Saudi Arabia
])
def test_benchmark_for_ticker_known(ticker, symbol, matched):
    b = benchmark_for_ticker(ticker)
    assert b["symbol"] == symbol
    assert b["matched"] is matched


def test_benchmark_for_ticker_unknown_suffix():
    b = benchmark_for_ticker("XYZ.ZZ")
    assert b["matched"] is False
    assert b["symbol"] is None
    assert b["suffix"] == "ZZ"


def test_benchmark_is_case_insensitive():
    assert benchmark_for_ticker("reliance.ns")["symbol"] == "^NSEI"


# --------------------------------------------------------------------------
# Graham signal reconstruction
# --------------------------------------------------------------------------

def _prices(values, start="2020-01-01"):
    idx = pd.date_range(start, periods=len(values), freq="D")
    return pd.Series(values, index=idx, dtype="float64")


def test_signal_series_computes_graham_number_and_entry_flags():
    # Constant fundamentals: EPS=5, BVPS=20 -> Graham Number = sqrt(2250) ~ 47.43
    prices = _prices([30.0, 40.0, 50.0])
    fundamentals = pd.DataFrame(
        {"eps_ttm": [5.0], "bvps": [20.0]},
        index=pd.to_datetime(["2019-06-01"]),  # known before the price window
    )
    out = compute_graham_signal_series(prices, fundamentals, mos_threshold=0.20)

    gn = math.sqrt(2250)
    assert out["graham_number"].round(2).eq(round(gn, 2)).all()
    # Price 30 -> MoS ~0.367 (>=0.20) => entry; 40 -> ~0.156 (<0.20) not entry;
    # 50 -> negative => not entry.
    assert list(out["is_entry"]) == [True, False, False]


def test_signal_series_forward_fills_fundamentals_by_report_date():
    prices = _prices([10.0, 10.0, 10.0, 10.0], start="2021-01-01")
    # A new, higher book value is reported partway through.
    fundamentals = pd.DataFrame(
        {"eps_ttm": [2.0, 4.0], "bvps": [5.0, 10.0]},
        index=pd.to_datetime(["2021-01-01", "2021-01-03"]),
    )
    out = compute_graham_signal_series(prices, fundamentals, mos_threshold=0.0)
    # First two days use the first report, last two use the second.
    gn_early = math.sqrt(22.5 * 2.0 * 5.0)
    gn_late = math.sqrt(22.5 * 4.0 * 10.0)
    assert out["graham_number"].iloc[0] == pytest.approx(gn_early, rel=1e-6)
    assert out["graham_number"].iloc[-1] == pytest.approx(gn_late, rel=1e-6)


def test_signal_series_handles_missing_fundamentals():
    prices = _prices([10.0, 11.0])
    out = compute_graham_signal_series(prices, pd.DataFrame(), mos_threshold=0.33)
    assert out["graham_number"].isna().all()
    assert not out["is_entry"].any()          # no signal without a Graham Number
    assert list(out["price"]) == [10.0, 11.0]  # price still passes through


def test_signal_series_ignores_negative_fundamentals():
    prices = _prices([10.0])
    fundamentals = pd.DataFrame(
        {"eps_ttm": [-1.0], "bvps": [5.0]},
        index=pd.to_datetime(["2019-01-01"]),
    )
    out = compute_graham_signal_series(prices, fundamentals)
    assert math.isnan(out["graham_number"].iloc[0])
    assert out["is_entry"].iloc[0] == False


def test_signal_series_empty_prices():
    out = compute_graham_signal_series(pd.Series(dtype="float64"), pd.DataFrame())
    assert out.empty


# --------------------------------------------------------------------------
# GrahamBacktester wiring (with a fake data fetcher)
# --------------------------------------------------------------------------

class _PriceFetcher:
    """Returns canned OHLC frames keyed by symbol."""

    def __init__(self, frames):
        self._frames = frames

    def get_historical_prices(self, symbol, start_date=None, end_date=None, period="5y"):
        return self._frames.get(symbol, pd.DataFrame())


def test_relative_performance_normalises_and_matches_benchmark():
    idx = pd.date_range("2020-01-01", periods=3, freq="D")
    frames = {
        "AAPL": pd.DataFrame({"Close": [100.0, 110.0, 120.0]}, index=idx),
        "^GSPC": pd.DataFrame({"Close": [200.0, 210.0, 220.0]}, index=idx),
    }
    bt = GrahamBacktester(_PriceFetcher(frames))
    res = bt.relative_performance("AAPL", period="5y")
    assert res["benchmark_name"] == "S&P 500"
    data = res["data"]
    # Both rebased to 100 at the start.
    assert data["stock"].iloc[0] == pytest.approx(100.0)
    assert data["benchmark"].iloc[0] == pytest.approx(100.0)
    assert data["stock"].iloc[-1] == pytest.approx(120.0)
    assert data["benchmark"].iloc[-1] == pytest.approx(110.0)


def test_relative_performance_without_benchmark_data():
    idx = pd.date_range("2020-01-01", periods=2, freq="D")
    frames = {"AAPL": pd.DataFrame({"Close": [100.0, 105.0]}, index=idx)}  # no ^GSPC
    bt = GrahamBacktester(_PriceFetcher(frames))
    res = bt.relative_performance("AAPL")
    assert "stock" in res["data"].columns
    assert "benchmark" not in res["data"].columns


# --------------------------------------------------------------------------
# ADR share-count mismatch: the Graham Entry Analysis page reconstructs BVPS
# from balance-sheet equity / balance-sheet shares outstanding, but for a
# foreign ADR that share count is denominated in the *ordinary* (home-market)
# shares, not the ADR itself — dividing ADR-level equity by that count
# produces a BVPS wrong by whatever the ADR ratio is (see ADOOY: ~55x too
# small, which made a genuinely undervalued stock look wildly overvalued).
# --------------------------------------------------------------------------

class _StatementFetcher:
    """Fake DataFetcher: canned prices, financial statements, and info."""

    def __init__(self, prices, income, balance, info):
        self._prices = prices
        self._income = income
        self._balance = balance
        self._info = info

    def get_historical_prices(self, symbol, start_date=None, end_date=None, period="5y"):
        return self._prices.get(symbol, pd.DataFrame())

    def get_financial_statements(self, ticker):
        return {"income_statement": self._income, "balance_sheet": self._balance}

    def get_stock_info(self, ticker):
        return self._info


def _adr_statements():
    dates = [pd.Timestamp("2024-12-31"), pd.Timestamp("2025-12-31")]
    income = pd.DataFrame(
        {dates[0]: {"Diluted EPS": 0.80}, dates[1]: {"Diluted EPS": 0.86}}
    )
    # Equity in ADR-scale dollars; shares in ordinary (home-market) count —
    # ~54x larger than the ADR's own sharesOutstanding, mirroring ADOOY.
    balance = pd.DataFrame(
        {
            dates[0]: {"Stockholders Equity": 4_926_856_000.0, "Ordinary Shares Number": 30_244_918_400.0},
            dates[1]: {"Stockholders Equity": 4_485_423_000.0, "Ordinary Shares Number": 28_800_494_200.0},
        }
    )
    return income, balance


def test_historical_fundamentals_patches_latest_bvps_on_adr_share_mismatch():
    income, balance = _adr_statements()
    info = {"sharesOutstanding": 576_009_884, "bookValue": 8.5, "trailingEps": 0.86}
    fetcher = _StatementFetcher({}, income, balance, info)
    bt = GrahamBacktester(fetcher)

    fundamentals = bt._historical_fundamentals("ADOOY")

    # Most recent point overridden with yfinance's quoted-security bvps/eps.
    latest = fundamentals.iloc[-1]
    assert latest["bvps"] == pytest.approx(8.5)
    assert latest["eps_ttm"] == pytest.approx(0.86)

    # Prior year is left as the (still statement-derived) reconstruction —
    # only the latest point is patched, since that's what info reflects.
    prior = fundamentals.iloc[0]
    assert prior["bvps"] == pytest.approx(4_926_856_000.0 / 30_244_918_400.0)


def test_historical_fundamentals_leaves_bvps_alone_when_share_counts_agree():
    income, balance = _adr_statements()
    # sharesOutstanding close to the balance sheet's count (no ADR-style
    # mismatch) -> statement-derived bvps should stand, un-patched.
    info = {"sharesOutstanding": 28_900_000_000, "bookValue": 999.0, "trailingEps": 999.0}
    fetcher = _StatementFetcher({}, income, balance, info)
    bt = GrahamBacktester(fetcher)

    fundamentals = bt._historical_fundamentals("NORMALCO")

    latest = fundamentals.iloc[-1]
    assert latest["bvps"] == pytest.approx(4_485_423_000.0 / 28_800_494_200.0)
    assert latest["bvps"] != pytest.approx(999.0)


def test_graham_entry_history_end_to_end_matches_screener_style_values():
    """The bug as reported: Entry Analysis's most recent margin-of-safety
    should agree with a Screener-style calculation using yfinance's own
    bookValue/trailingEps, not the statement-derived (mismatched) BVPS."""
    income, balance = _adr_statements()
    info = {"sharesOutstanding": 576_009_884, "bookValue": 8.5, "trailingEps": 0.86}
    idx = pd.date_range("2026-07-01", periods=3, freq="D")
    prices = {"ADOOY": pd.DataFrame({"Close": [7.12, 7.12, 7.12]}, index=idx)}
    fetcher = _StatementFetcher(prices, income, balance, info)
    bt = GrahamBacktester(fetcher)

    signals = bt.graham_entry_history("ADOOY", period="5y")
    last = signals.iloc[-1]

    expected_graham_number = (22.5 * 0.86 * 8.5) ** 0.5
    assert last["graham_number"] == pytest.approx(expected_graham_number)
    expected_mos = (expected_graham_number - 7.12) / expected_graham_number
    assert last["margin_of_safety"] == pytest.approx(expected_mos)
    assert last["is_entry"] == (expected_mos >= 0.33)
