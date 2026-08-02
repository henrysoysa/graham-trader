"""Tests for the live index-constituents framework (network-free)."""
import time

import pytest

from src.data import constituents as C
from src.data.data_fetcher import DataFetcher


def _html_table(symbols, colname="Symbol", table_id=None):
    idattr = f' id="{table_id}"' if table_id else ""
    header = f"<tr><th>{colname}</th><th>Company</th></tr>"
    rows = "".join(f"<tr><td>{s}</td><td>Example Corp</td></tr>" for s in symbols)
    return f"<table{idattr}>{header}{rows}</table>"


class _Resp:
    def __init__(self, text):
        self.text = text

    def raise_for_status(self):
        return None


@pytest.fixture(autouse=True)
def _isolate_cache(tmp_path, monkeypatch):
    """Point the cache at a temp dir so tests never touch the real home cache."""
    monkeypatch.setattr(C, "_cache_dir", lambda: tmp_path)


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------

def test_fetch_live_parses_and_normalises_us(monkeypatch):
    import requests
    symbols = [f"SYM{i}" for i in range(100)]
    symbols[0] = "BRK.B"
    monkeypatch.setattr(requests, "get", lambda *a, **k: _Resp(_html_table(symbols)))
    out = C._fetch_live({"url": "http://x", "suffix": "", "min_count": 90})
    assert len(out) == 100
    assert "BRK-B" in out and "BRK.B" not in out


def test_fetch_live_appends_exchange_suffix(monkeypatch):
    import requests
    monkeypatch.setattr(requests, "get", lambda *a, **k: _Resp(_html_table(["ADS", "BMW"], colname="Ticker")))
    out = C._fetch_live({"url": "http://x", "suffix": ".DE", "min_count": 1})
    assert out == ["ADS.DE", "BMW.DE"]


def test_clean_symbol_handles_exchange_prefix_and_float_codes():
    assert C._clean_symbol("ETR: ADS", ".DE") == "ADS.DE"
    assert C._clean_symbol("7203.0", ".T") == "7203.T"   # float-read Tokyo code
    assert C._clean_symbol("BT.A", ".L") == "BT-A.L"     # class share
    assert C._clean_symbol("AAPL", "") == "AAPL"
    assert C._clean_symbol("nan", ".DE") is None


def test_fetch_live_picks_largest_table_with_tickers(monkeypatch):
    import requests
    small = _html_table(["AAA", "BBB"], colname="Symbol", table_id="changes")
    big = _html_table([f"T{i}" for i in range(60)], colname="Symbol", table_id="constituents")
    # Note: 'constituents' id is tried first, so the big table is found directly.
    monkeypatch.setattr(requests, "get", lambda *a, **k: _Resp(small + big))
    out = C._fetch_live({"url": "http://x", "suffix": "", "min_count": 50})
    assert len(out) == 60


# --------------------------------------------------------------------------
# Caching / fallback ordering
# --------------------------------------------------------------------------

def _ishares_csv(equities, extra_rows=0):
    """Build an iShares-style holdings CSV with a metadata preamble."""
    preamble = (
        '"iShares Russell 2000 ETF"\n'
        '"Fund Holdings as of","Jul 22, 2026"\n'
        '"Inception Date","May 22, 2000"\n'
        '\xa0\n'
    )
    header = "Ticker,Name,Sector,Asset Class,Weight (%)\n"
    rows = "".join(f"{t},{t} Inc,Industrials,Equity,0.05\n" for t in equities)
    # Non-equity rows that must be filtered out.
    cash = "".join(
        f"CASH{i},US Dollar,Cash,Cash and/or Derivatives,0.01\n" for i in range(extra_rows)
    )
    margin = "-,Margin Balance,Cash,Cash and/or Derivatives,0.00\n"
    return preamble + header + rows + cash + margin


def _fake_session_factory(holdings_text):
    """Factory returning a fake requests.Session: the product page primes
    cookies (returns HTML), the .ajax holdings URL returns holdings_text."""
    class _Session:
        def __init__(self):
            self.headers = {}

        def get(self, url, timeout=None):
            if url.endswith("/"):          # product page
                return _Resp("<html>consent</html>")
            return _Resp(holdings_text)    # holdings download

    return _Session


_ISHARES_SPEC = {
    "source": "ishares_csv",
    "url": "http://x/1467271812596.ajax?fileType=csv",
    "product_url": "http://x/",
    "suffix": "",
    "min_count": 1000,
}


def test_fetch_ishares_holdings_parses_full_equity_list(monkeypatch):
    import requests
    equities = [f"SC{i}" for i in range(1500)]
    equities[0] = "BRK.B"  # a dotted class ticker to normalise
    csv_text = _ishares_csv(equities, extra_rows=5)
    monkeypatch.setattr(requests, "Session", _fake_session_factory(csv_text))

    out = C._fetch_live(_ISHARES_SPEC)  # dispatches to the iShares parser
    assert len(out) == 1500
    assert "BRK-B" in out
    assert not any(t.startswith("CASH") for t in out)  # cash rows filtered
    assert "-" not in out


def test_fetch_ishares_holdings_raises_on_html_interstitial(monkeypatch):
    """If iShares serves its HTML page instead of the CSV, we must raise so the
    caller falls back to the curated list rather than mis-parsing a web page."""
    import requests
    html = "<!DOCTYPE html><html><head></head><body>iShares</body></html>"
    monkeypatch.setattr(requests, "Session", _fake_session_factory(html))
    with pytest.raises(ValueError):
        C._fetch_live(_ISHARES_SPEC)


def test_russell2000_uses_its_configured_live_source(monkeypatch):
    monkeypatch.setattr(C, "_fetch_live", lambda spec: [f"SC{i}" for i in range(1800)])
    d = DataFetcher()
    out = d.get_index_tickers("RUSSELL2000")
    assert len(out) == 1800


def test_russell2000_falls_back_to_curated_when_source_down(monkeypatch):
    def _raise(spec):
        raise RuntimeError("source unreachable")

    monkeypatch.setattr(C, "_fetch_live", _raise)
    d = DataFetcher()
    out = d.get_index_tickers("RUSSELL2000")
    # curated Russell 2000 fallback (~90 small caps), not empty
    assert 50 <= len(out) <= 150
    assert all(isinstance(t, str) for t in out)


# --------------------------------------------------------------------------
# SEC market-cap-banded source (RUSSELL2000's live source: iShares' official
# holdings CSV sits behind Akamai bot-detection that blocks every plain HTTP
# client, so the small/mid-cap band is approximated from SEC filers instead)
# --------------------------------------------------------------------------

def test_fetch_sec_market_cap_filters_by_band(monkeypatch):
    cik_map = {f"T{i}": str(i).zfill(10) for i in range(6)}
    caps = {"T0": 50e9, "T1": 5e9, "T2": 1e9, "T3": 300e6, "T4": 100e6, "T5": None}
    monkeypatch.setattr(C, "_get_sec_cik_map", lambda: cik_map)

    class _FakeTicker:
        def __init__(self, t):
            self.t = t

        @property
        def info(self):
            cap = caps.get(self.t)
            return {"marketCap": cap} if cap is not None else {}

    import yfinance
    monkeypatch.setattr(yfinance, "Ticker", _FakeTicker)

    spec = {"min_market_cap": 300e6, "max_market_cap": 10e9}
    out = C._fetch_sec_market_cap(spec)
    # T0 too big, T4 too small, T5 has no marketCap -> excluded.
    assert sorted(out) == ["T1", "T2", "T3"]


def test_fetch_sec_market_cap_respects_max_candidates(monkeypatch):
    cik_map = {f"T{i}": str(i).zfill(10) for i in range(20)}
    monkeypatch.setattr(C, "_get_sec_cik_map", lambda: cik_map)

    seen = []

    class _FakeTicker:
        def __init__(self, t):
            seen.append(t)
            self.t = t

        @property
        def info(self):
            return {"marketCap": 1e9}  # always in-band

    import yfinance
    monkeypatch.setattr(yfinance, "Ticker", _FakeTicker)

    spec = {"min_market_cap": 300e6, "max_market_cap": 10e9, "max_candidates": 5}
    out = C._fetch_sec_market_cap(spec)
    assert len(out) == 5
    assert len(seen) == 5  # never probed beyond the cap


def test_fetch_sec_market_cap_respects_skip_ranks(monkeypatch):
    """skip_ranks jumps past the market-cap-descending head of SEC's list
    (mega/large-caps) before applying max_candidates, so probe budget lands
    on the band actually being searched for instead of names already known
    to be out of range."""
    cik_map = {f"T{i}": str(i).zfill(10) for i in range(20)}
    monkeypatch.setattr(C, "_get_sec_cik_map", lambda: cik_map)

    seen = []

    class _FakeTicker:
        def __init__(self, t):
            seen.append(t)
            self.t = t

        @property
        def info(self):
            return {"marketCap": 1e9}

    import yfinance
    monkeypatch.setattr(yfinance, "Ticker", _FakeTicker)

    spec = {"min_market_cap": 300e6, "max_market_cap": 10e9, "skip_ranks": 10, "max_candidates": 5}
    out = C._fetch_sec_market_cap(spec)
    assert seen == [f"T{i}" for i in range(10, 15)]  # T0..T9 skipped entirely
    assert len(out) == 5


def test_fetch_sec_market_cap_skips_non_equity_looking_symbols(monkeypatch):
    cik_map = {"AAPL": "1", "BRK-B": "2", "SOMEWARRANT.WS": "3", "1234567": "4"}
    monkeypatch.setattr(C, "_get_sec_cik_map", lambda: cik_map)

    seen = []

    class _FakeTicker:
        def __init__(self, t):
            seen.append(t)
            self.t = t

        @property
        def info(self):
            return {"marketCap": 1e9}

    import yfinance
    monkeypatch.setattr(yfinance, "Ticker", _FakeTicker)

    spec = {"min_market_cap": 300e6, "max_market_cap": 10e9}
    C._fetch_sec_market_cap(spec)
    assert "SOMEWARRANT.WS" not in seen
    assert "1234567" not in seen
    assert "AAPL" in seen and "BRK-B" in seen


def test_dispatches_to_sec_market_cap_source(monkeypatch):
    monkeypatch.setattr(C, "_fetch_sec_market_cap", lambda spec: [f"SC{i}" for i in range(1200)])
    out = C._fetch_live({"source": "sec_market_cap", "min_market_cap": 300e6, "max_market_cap": 10e9})
    assert len(out) == 1200


def test_fresh_cache_is_used_without_fetching(monkeypatch):
    C._write_cache("DAX", ["CACHED.DE"])

    def _boom(*a, **k):
        raise AssertionError("should not fetch when cache is fresh")

    monkeypatch.setattr(C, "_fetch_live", _boom)
    out = C.get_constituents("DAX", fallback=lambda: ["FALLBACK.DE"])
    assert out == ["CACHED.DE"]


def test_stale_cache_triggers_refetch(monkeypatch):
    # Write a stale cache entry (8 days old).
    p = C._cache_path("DAX")
    import json
    with open(p, "w") as fh:
        json.dump({"fetched_at": time.time() - 8 * 86400, "tickers": ["OLD.DE"]}, fh)

    monkeypatch.setattr(C, "_fetch_live", lambda spec: [f"NEW{i}.DE" for i in range(40)])
    out = C.get_constituents("DAX", fallback=lambda: ["FALLBACK.DE"])
    assert out[0] == "NEW0.DE" and len(out) == 40


def test_implausible_parse_falls_back(monkeypatch):
    # A parse returning fewer than min_count is treated as a mis-parse.
    monkeypatch.setattr(C, "_fetch_live", lambda spec: ["ONLY.DE"])
    out = C.get_constituents("DAX", fallback=lambda: ["FALLBACK.DE"])
    assert out == ["FALLBACK.DE"]


def test_fetch_error_falls_back_to_curated(monkeypatch):
    def _raise(spec):
        raise RuntimeError("network down")

    monkeypatch.setattr(C, "_fetch_live", _raise)
    out = C.get_constituents("DAX", fallback=lambda: ["FALLBACK.DE"])
    assert out == ["FALLBACK.DE"]


def test_fetch_error_prefers_stale_cache_over_curated(monkeypatch):
    p = C._cache_path("DAX")
    import json
    with open(p, "w") as fh:
        json.dump({"fetched_at": time.time() - 30 * 86400, "tickers": ["STALE.DE"]}, fh)

    monkeypatch.setattr(C, "_fetch_live", lambda spec: (_ for _ in ()).throw(RuntimeError("down")))
    out = C.get_constituents("DAX", fallback=lambda: ["FALLBACK.DE"])
    assert out == ["STALE.DE"]


def test_unsourced_key_uses_fallback():
    # BRAZIL_BOVESPA has no live source -> always the curated list.
    out = C.get_constituents("BRAZIL_BOVESPA", fallback=lambda: ["PETR4.SA"])
    assert out == ["PETR4.SA"]


# --------------------------------------------------------------------------
# get_constituents_with_source / get_index_tickers_with_source
#
# These let a caller (the Streamlit UI) tell a genuine live scan apart from a
# silent drop to the small curated fallback list, so it can warn the user
# instead of quietly serving a ~90-name list under a "found N stocks" banner.
# --------------------------------------------------------------------------

def test_with_source_reports_live_on_success(monkeypatch):
    monkeypatch.setattr(C, "_fetch_live", lambda spec: [f"NEW{i}.DE" for i in range(40)])
    out, source = C.get_constituents_with_source("DAX", fallback=lambda: ["FALLBACK.DE"])
    assert source == "live" and len(out) == 40


def test_with_source_reports_fresh_cache(monkeypatch):
    C._write_cache("DAX", ["CACHED.DE"])
    monkeypatch.setattr(C, "_fetch_live", lambda spec: (_ for _ in ()).throw(AssertionError("no fetch")))
    out, source = C.get_constituents_with_source("DAX", fallback=lambda: ["FALLBACK.DE"])
    assert source == "fresh_cache" and out == ["CACHED.DE"]


def test_with_source_reports_stale_cache(monkeypatch):
    p = C._cache_path("DAX")
    import json
    with open(p, "w") as fh:
        json.dump({"fetched_at": time.time() - 30 * 86400, "tickers": ["STALE.DE"]}, fh)
    monkeypatch.setattr(C, "_fetch_live", lambda spec: (_ for _ in ()).throw(RuntimeError("down")))
    out, source = C.get_constituents_with_source("DAX", fallback=lambda: ["FALLBACK.DE"])
    assert source == "stale_cache" and out == ["STALE.DE"]


def test_with_source_reports_fallback_when_everything_fails(monkeypatch):
    monkeypatch.setattr(C, "_fetch_live", lambda spec: (_ for _ in ()).throw(RuntimeError("down")))
    out, source = C.get_constituents_with_source("DAX", fallback=lambda: ["FALLBACK.DE"])
    assert source == "fallback" and out == ["FALLBACK.DE"]


def test_with_source_reports_fallback_for_unsourced_key():
    out, source = C.get_constituents_with_source("BRAZIL_BOVESPA", fallback=lambda: ["PETR4.SA"])
    assert source == "fallback" and out == ["PETR4.SA"]


def test_get_constituents_still_returns_plain_list(monkeypatch):
    """Backward compatibility: existing callers using get_constituents (not
    the _with_source variant) must keep getting a plain list back."""
    monkeypatch.setattr(C, "_fetch_live", lambda spec: [f"X{i}.DE" for i in range(40)])
    out = C.get_constituents("DAX", fallback=lambda: ["FALLBACK.DE"])
    assert isinstance(out, list) and len(out) == 40


# --------------------------------------------------------------------------
# Integration with DataFetcher.get_index_tickers
# --------------------------------------------------------------------------

def test_get_index_tickers_uses_live_source_for_sourced_key(monkeypatch):
    monkeypatch.setattr(C, "_fetch_live", lambda spec: [f"X{i}.DE" for i in range(40)])
    d = DataFetcher()
    out = d.get_index_tickers("DAX")
    assert len(out) == 40 and out[0] == "X0.DE"


def test_get_index_tickers_curated_for_unsourced_key():
    # Vietnam VN30 is not wired live; returns its hard-coded curated list.
    d = DataFetcher()
    out = d.get_index_tickers("VIETNAM_VN30")
    assert out and all(isinstance(t, str) for t in out)
    assert "VNM" in out  # from the curated list


def test_get_index_tickers_with_source_reports_live(monkeypatch):
    monkeypatch.setattr(C, "_fetch_live", lambda spec: [f"X{i}.DE" for i in range(40)])
    d = DataFetcher()
    out, source = d.get_index_tickers_with_source("DAX")
    assert source == "live" and len(out) == 40


def test_get_index_tickers_with_source_reports_fallback_on_failure(monkeypatch):
    monkeypatch.setattr(C, "_fetch_live", lambda spec: (_ for _ in ()).throw(RuntimeError("down")))
    d = DataFetcher()
    out, source = d.get_index_tickers_with_source("RUSSELL2000")
    assert source == "fallback"
    assert 50 <= len(out) <= 150  # curated Russell 2000 sample


def test_get_index_tickers_with_source_reports_fallback_for_unknown_key():
    d = DataFetcher()
    out, source = d.get_index_tickers_with_source("NOT_A_REAL_INDEX")
    assert source == "fallback"
    assert out  # falls back to S&P 500
