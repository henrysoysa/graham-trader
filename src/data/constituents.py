"""Live index-constituent fetching with disk caching.

Provides full, up-to-date membership for developed-market indexes that publish a
clean constituents table (Wikipedia). Design goals:

* **Config-driven** — each index is one entry in ``INDEX_SOURCES``.
* **Resilient parsing** — instead of hard-coding a table index/column name
  (which drifts), we scan every table on the page for a plausible ticker column
  and pick the largest matching table, then sanity-check the row count.
* **Cached with periodic refresh** — results are cached to disk and only
  re-fetched when older than ``MAX_AGE_DAYS``; a temporarily unreachable source
  falls back to the (possibly stale) cache, then to the caller's curated list.

Anything not in ``INDEX_SOURCES`` (most emerging/frontier indexes, ADR baskets)
keeps using its curated list unchanged.
"""
from __future__ import annotations

import io
import json
import logging
import re
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

MAX_AGE_DAYS = 7
_USER_AGENT = "Mozilla/5.0 (graham-trader constituents fetcher)"

# Column headers (lower-cased) that typically hold the exchange ticker.
_TICKER_HEADER_HINTS = ("ticker", "symbol", "epic", "code")

# Registry of index -> live source.
#   url:        page to fetch
#   suffix:     yfinance exchange suffix to append (US = "")
#   min_count:  minimum plausible constituents; a smaller parse is treated as a
#               mis-parse and we fall back rather than return garbage.
INDEX_SOURCES: Dict[str, Dict] = {
    # United States (no suffix; class shares use '-', handled in _clean_symbol)
    "NASDAQ100": {"url": "https://en.wikipedia.org/wiki/Nasdaq-100", "suffix": "", "min_count": 90},
    "DOW30": {"url": "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average", "suffix": "", "min_count": 28},
    # United Kingdom (LSE, .L)
    "FTSE100": {"url": "https://en.wikipedia.org/wiki/FTSE_100_Index", "suffix": ".L", "min_count": 90},
    "FTSE250": {"url": "https://en.wikipedia.org/wiki/FTSE_250_Index", "suffix": ".L", "min_count": 200},
    # Germany (XETRA, .DE)
    "DAX": {"url": "https://en.wikipedia.org/wiki/DAX", "suffix": ".DE", "min_count": 38},
    # France (Euronext Paris, .PA)
    "CAC40": {"url": "https://en.wikipedia.org/wiki/CAC_40", "suffix": ".PA", "min_count": 38},
    # Australia (ASX, .AX)
    "ASX200": {"url": "https://en.wikipedia.org/wiki/S%26P/ASX_200", "suffix": ".AX", "min_count": 180},
    # Canada (TSX, .TO)
    "TSX60": {"url": "https://en.wikipedia.org/wiki/S%26P/TSX_60", "suffix": ".TO", "min_count": 55},
    # Switzerland (SIX, .SW)
    "SMI_SWISS": {"url": "https://en.wikipedia.org/wiki/Swiss_Market_Index", "suffix": ".SW", "min_count": 18},
    # Japan (TSE, .T)
    "NIKKEI225": {"url": "https://en.wikipedia.org/wiki/Nikkei_225", "suffix": ".T", "min_count": 200},
    # India (NSE, .NS / BSE, .BO)
    "INDIA_NIFTY50": {"url": "https://en.wikipedia.org/wiki/NIFTY_50", "suffix": ".NS", "min_count": 48},
    "INDIA_SENSEX": {"url": "https://en.wikipedia.org/wiki/BSE_SENSEX", "suffix": ".BO", "min_count": 28},
    # US small caps. The "official" source would be the iShares Russell 2000
    # ETF (IWM) daily holdings CSV, but iShares sits behind Akamai Bot Manager
    # (a JS challenge, not just a consent cookie) and blocks every plain HTTP
    # client — including curl_cffi with browser TLS impersonation. Rather than
    # silently falling back to a 90-name curated list on every single run, we
    # approximate Russell 2000 membership from SEC's full filer list, filtered
    # to the small/mid-cap market-cap band the index actually covers. It won't
    # match official membership name-for-name, but it's a real, current
    # universe of ~1500-2000+ names instead of a static ~90.
    "RUSSELL2000": {
        "source": "sec_market_cap",
        "min_market_cap": 300e6,
        "max_market_cap": 10e9,
        # SEC's full filer list is ~10k tickers; probing every one of them
        # with a yfinance .info call is a multi-minute scan even threaded.
        # Cap how many candidates get probed so a cold cache still resolves
        # in reasonable time — matched names are added to disk cache, so
        # later runs skip straight to the fresh-cache path in get_constituents.
        "max_candidates": 3000,
        "suffix": "",
        "min_count": 500,
    },
}

# sec_market_cap sources cache for longer than the Wikipedia-table sources:
# each refresh costs one yfinance .info call per SEC filer (~10k), so refresh
# weekly rather than matching MAX_AGE_DAYS' default cadence.
SEC_MARKET_CAP_MAX_AGE_DAYS = 7


def _cache_dir() -> Optional[Path]:
    try:
        d = Path.home() / ".graham_trader" / "constituents"
        d.mkdir(parents=True, exist_ok=True)
        return d
    except Exception as e:  # pragma: no cover - home not writable
        logger.debug(f"constituents cache unavailable: {e}")
        return None


def _cache_path(key: str) -> Optional[Path]:
    d = _cache_dir()
    return (d / f"{key}.json") if d else None


def _read_cache(key: str) -> Optional[Dict]:
    p = _cache_path(key)
    if not p or not p.exists():
        return None
    try:
        with open(p, "r") as fh:
            return json.load(fh)
    except Exception:
        return None


def _write_cache(key: str, tickers: List[str]) -> None:
    p = _cache_path(key)
    if not p:
        return
    try:
        with open(p, "w") as fh:
            json.dump({"fetched_at": time.time(), "tickers": tickers}, fh)
    except Exception as e:  # pragma: no cover
        logger.debug(f"could not write constituents cache for {key}: {e}")


def _is_fresh(entry: Dict, max_age_days: int) -> bool:
    return (time.time() - entry.get("fetched_at", 0)) < max_age_days * 86400


def _flatten_columns(df) -> List[str]:
    cols = []
    for c in df.columns:
        if isinstance(c, tuple):
            c = " ".join(str(x) for x in c if str(x) != "nan")
        cols.append(str(c).strip())
    return cols


def _clean_symbol(raw: str, suffix: str) -> Optional[str]:
    s = re.sub(r"\[.*?\]", "", str(raw)).strip()  # drop footnote markers
    if not s or s.lower() == "nan":
        return None
    # Strip an exchange prefix like "ETR: ADS" or "TYO: 7203" -> keep the code.
    if ":" in s:
        s = s.split(":")[-1].strip()
    s = s.split()[0] if s.split() else s          # drop trailing text in the cell
    # Numeric codes (e.g. Tokyo 7203) can be read as floats -> "7203.0".
    if s.endswith(".0") and s[:-2].isdigit():
        s = s[:-2]
    s = s.replace(".", "-")         # class shares: BRK.B -> BRK-B, BT.A -> BT-A
    if not s or s in {"-", "nan"}:
        return None
    if suffix and not s.endswith(suffix):
        s = s + suffix
    return s


def _pick_symbol_column(df):
    """Return the column in ``df`` most likely to hold tickers, or None."""
    flat = _flatten_columns(df)
    df.columns = flat
    for hint in _TICKER_HEADER_HINTS:
        for col in flat:
            if hint == col.lower() or col.lower().endswith(hint) or hint in col.lower():
                return col
    return None


def _fetch_live(spec: Dict) -> List[str]:
    """Fetch and parse constituents for one source spec, by source type.

    Raises on failure so the caller can fall back to cache / curated list.
    """
    if spec.get("source") == "ishares_csv":
        return _fetch_ishares_holdings(spec)
    if spec.get("source") == "sec_market_cap":
        return _fetch_sec_market_cap(spec)
    return _fetch_wikipedia(spec)


def _get_sec_cik_map() -> Dict[str, str]:
    """SEC publishes one JSON file mapping ticker -> CIK for every filer."""
    import requests

    url = "https://www.sec.gov/files/company_tickers.json"
    headers = {"User-Agent": _USER_AGENT}
    resp = requests.get(url, headers=headers, timeout=20)
    resp.raise_for_status()
    data = resp.json()
    return {v["ticker"].upper(): str(v["cik_str"]).zfill(10) for v in data.values()}


def _sec_market_cap_lookup(ticker: str, min_mc: float, max_mc: Optional[float]) -> Optional[str]:
    import yfinance as yf

    try:
        info = yf.Ticker(ticker).info
        mc = info.get("marketCap")
    except Exception:
        return None
    if not mc or mc < min_mc:
        return None
    if max_mc and mc > max_mc:
        return None
    return ticker


def _fetch_sec_market_cap(spec: Dict) -> List[str]:
    """Approximate a market-cap-banded universe (e.g. Russell 2000) from SEC's
    full filer list, filtered by live market cap via yfinance.

    This still costs ~one yfinance call per SEC filer (~10k), so it's run in
    a thread pool (I/O-bound, GIL-released during the network call) and only
    when the cache is stale — ``get_constituents`` handles that. Not official
    index membership, a market-cap-band proxy for it.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    min_mc = spec.get("min_market_cap", 0)
    max_mc = spec.get("max_market_cap")
    cik_map = _get_sec_cik_map()
    # Skip obvious non-common-stock symbols (warrants, units, rights, etc.)
    # that would otherwise burn a request each with little chance of matching.
    # SEC's company_tickers.json is roughly market-cap ordered (largest
    # first), so capping here still favors well-known names over penny stocks.
    tickers = [t for t in cik_map if re.fullmatch(r"[A-Z][A-Z0-9\-]{0,6}", t)]
    max_candidates = spec.get("max_candidates")
    if max_candidates:
        tickers = tickers[:max_candidates]

    out: List[str] = []
    done = 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {
            pool.submit(_sec_market_cap_lookup, t, min_mc, max_mc): t for t in tickers
        }
        for future in as_completed(futures):
            done += 1
            if done % 500 == 0:
                logger.info(f"SEC market-cap scan: {done}/{len(tickers)} ({len(out)} matched so far)")
            result = future.result()
            if result:
                out.append(result)
    out.sort()
    return out


def _looks_like_html(text: str) -> bool:
    head = text.lstrip()[:400].lower()
    return head.startswith("<!doctype") or "<html" in head


def _fetch_ishares_holdings(spec: Dict) -> List[str]:
    """Parse the full equity holdings from an iShares ETF holdings CSV.

    iShares gates the CSV behind a region/investor-type consent wall and will
    otherwise return its HTML web page (even with a text/csv Content-Type). We
    therefore prime consent cookies from the product page first and ask
    explicitly for CSV; if an HTML page still comes back we raise so the caller
    falls back cleanly rather than mis-parsing a web page.

    The file has a metadata preamble before the real header row (the one that
    starts with 'Ticker'); we skip to it, then keep only equity rows.
    """
    import pandas as pd
    import requests

    session = requests.Session()
    session.headers.update({
        "User-Agent": _USER_AGENT,
        "Accept": "text/csv,application/csv,application/octet-stream,*/*",
    })
    product_url = spec.get("product_url")
    if product_url:
        try:  # best-effort: sets consent/region cookies
            session.get(product_url, timeout=20)
        except Exception:
            pass

    resp = session.get(spec["url"], timeout=30)
    resp.raise_for_status()
    if _looks_like_html(resp.text):
        raise ValueError(
            "iShares returned an HTML page instead of the holdings CSV "
            "(the download is gated); using fallback."
        )
    lines = resp.text.splitlines()

    header_idx = None
    for i, line in enumerate(lines):
        first_cell = line.split(",")[0].strip().strip('"').lower()
        if first_cell == "ticker":
            header_idx = i
            break
    if header_idx is None:
        raise ValueError("iShares holdings header row not found")

    df = pd.read_csv(io.StringIO("\n".join(lines[header_idx:])))
    cols = {str(c).strip().lower(): c for c in df.columns}
    tcol = cols.get("ticker")
    if tcol is None:
        raise ValueError("no Ticker column in holdings file")

    # Keep only equity positions (drop cash, futures, FX, etc.).
    acol = cols.get("asset class")
    if acol is not None:
        df = df[df[acol].astype(str).str.strip().str.lower() == "equity"]

    suffix = spec.get("suffix", "")
    seen, out = set(), []
    for raw in df[tcol].tolist():
        sym = _clean_symbol(raw, suffix)
        if not sym or "_" in sym or sym == "-":
            continue  # placeholder / cash rows
        if sym not in seen:
            seen.add(sym)
            out.append(sym)
    return out


def _fetch_wikipedia(spec: Dict) -> List[str]:
    """Fetch and parse a Wikipedia constituents table. Raises on failure."""
    import pandas as pd
    import requests

    resp = requests.get(spec["url"], headers={"User-Agent": _USER_AGENT}, timeout=20)
    resp.raise_for_status()
    html = resp.text

    # Prefer a table explicitly id'd 'constituents' (S&P/Nasdaq style), else all.
    tables = []
    try:
        tables = pd.read_html(io.StringIO(html), attrs={"id": "constituents"})
    except Exception:
        tables = []
    if not tables:
        tables = pd.read_html(io.StringIO(html))

    best, best_col = None, None
    for t in tables:
        col = _pick_symbol_column(t)
        if col is not None and (best is None or len(t) > len(best)):
            best, best_col = t, col
    if best is None:
        raise ValueError("no constituent table with a ticker column found")

    suffix = spec.get("suffix", "")
    seen, out = set(), []
    for raw in best[best_col].tolist():
        sym = _clean_symbol(raw, suffix)
        if sym and sym not in seen:
            seen.add(sym)
            out.append(sym)
    return out


def get_constituents(
    key: str,
    fallback: Callable[[], List[str]],
    max_age_days: int = MAX_AGE_DAYS,
) -> List[str]:
    """Return full constituents for ``key``.

    Order of preference: fresh cache -> live fetch (sanity-checked) -> stale
    cache -> ``fallback()`` (the curated hard-coded list).
    """
    tickers, _source = get_constituents_with_source(key, fallback, max_age_days)
    return tickers


def get_constituents_with_source(
    key: str,
    fallback: Callable[[], List[str]],
    max_age_days: int = MAX_AGE_DAYS,
) -> "tuple[List[str], str]":
    """Same as ``get_constituents``, but also reports where the list came
    from: ``"fresh_cache"``, ``"live"``, ``"stale_cache"``, or ``"fallback"``.

    Callers that want to warn the user when they've silently landed on the
    small curated list (e.g. the Streamlit UI) should use this instead of
    ``get_constituents`` and check for ``source == "fallback"``.
    """
    spec = INDEX_SOURCES.get(key)
    if spec is None:
        return fallback(), "fallback"

    if spec.get("source") == "sec_market_cap":
        max_age_days = max(max_age_days, SEC_MARKET_CAP_MAX_AGE_DAYS)

    cached = _read_cache(key)
    if cached and _is_fresh(cached, max_age_days) and cached.get("tickers"):
        return cached["tickers"], "fresh_cache"

    try:
        tickers = _fetch_live(spec)
        if len(tickers) >= spec.get("min_count", 1):
            _write_cache(key, tickers)
            logger.info(f"Fetched {len(tickers)} live constituents for {key}")
            return tickers, "live"
        logger.warning(
            f"Live constituents for {key} looked implausible "
            f"({len(tickers)} < {spec.get('min_count')}); falling back"
        )
    except Exception as e:
        logger.warning(f"Live constituents fetch failed for {key}: {e}")

    if cached and cached.get("tickers"):
        logger.info(f"Using cached (stale) constituents for {key}")
        return cached["tickers"], "stale_cache"

    return fallback(), "fallback"
