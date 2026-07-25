"""Per-stock Graham backtesting.

Two capabilities, matching the product goals:

  (a) Relative performance — a stock vs. its home-country main index, so you can
      see whether it has beaten or lagged the market average over time.
  (b) Graham entry signals over time — reconstruct the Graham Number and margin
      of safety across ~5-10 years (at least one economic cycle) and mark the
      windows when the stock traded at a Graham-style discount.

Design notes
------------
The *pure* functions (``benchmark_for_ticker`` and ``compute_graham_signal_series``)
contain the logic and are unit-tested without any network access. The
``GrahamBacktester`` class wires them to the live DataFetcher.

Historical fundamentals are inherently limited by the free data source
(yfinance typically exposes only ~4 annual statements), so entry-signal
reconstruction is best-effort and the UI is explicit about how much history it
actually had to work with.
"""
from typing import Dict, Optional

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


# yfinance exchange suffix -> (benchmark index symbol, human name).
# The empty-string key covers US tickers, which carry no suffix.
BENCHMARK_MAP: Dict[str, tuple] = {
    "": ("^GSPC", "S&P 500"),          # United States
    # --- Asia-Pacific ---
    "NS": ("^NSEI", "NIFTY 50"),        # India (NSE)
    "BO": ("^BSESN", "BSE SENSEX"),     # India (BSE)
    "T": ("^N225", "Nikkei 225"),       # Japan
    "HK": ("^HSI", "Hang Seng"),        # Hong Kong
    "SS": ("000001.SS", "SSE Composite"),  # China (Shanghai)
    "SZ": ("399001.SZ", "SZSE Component"),  # China (Shenzhen)
    "KS": ("^KS11", "KOSPI"),           # South Korea
    "TW": ("^TWII", "TAIEX"),           # Taiwan
    "AX": ("^AXJO", "S&P/ASX 200"),     # Australia
    "SI": ("^STI", "Straits Times"),    # Singapore
    "JK": ("^JKSE", "IDX Composite"),   # Indonesia
    "BK": ("^SET.BK", "SET Index"),     # Thailand
    "KL": ("^KLSE", "FTSE Bursa KLCI"), # Malaysia
    "PS": ("PSEI.PS", "PSEi"),          # Philippines
    "VN": ("^VNINDEX", "VN-Index"),     # Vietnam (patchy coverage)
    "KA": ("^KSE", "KSE 100"),          # Pakistan
    # --- Europe ---
    "L": ("^FTSE", "FTSE 100"),         # United Kingdom
    "DE": ("^GDAXI", "DAX"),            # Germany
    "PA": ("^FCHI", "CAC 40"),          # France
    "MI": ("FTSEMIB.MI", "FTSE MIB"),   # Italy
    "MC": ("^IBEX", "IBEX 35"),         # Spain
    "AS": ("^AEX", "AEX"),              # Netherlands
    "SW": ("^SSMI", "SMI"),             # Switzerland
    "ST": ("^OMXS30", "OMX Stockholm 30"),  # Sweden
    "HE": ("^OMXH25", "OMX Helsinki 25"),   # Finland
    "OL": ("OSEBX.OL", "Oslo OBX"),     # Norway
    "CO": ("^OMXC25", "OMX Copenhagen 25"),  # Denmark
    "WA": ("WIG20.WA", "WIG20"),        # Poland
    "IS": ("XU100.IS", "BIST 100"),     # Turkey
    # --- Americas ---
    "TO": ("^GSPTSE", "S&P/TSX"),       # Canada
    "SA": ("^BVSP", "Bovespa"),         # Brazil
    "MX": ("^MXX", "IPC Mexico"),       # Mexico
    # --- Middle East / Africa ---
    "SR": ("^TASI.SR", "Tadawul TASI"), # Saudi Arabia
    "JO": ("^J203.JO", "JSE All Share"),  # South Africa
    "CA": ("^CASE30", "EGX 30"),        # Egypt
}


def benchmark_for_ticker(ticker: str) -> Dict[str, Optional[str]]:
    """Resolve a ticker to its home-country benchmark index.

    Returns a dict with ``symbol`` (yfinance index symbol or None), ``name``,
    ``suffix`` (the exchange suffix detected), and ``matched`` (whether a
    benchmark was found).
    """
    t = (ticker or "").upper().strip()
    suffix = t.rsplit(".", 1)[1] if "." in t else ""
    entry = BENCHMARK_MAP.get(suffix)
    if entry is None:
        return {"symbol": None, "name": None, "suffix": suffix, "matched": False}
    return {"symbol": entry[0], "name": entry[1], "suffix": suffix, "matched": True}


def compute_graham_signal_series(
    prices: pd.Series,
    fundamentals: pd.DataFrame,
    mos_threshold: float = 0.33,
    multiplier: float = 22.5,
) -> pd.DataFrame:
    """Reconstruct Graham Number / margin-of-safety over time.

    Args:
        prices: Close price Series indexed by date.
        fundamentals: DataFrame indexed by report date with columns
            ``eps_ttm`` and ``bvps`` (point-in-time fundamentals). These are
            forward-filled onto the price dates, since a figure is only known
            from its report date onward.
        mos_threshold: Margin of safety required to flag an "entry" window.
        multiplier: Graham Number multiplier (22.5 = 15 x 1.5).

    Returns:
        DataFrame indexed by ``prices.index`` with columns: price, eps_ttm,
        bvps, graham_number, margin_of_safety, is_entry.
    """
    if prices is None or len(prices) == 0:
        return pd.DataFrame(
            columns=["price", "eps_ttm", "bvps", "graham_number", "margin_of_safety", "is_entry"]
        )

    prices = prices.sort_index()
    out = pd.DataFrame(index=prices.index)
    out["price"] = prices.astype(float)

    if fundamentals is None or fundamentals.empty:
        out["eps_ttm"] = np.nan
        out["bvps"] = np.nan
    else:
        f = fundamentals.sort_index()
        # Forward-fill each fundamental onto the price timeline (known from the
        # report date onward), then align to the price index.
        aligned = f.reindex(out.index.union(f.index)).ffill().reindex(out.index)
        out["eps_ttm"] = aligned.get("eps_ttm", np.nan)
        out["bvps"] = aligned.get("bvps", np.nan)

    eps = out["eps_ttm"]
    bvps = out["bvps"]
    valid = (eps > 0) & (bvps > 0)
    out["graham_number"] = np.where(valid, np.sqrt(multiplier * eps.where(valid) * bvps.where(valid)), np.nan)
    out["margin_of_safety"] = (out["graham_number"] - out["price"]) / out["graham_number"]
    out["is_entry"] = out["margin_of_safety"] >= mos_threshold
    # Where Graham Number is undefined, there is no entry signal.
    out.loc[out["graham_number"].isna(), "is_entry"] = False
    return out


class GrahamBacktester:
    """Runs per-stock relative-performance and Graham-entry backtests."""

    def __init__(self, data_fetcher):
        self.data_fetcher = data_fetcher

    def relative_performance(self, ticker: str, period: str = "5y") -> Dict:
        """Stock vs. its home benchmark, both normalised to 100 at the start.

        Returns a dict with ``data`` (DataFrame: date index, columns 'stock' and
        'benchmark'), ``benchmark_name``, and ``benchmark_symbol``. ``data`` may
        be empty if prices could not be retrieved.
        """
        bench = benchmark_for_ticker(ticker)
        result = {
            "data": pd.DataFrame(),
            "benchmark_name": bench["name"],
            "benchmark_symbol": bench["symbol"],
            "matched": bench["matched"],
        }

        stock_px = self._close_series(ticker, period)
        if stock_px is None or stock_px.empty:
            return result

        frame = pd.DataFrame({"stock": stock_px})
        if bench["symbol"]:
            bench_px = self._close_series(bench["symbol"], period)
            if bench_px is not None and not bench_px.empty:
                frame["benchmark"] = bench_px

        frame = frame.dropna(how="all").sort_index()
        # Normalise each column to 100 at its first valid observation on the
        # common timeline.
        frame = frame.dropna()
        if frame.empty:
            result["data"] = frame
            return result
        normalised = frame / frame.iloc[0] * 100.0
        result["data"] = normalised
        return result

    def graham_entry_history(
        self, ticker: str, period: str = "10y", mos_threshold: float = 0.33
    ) -> pd.DataFrame:
        """Reconstruct Graham entry signals for ``ticker`` over ``period``."""
        prices = self._close_series(ticker, period)
        if prices is None or prices.empty:
            return pd.DataFrame(
                columns=["price", "eps_ttm", "bvps", "graham_number", "margin_of_safety", "is_entry"]
            )
        fundamentals = self._historical_fundamentals(ticker)
        return compute_graham_signal_series(prices, fundamentals, mos_threshold=mos_threshold)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _close_series(self, symbol: str, period: str) -> Optional[pd.Series]:
        try:
            df = self.data_fetcher.get_historical_prices(symbol, period=period)
            if df is None or df.empty or "Close" not in df.columns:
                return None
            s = df["Close"].copy()
            s.index = pd.to_datetime(s.index).tz_localize(None)
            return s
        except Exception as e:  # pragma: no cover - network/formatting guard
            logger.warning(f"Could not fetch prices for {symbol}: {e}")
            return None

    def _historical_fundamentals(self, ticker: str) -> pd.DataFrame:
        """Best-effort EPS (TTM proxy) and book value per share by report date.

        Uses annual statements from the data source. Returns an empty frame when
        fundamentals are unavailable — the caller then falls back to a
        price-only view.
        """
        try:
            statements = self.data_fetcher.get_financial_statements(ticker)
        except Exception as e:  # pragma: no cover
            logger.warning(f"No statements for {ticker}: {e}")
            return pd.DataFrame()

        income = statements.get("income_statement")
        balance = statements.get("balance_sheet")
        if income is None or balance is None or income.empty or balance.empty:
            return pd.DataFrame()

        eps = self._extract_row(income, ["Diluted EPS", "Basic EPS"])
        net_income = self._extract_row(income, ["Net Income", "Net Income Common Stockholders"])
        equity = self._extract_row(balance, ["Stockholders Equity", "Total Stockholder Equity", "Common Stock Equity"])
        shares = self._extract_row(balance, ["Ordinary Shares Number", "Share Issued", "Common Stock"])

        records = {}
        # EPS: prefer reported EPS, else Net Income / shares.
        if eps is not None:
            records["eps_ttm"] = eps
        elif net_income is not None and shares is not None:
            records["eps_ttm"] = net_income / shares.replace(0, np.nan)

        # Book value per share.
        if equity is not None and shares is not None:
            records["bvps"] = equity / shares.replace(0, np.nan)

        if "eps_ttm" not in records or "bvps" not in records:
            return pd.DataFrame()

        df = pd.DataFrame(records)
        df.index = pd.to_datetime(df.index).tz_localize(None)
        return df.sort_index().dropna(how="all")

    @staticmethod
    def _extract_row(statement: pd.DataFrame, candidate_labels) -> Optional[pd.Series]:
        """Return a statement row (indexed by report date) for the first label
        that exists, or None. Statement columns are report dates."""
        for label in candidate_labels:
            if label in statement.index:
                row = statement.loc[label]
                # Statement columns are dates; make a date-indexed Series.
                return pd.Series(row.values, index=statement.columns, dtype="float64")
        return None
