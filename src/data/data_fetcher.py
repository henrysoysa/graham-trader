"""
Unified data fetcher integrating multiple data sources:
- yfinance (Yahoo Finance)
- fundamentalanalysis
- edgartools (SEC EDGAR)
- openbb
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataFetcher:
    """
    Unified interface for fetching stock data from multiple sources.
    Provides fallback mechanisms and data validation.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize data fetcher with optional API key.

        Args:
            api_key: API key for services that require it (FMP, etc.)
        """
        self.api_key = api_key
        self._initialize_clients()

    def _initialize_clients(self):
        """Initialize API clients for various data sources."""
        try:
            import yfinance as yf
            self.yf = yf
            logger.info("yfinance initialized successfully")
        except ImportError:
            logger.warning("yfinance not available")
            self.yf = None

        try:
            import fundamentalanalysis as fa
            self.fa = fa
            logger.info("fundamentalanalysis initialized successfully")
        except ImportError:
            logger.warning("fundamentalanalysis not available")
            self.fa = None

        try:
            from edgar import Company
            self.edgar = Company
            logger.info("edgartools initialized successfully")
        except ImportError:
            logger.warning("edgartools not available")
            self.edgar = None

        try:
            from openbb import obb
            self.obb = obb
            logger.info("openbb initialized successfully")
        except (ImportError, ValueError) as e:
            logger.warning(f"openbb not available: {e}")
            self.obb = None

    def get_stock_info(self, ticker: str) -> Dict[str, Any]:
        """
        Get basic stock information.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dictionary containing stock information
        """
        if self.yf is None:
            raise ValueError("yfinance is required for this operation")

        try:
            stock = self.yf.Ticker(ticker)
            info = stock.info
            return info
        except Exception as e:
            logger.error(f"Error fetching info for {ticker}: {e}")
            return {}

    def get_financial_statements(self, ticker: str) -> Dict[str, pd.DataFrame]:
        """
        Get financial statements (income statement, balance sheet, cash flow).

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dictionary with 'income_statement', 'balance_sheet', 'cash_flow'
        """
        statements = {}

        # Try yfinance first (free and reliable)
        if self.yf:
            try:
                stock = self.yf.Ticker(ticker)
                statements['income_statement'] = stock.income_stmt
                statements['balance_sheet'] = stock.balance_sheet
                statements['cash_flow'] = stock.cashflow
                logger.info(f"Fetched financial statements for {ticker} from yfinance")
                return statements
            except Exception as e:
                logger.warning(f"yfinance failed for {ticker}: {e}")

        # Fallback to fundamentalanalysis if available
        if self.fa and self.api_key:
            try:
                statements['income_statement'] = self.fa.income_statement(
                    ticker, self.api_key, period='annual'
                )
                statements['balance_sheet'] = self.fa.balance_sheet_statement(
                    ticker, self.api_key, period='annual'
                )
                statements['cash_flow'] = self.fa.cash_flow_statement(
                    ticker, self.api_key, period='annual'
                )
                logger.info(f"Fetched financial statements for {ticker} from fundamentalanalysis")
                return statements
            except Exception as e:
                logger.warning(f"fundamentalanalysis failed for {ticker}: {e}")

        return statements

    def get_key_metrics(self, ticker: str) -> Dict[str, float]:
        """
        Get key financial metrics needed for Graham analysis.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dictionary with key metrics: EPS, book_value, PE, PB, current_ratio, etc.
        """
        metrics = {}

        try:
            # Get basic info from yfinance
            if self.yf:
                stock = self.yf.Ticker(ticker)
                info = stock.info

                # Extract key Graham metrics
                metrics['current_price'] = info.get('currentPrice', info.get('regularMarketPrice', 0))
                metrics['eps'] = info.get('trailingEps', 0)
                metrics['book_value_per_share'] = info.get('bookValue', 0)
                metrics['pe_ratio'] = info.get('trailingPE', 0)
                metrics['pb_ratio'] = info.get('priceToBook', 0)
                metrics['dividend_yield'] = info.get('dividendYield', 0)
                metrics['debt_to_equity'] = info.get('debtToEquity', 0)
                metrics['current_ratio'] = info.get('currentRatio', 0)
                metrics['quick_ratio'] = info.get('quickRatio', 0)
                metrics['return_on_equity'] = info.get('returnOnEquity', 0)
                metrics['revenue_growth'] = info.get('revenueGrowth', 0)
                metrics['earnings_growth'] = info.get('earningsGrowth', 0)
                metrics['market_cap'] = info.get('marketCap', 0)

                # Calculate additional metrics
                statements = self.get_financial_statements(ticker)
                if statements.get('balance_sheet') is not None and not statements['balance_sheet'].empty:
                    balance_sheet = statements['balance_sheet']

                    # Try to calculate current ratio if not available
                    if metrics['current_ratio'] == 0:
                        try:
                            current_assets = balance_sheet.loc['Current Assets'].iloc[0]
                            current_liabilities = balance_sheet.loc['Current Liabilities'].iloc[0]
                            metrics['current_ratio'] = current_assets / current_liabilities if current_liabilities != 0 else 0
                        except:
                            pass

                    # Calculate debt to current assets (Graham metric)
                    try:
                        total_debt = balance_sheet.loc['Total Debt'].iloc[0]
                        current_assets = balance_sheet.loc['Current Assets'].iloc[0]
                        metrics['debt_to_current_assets'] = total_debt / current_assets if current_assets != 0 else 0
                    except:
                        metrics['debt_to_current_assets'] = 0

        except Exception as e:
            logger.error(f"Error fetching metrics for {ticker}: {e}")

        return metrics

    def get_historical_prices(
        self,
        ticker: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        period: str = '5y'
    ) -> pd.DataFrame:
        """
        Get historical price data.

        Args:
            ticker: Stock ticker symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            period: Period to fetch (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)

        Returns:
            DataFrame with OHLCV data
        """
        if self.yf is None:
            raise ValueError("yfinance is required for historical data")

        try:
            stock = self.yf.Ticker(ticker)
            if start_date and end_date:
                df = stock.history(start=start_date, end=end_date)
            else:
                df = stock.history(period=period)

            return df
        except Exception as e:
            logger.error(f"Error fetching historical data for {ticker}: {e}")
            return pd.DataFrame()

    def get_earnings_history(self, ticker: str, years: int = 10) -> pd.DataFrame:
        """
        Get historical earnings data.

        Args:
            ticker: Stock ticker symbol
            years: Number of years to fetch

        Returns:
            DataFrame with historical EPS data
        """
        try:
            if self.yf:
                stock = self.yf.Ticker(ticker)
                financials = stock.financials

                if not financials.empty and 'Net Income' in financials.index:
                    # Get shares outstanding to calculate EPS
                    info = stock.info
                    shares = info.get('sharesOutstanding', 0)

                    net_income = financials.loc['Net Income']
                    eps_history = net_income / shares if shares > 0 else net_income

                    return pd.DataFrame({'EPS': eps_history})

            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error fetching earnings history for {ticker}: {e}")
            return pd.DataFrame()

    def get_dividend_history(self, ticker: str) -> pd.DataFrame:
        """
        Get historical dividend data.

        Args:
            ticker: Stock ticker symbol

        Returns:
            DataFrame with dividend history
        """
        if self.yf is None:
            raise ValueError("yfinance is required for dividend data")

        try:
            stock = self.yf.Ticker(ticker)
            dividends = stock.dividends
            return pd.DataFrame({'Dividend': dividends})
        except Exception as e:
            logger.error(f"Error fetching dividend history for {ticker}: {e}")
            return pd.DataFrame()

    def _index_fetcher_map(self) -> Dict[str, Any]:
        """Map of index key -> curated-list fetcher method (the fallback used
        when a key has no live source, or its live source fails)."""
        return {
            # Developed Markets
            'SP500': self._get_sp500_tickers,
            'NASDAQ100': self._get_nasdaq100_tickers,
            'DOW30': self._get_dow30_tickers,
            'FTSE100': self._get_ftse100_tickers,
            'DAX': self._get_dax_tickers,
            'CAC40': self._get_cac40_tickers,
            'NIKKEI225': self._get_nikkei225_tickers,
            'ASX200': self._get_asx200_tickers,
            'TSX60': self._get_tsx60_tickers,
            # Emerging Markets - Large Cap
            'INDIA_NIFTY50': self._get_india_nifty50_tickers,
            'INDIA_SENSEX': self._get_india_sensex_tickers,
            'BRAZIL_BOVESPA': self._get_brazil_bovespa_tickers,
            'CHINA_HSI': self._get_china_hsi_tickers,
            'CHINA_ADR': self._get_china_adr_tickers,
            'SOUTHAFRICA_TOP40': self._get_southafrica_tickers,
            'MEXICO_IPC': self._get_mexico_ipc_tickers,
            'INDONESIA_IDX': self._get_indonesia_idx_tickers,
            'VIETNAM_VN30': self._get_vietnam_vn30_tickers,
            'EMERGING_ADR': self._get_emerging_market_adrs,
            # Small/Mid Cap Discovery
            'RUSSELL2000': self._get_russell2000_tickers,
            'INDIA_MIDCAP': self._get_india_midcap_tickers,
            'BRAZIL_SMALLCAP': self._get_brazil_smallcap_tickers,
            'CHINA_SMALLCAP': self._get_china_smallcap_adrs,
            'EMERGING_SMALLCAP': self._get_emerging_smallcap_adrs,
            # Broader Developed Markets
            'STOXX600': self._get_stoxx600_tickers,
            'FTSE250': self._get_ftse250_tickers,
            'TOPIX_CORE30': self._get_topix_core30_tickers,
            'KOSPI200': self._get_kospi200_tickers,
            'TAIWAN50': self._get_taiwan50_tickers,
            'STI_SINGAPORE': self._get_sti_singapore_tickers,
            'SMI_SWISS': self._get_smi_swiss_tickers,
            # High-Growth Emerging Economies (per IMF 2026–2030 GDP forecasts)
            'INDIA_NIFTY_NEXT50': self._get_india_nifty_next50_tickers,
            'INDIA_SMALLCAP100': self._get_india_smallcap100_tickers,
            'VIETNAM_VN100': self._get_vietnam_vn100_tickers,
            'PHILIPPINES_PSEI': self._get_philippines_psei_tickers,
            'THAILAND_SET50': self._get_thailand_set50_tickers,
            'MALAYSIA_KLCI': self._get_malaysia_klci_tickers,
            'BANGLADESH_DSE': self._get_bangladesh_dse_tickers,
            'EGYPT_EGX30': self._get_egypt_egx30_tickers,
            'SAUDI_TASI': self._get_saudi_tasi_tickers,
            'UAE_ADX_DFM': self._get_uae_tickers,
            'TURKEY_BIST100': self._get_turkey_bist100_tickers,
            'POLAND_WIG20': self._get_poland_wig20_tickers,
            'PAKISTAN_KSE100': self._get_pakistan_kse100_tickers,
            # High-Growth ADRs (US-listed, easiest to trade)
            'GROWTH_MARKETS_ADR': self._get_growth_markets_adrs,
        }

    def get_index_tickers(self, index_name: str = 'SP500') -> List[str]:
        """
        Get list of tickers for various stock market indexes.

        Args:
            index_name: Name of the index (SP500, NASDAQ100, DOW30, FTSE100, DAX, CAC40, NIKKEI225, ASX200, TSX60)

        Returns:
            List of ticker symbols
        """
        tickers, _source = self.get_index_tickers_with_source(index_name)
        return tickers

    def get_index_tickers_with_source(self, index_name: str = 'SP500') -> "tuple[List[str], str]":
        """Same as ``get_index_tickers``, but also reports where the list came
        from: ``"fresh_cache"``, ``"live"``, ``"stale_cache"``, or
        ``"fallback"`` (the small curated list). Callers that want to warn
        the user about a silent fallback (e.g. the Streamlit UI) should use
        this instead of ``get_index_tickers``.
        """
        index_map = self._index_fetcher_map()

        fetcher = index_map.get(index_name)
        if fetcher is None:
            logger.warning(
                f"Unknown index '{index_name}' requested — no matching universe. "
                f"Falling back to S&P 500 (US). Valid keys: {sorted(index_map)}"
            )
            return self._get_sp500_tickers(), "fallback"

        # Upgrade to live, full constituents where a reliable source is wired up
        # (developed-market indexes); the curated method is the fallback.
        from .constituents import INDEX_SOURCES, get_constituents_with_source
        if index_name in INDEX_SOURCES:
            return get_constituents_with_source(index_name, fallback=fetcher)

        return fetcher(), "live"

    def get_sp500_tickers(self) -> List[str]:
        """Legacy method - calls get_index_tickers('SP500')"""
        return self.get_index_tickers('SP500')

    def _get_sp500_tickers(self) -> List[str]:
        """
        Get the full list of S&P 500 tickers (all ~500 constituents).

        Fetches live constituents from Wikipedia, with a maintained-CSV
        secondary source, and only falls back to a curated ~90-name list if
        both live sources fail.

        Returns:
            List of ticker symbols
        """
        import io

        def _clean(symbols):
            # yfinance uses '-' where tickers carry a class suffix (BRK.B -> BRK-B).
            return [str(s).strip().replace('.', '-') for s in symbols if str(s).strip()]

        # Primary: Wikipedia constituents table. A real browser User-Agent is
        # required or Wikipedia returns a block page. (The previous version
        # passed an invalid storage_options argument to read_html, which raised
        # every time and silently dropped to the ~90-name fallback below.)
        try:
            import requests
            url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
            resp = requests.get(url, headers={'User-Agent': 'Mozilla/5.0 (graham-trader)'}, timeout=15)
            resp.raise_for_status()
            tables = pd.read_html(io.StringIO(resp.text))
            tickers = _clean(tables[0]['Symbol'].tolist())
            if len(tickers) >= 400:
                logger.info(f"Fetched {len(tickers)} S&P 500 tickers from Wikipedia")
                return tickers
            logger.warning(f"Wikipedia returned only {len(tickers)} S&P 500 tickers; trying secondary source")
        except Exception as e:
            logger.warning(f"Wikipedia S&P 500 fetch failed: {e}")

        # Secondary: maintained constituents CSV (datahub).
        try:
            import requests
            csv_url = 'https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv'
            resp = requests.get(csv_url, timeout=15)
            resp.raise_for_status()
            df = pd.read_csv(io.StringIO(resp.text))
            col = 'Symbol' if 'Symbol' in df.columns else df.columns[0]
            tickers = _clean(df[col].tolist())
            if len(tickers) >= 400:
                logger.info(f"Fetched {len(tickers)} S&P 500 tickers from datahub CSV")
                return tickers
        except Exception as e:
            logger.warning(f"datahub S&P 500 fetch failed: {e}")

        # Last resort: curated list of major S&P 500 stocks.
        logger.info("Using fallback list of ~90 major S&P 500 stocks")
        return [
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B',
                'JNJ', 'V', 'WMT', 'JPM', 'MA', 'PG', 'UNH', 'HD', 'CVX', 'XOM',
                'LLY', 'ABBV', 'MRK', 'KO', 'PEP', 'COST', 'AVGO', 'TMO', 'ADBE',
                'MCD', 'CSCO', 'ACN', 'NKE', 'DIS', 'ABT', 'WFC', 'VZ', 'CRM',
                'NFLX', 'CMCSA', 'DHR', 'INTC', 'TXN', 'PM', 'BMY', 'UPS', 'NEE',
                'QCOM', 'HON', 'LOW', 'UNP', 'AMD', 'RTX', 'LMT', 'SPGI', 'INTU',
                'BA', 'CAT', 'GE', 'PLD', 'DE', 'GS', 'AXP', 'MMC', 'BKNG', 'BLK',
                'MDT', 'GILD', 'SYK', 'ADP', 'MDLZ', 'CVS', 'CI', 'ZTS', 'SBUX',
                'AMT', 'VRTX', 'ADI', 'ISRG', 'TJX', 'PFE', 'REGN', 'CB', 'SO',
                'MO', 'DUK', 'BSX', 'SLB', 'SCHW', 'CME', 'EOG', 'PGR', 'ITW'
            ]

    def _get_nasdaq100_tickers(self) -> List[str]:
        """Get NASDAQ-100 tickers."""
        logger.info("Using curated list of NASDAQ-100 stocks")
        return [
            'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'NVDA', 'META', 'TSLA', 'AVGO',
            'COST', 'NFLX', 'AMD', 'PEP', 'ADBE', 'CSCO', 'TMUS', 'CMCSA', 'INTC',
            'QCOM', 'TXN', 'AMGN', 'HON', 'INTU', 'AMAT', 'ISRG', 'BKNG', 'ADP',
            'SBUX', 'GILD', 'ADI', 'VRTX', 'REGN', 'MU', 'LRCX', 'PANW', 'PYPL',
            'KLAC', 'SNPS', 'CDNS', 'MRVL', 'ASML', 'NXPI', 'MELI', 'CTAS', 'ORLY',
            'ABNB', 'CRWD', 'FTNT', 'DASH', 'PCAR', 'CHTR', 'MNST', 'PAYX', 'ODFL',
            'CPRT', 'ROST', 'FAST', 'KDP', 'EA', 'CTSH', 'VRSK', 'BKR', 'DXCM',
            'TEAM', 'IDXX', 'CSGP', 'GEHC', 'CCEP', 'ZS', 'ANSS', 'DDOG', 'BIIB'
        ]

    def _get_dow30_tickers(self) -> List[str]:
        """Get Dow Jones Industrial Average (30) tickers."""
        logger.info("Using Dow 30 stocks")
        return [
            'AAPL', 'MSFT', 'UNH', 'GS', 'HD', 'MCD', 'CAT', 'V', 'AMGN', 'BA',
            'TRV', 'AXP', 'HON', 'JPM', 'IBM', 'CRM', 'JNJ', 'PG', 'CVX', 'MRK',
            'WMT', 'DIS', 'NKE', 'MMM', 'KO', 'DOW', 'VZ', 'CSCO', 'INTC', 'WBA'
        ]

    def _get_ftse100_tickers(self) -> List[str]:
        """Get FTSE 100 (UK) tickers with .L suffix."""
        logger.info("Using curated list of FTSE 100 stocks")
        return [
            'SHEL.L', 'AZN.L', 'HSBA.L', 'ULVR.L', 'DGE.L', 'BP.L', 'GSK.L',
            'RIO.L', 'NG.L', 'REL.L', 'LSEG.L', 'VOD.L', 'BARC.L', 'LLOY.L',
            'AAL.L', 'TSCO.L', 'PRU.L', 'IMB.L', 'CPG.L', 'BT-A.L', 'EXPN.L',
            'AV.L', 'BDEV.L', 'CRH.L', 'FERG.L', 'RKT.L', 'ANTO.L', 'GLEN.L',
            'SMDS.L', 'WPP.L', 'OCDO.L', 'BRBY.L', 'RR.L', 'LAND.L', 'SGE.L'
        ]

    def _get_dax_tickers(self) -> List[str]:
        """Get DAX (Germany) tickers - using US-listed ADRs where available."""
        logger.info("Using DAX stocks (US-listed ADRs where available)")
        return [
            'SAP', 'ADDYY', 'SIEGY', 'BAMXF', 'ALIZF', 'DBOEY', 'MBGAF', 'DTEGF',
            'BAYZF', 'BASFY', 'VLKAF', 'HENOY', 'MKKGY', 'BMWYY', 'IDSMY', 'RHHBY',
            'DPSGY', 'HEINY', 'FNTN', 'DDAIF', 'SMNEY', 'RWEOY', 'EOAN', 'SAPGF'
        ]

    def _get_cac40_tickers(self) -> List[str]:
        """Get CAC 40 (France) tickers - using US-listed ADRs where available."""
        logger.info("Using CAC 40 stocks (US-listed ADRs where available)")
        return [
            'AXAHY', 'BNPQY', 'TOTF', 'LVMUY', 'DANOY', 'SAUHF', 'LRLCY', 'OREP',
            'AIRYY', 'VLVLY', 'SASY', 'SCGLY', 'CAGNY', 'UNLVF', 'CRARY', 'REXR',
            'VIVHY', 'BOUYF', 'SGOBF', 'PRTP', 'EGRNF', 'GECFF', 'MC.PA', 'OR.PA'
        ]

    def _get_nikkei225_tickers(self) -> List[str]:
        """Get Nikkei 225 (Japan) tickers - using US-listed ADRs."""
        logger.info("Using Nikkei 225 stocks (US-listed ADRs)")
        return [
            'TM', 'SONY', 'SMFG', 'MFG', 'NMR', 'HTHIY', 'HMC', 'FUJIY', 'MSBHF',
            'SNEJF', 'NSANY', 'TKOMY', 'DNZOY', 'CAJFF', 'MITSY', 'FANUY', 'KDDIY',
            'SOMLY', 'SFNCY', 'AIQUY', 'NTDOY', 'SFPHY', 'KBHYF', 'SMFNF', 'SHKNY'
        ]

    def _get_asx200_tickers(self) -> List[str]:
        """Get ASX 200 (Australia) tickers with .AX suffix."""
        logger.info("Using curated list of ASX 200 stocks")
        return [
            'BHP.AX', 'CBA.AX', 'CSL.AX', 'NAB.AX', 'WBC.AX', 'ANZ.AX', 'WES.AX',
            'MQG.AX', 'FMG.AX', 'WDS.AX', 'TLS.AX', 'RIO.AX', 'GMG.AX', 'TCL.AX',
            'WOW.AX', 'STO.AX', 'REA.AX', 'WTC.AX', 'COL.AX', 'QBE.AX', 'RMD.AX',
            'NCM.AX', 'ALL.AX', 'IAG.AX', 'SCG.AX', 'SHL.AX', 'CPU.AX', 'APA.AX'
        ]

    def _get_tsx60_tickers(self) -> List[str]:
        """Get TSX 60 (Canada) tickers with .TO suffix."""
        logger.info("Using TSX 60 stocks")
        return [
            'RY.TO', 'TD.TO', 'SHOP.TO', 'BNS.TO', 'ENB.TO', 'BMO.TO', 'CNR.TO',
            'CNQ.TO', 'CP.TO', 'TRI.TO', 'WCN.TO', 'SU.TO', 'MFC.TO', 'ABX.TO',
            'CM.TO', 'BCE.TO', 'TRP.TO', 'CVE.TO', 'ATD.TO', 'FNV.TO', 'NTR.TO',
            'QSR.TO', 'SLF.TO', 'IMO.TO', 'POW.TO', 'BAM.TO', 'GIB-A.TO', 'CCL-B.TO',
            'MG.TO', 'WPM.TO', 'FM.TO', 'L.TO', 'DOL.TO', 'EMA.TO', 'FTS.TO'
        ]

    # ========================== EMERGING MARKETS ==========================

    def _get_india_nifty50_tickers(self) -> List[str]:
        """Get India NIFTY 50 tickers with .NS suffix (NSE exchange)."""
        logger.info("Using India NIFTY 50 stocks (.NS suffix for NSE)")
        return [
            'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS',
            'HINDUNILVR.NS', 'ITC.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'BAJFINANCE.NS',
            'KOTAKBANK.NS', 'LT.NS', 'HCLTECH.NS', 'AXISBANK.NS', 'ASIANPAINT.NS',
            'MARUTI.NS', 'SUNPHARMA.NS', 'TITAN.NS', 'ULTRACEMCO.NS', 'NESTLEIND.NS',
            'WIPRO.NS', 'BAJAJFINSV.NS', 'ADANIPORTS.NS', 'ONGC.NS', 'NTPC.NS',
            'POWERGRID.NS', 'M&M.NS', 'TATAMOTORS.NS', 'TATASTEEL.NS', 'TECHM.NS',
            'INDUSINDBK.NS', 'DRREDDY.NS', 'JSWSTEEL.NS', 'GRASIM.NS', 'CIPLA.NS',
            'DIVISLAB.NS', 'HEROMOTOCO.NS', 'EICHERMOT.NS', 'BRITANNIA.NS', 'BPCL.NS',
            'COALINDIA.NS', 'SHREECEM.NS', 'HINDALCO.NS', 'APOLLOHOSP.NS', 'ADANIENT.NS',
            'BAJAJ-AUTO.NS', 'TATACONSUM.NS', 'SBILIFE.NS', 'HDFCLIFE.NS', 'UPL.NS'
        ]

    def _get_india_sensex_tickers(self) -> List[str]:
        """Get India BSE SENSEX 30 tickers with .BO suffix (BSE exchange)."""
        logger.info("Using India SENSEX 30 stocks (.BO suffix for BSE)")
        return [
            'RELIANCE.BO', 'TCS.BO', 'HDFCBANK.BO', 'INFY.BO', 'ICICIBANK.BO',
            'HINDUNILVR.BO', 'ITC.BO', 'SBIN.BO', 'BHARTIARTL.BO', 'BAJFINANCE.BO',
            'KOTAKBANK.BO', 'LT.BO', 'HCLTECH.BO', 'AXISBANK.BO', 'ASIANPAINT.BO',
            'MARUTI.BO', 'SUNPHARMA.BO', 'TITAN.BO', 'ULTRACEMCO.BO', 'NESTLEIND.BO',
            'WIPRO.BO', 'M&M.BO', 'TATAMOTORS.BO', 'TATASTEEL.BO', 'TECHM.BO',
            'INDUSINDBK.BO', 'NTPC.BO', 'POWERGRID.BO', 'BAJAJFINSV.BO', 'ONGC.BO'
        ]

    def _get_brazil_bovespa_tickers(self) -> List[str]:
        """Get Brazil BOVESPA top stocks with .SA suffix (B3 exchange)."""
        logger.info("Using Brazil BOVESPA stocks (.SA suffix for B3)")
        return [
            'PETR4.SA', 'VALE3.SA', 'ITUB4.SA', 'BBDC4.SA', 'ABEV3.SA',
            'BBAS3.SA', 'WEGE3.SA', 'RENT3.SA', 'B3SA3.SA', 'SUZB3.SA',
            'RDOR3.SA', 'RAIL3.SA', 'JBSS3.SA', 'MGLU3.SA', 'LREN3.SA',
            'GGBR4.SA', 'KLBN11.SA', 'EMBR3.SA', 'ELET3.SA', 'SANB11.SA',
            'CSNA3.SA', 'VIVT3.SA', 'CPLE6.SA', 'ENBR3.SA', 'EQTL3.SA',
            'HAPV3.SA', 'TOTS3.SA', 'PCAR3.SA', 'RADL3.SA', 'CCRO3.SA'
        ]

    def _get_china_hsi_tickers(self) -> List[str]:
        """Get China/Hong Kong Hang Seng Index stocks with .HK suffix."""
        logger.info("Using Hong Kong Hang Seng stocks (.HK suffix)")
        return [
            '0700.HK', '9988.HK', '0939.HK', '1299.HK', '0941.HK',  # Tech giants
            '2318.HK', '0388.HK', '3988.HK', '1398.HK', '0005.HK',  # Banks
            '0001.HK', '0002.HK', '0003.HK', '0004.HK', '0006.HK',  # Utilities
            '0016.HK', '0017.HK', '0011.HK', '0012.HK', '0013.HK',  # Property
            '0883.HK', '0386.HK', '0688.HK', '0857.HK', '2382.HK',  # Energy
            '1109.HK', '1113.HK', '1177.HK', '2007.HK', '2313.HK'   # Industrials
        ]

    def _get_china_adr_tickers(self) -> List[str]:
        """Get US-listed Chinese ADRs (most accessible for US investors)."""
        logger.info("Using US-listed Chinese ADRs")
        return [
            'BABA', 'JD', 'PDD', 'BIDU', 'NIO', 'XPEV', 'LI', 'NTES',
            'TME', 'BILI', 'IQ', 'VIPS', 'DIDI', 'EDU', 'TAL', 'YMM',
            'YUMC', 'ATHM', 'TIGR', 'MNSO', 'BZ', 'LX', 'TUYA', 'DQ',
            'WB', 'BZUN', 'HUYA', 'DOYU', 'KC', 'QTT', 'MOMO', 'YY',
            'RERE', 'SOHU', 'FENG', 'CAAS', 'BEST', 'EH', 'GOTU', 'RLX'
        ]

    def _get_southafrica_tickers(self) -> List[str]:
        """Get South Africa JSE Top 40 stocks with .JO suffix."""
        logger.info("Using South Africa JSE Top 40 stocks (.JO suffix)")
        return [
            'AGL.JO', 'ANG.JO', 'APN.JO', 'BHP.JO', 'BTI.JO', 'BVT.JO',
            'CFR.JO', 'CPI.JO', 'DSY.JO', 'FSR.JO', 'GFI.JO', 'GLN.JO',
            'GRT.JO', 'IMP.JO', 'INL.JO', 'INP.JO', 'MCG.JO', 'MEI.JO',
            'MNP.JO', 'MRP.JO', 'MTN.JO', 'NED.JO', 'NPN.JO', 'OMU.JO',
            'PRX.JO', 'REM.JO', 'RNI.JO', 'SBK.JO', 'SHP.JO', 'SLM.JO',
            'SNH.JO', 'SOL.JO', 'SPP.JO', 'SSW.JO', 'TBS.JO', 'VOD.JO'
        ]

    def _get_mexico_ipc_tickers(self) -> List[str]:
        """Get Mexico IPC top stocks with .MX suffix."""
        logger.info("Using Mexico IPC stocks (.MX suffix)")
        return [
            'WALMEX.MX', 'AMXL.MX', 'FEMSAUBD.MX', 'GFNORTEO.MX', 'CEMEXCPO.MX',
            'GMEXICOB.MX', 'ALFAA.MX', 'BIMBOA.MX', 'TLEVISACPO.MX', 'KIMBERA.MX',
            'LABB.MX', 'GAPB.MX', 'GRUMAB.MX', 'ELEKTRA.MX', 'PINFRA.MX',
            'ASURB.MX', 'OMAB.MX', 'GCARSOA1.MX', 'LIVEPOLC-1.MX', 'MEXCHEM.MX'
        ]

    def _get_indonesia_idx_tickers(self) -> List[str]:
        """Get Indonesia IDX top stocks with .JK suffix."""
        logger.info("Using Indonesia IDX stocks (.JK suffix)")
        return [
            'BBCA.JK', 'BBRI.JK', 'BMRI.JK', 'BBNI.JK', 'ASII.JK',
            'TLKM.JK', 'UNVR.JK', 'ICBP.JK', 'INDF.JK', 'KLBF.JK',
            'HMSP.JK', 'SMGR.JK', 'INCO.JK', 'PTBA.JK', 'GGRM.JK',
            'ADRO.JK', 'EXCL.JK', 'CPIN.JK', 'PGAS.JK', 'JSMR.JK'
        ]

    def _get_vietnam_vn30_tickers(self) -> List[str]:
        """Get Vietnam VN30 top stocks (Note: Limited yfinance support)."""
        logger.info("Using Vietnam VN30 stocks (Note: Limited data availability)")
        return [
            'VNM', 'VIC', 'VHM', 'GAS', 'TCB', 'BID', 'CTG', 'VCB',
            'HPG', 'MSN', 'VPB', 'PLX', 'POW', 'SAB', 'MWG', 'VRE',
            'FPT', 'GVR', 'VJC', 'HVN', 'STB', 'BVH', 'SSI', 'PDR'
        ]

    def _get_emerging_market_adrs(self) -> List[str]:
        """
        Get curated list of emerging market ADRs across multiple countries.
        These are US-listed stocks, making them most accessible for screening.
        """
        logger.info("Using curated emerging market ADRs (most accessible)")
        return [
            # China
            'BABA', 'JD', 'PDD', 'BIDU', 'NIO', 'XPEV', 'LI', 'NTES', 'TME', 'BILI',
            # India
            'INFY', 'WIT', 'HDB', 'IBN', 'SIFY', 'RDY', 'TTM', 'VEDL',
            # Brazil
            'VALE', 'PBR', 'ITUB', 'BBD', 'ABEV', 'SBS', 'ERJ', 'GGB',
            # Taiwan
            'TSM', 'UMC', 'ASX',
            # South Korea
            'KB', 'PKX', 'LPL', 'SKM',
            # South Africa
            'GOLD', 'NMR', 'SBSW',
            # Mexico
            'AMX', 'TV', 'CX',
            # Argentina
            'YPF', 'TEO', 'BMA', 'GGAL', 'PAM',
            # Chile
            'SQM', 'ENIA', 'LTM',
            # Colombia
            'EC',
            # Various emerging markets
            'VALE', 'RIO', 'BHP'  # Diversified miners with EM exposure
        ]

    # ========================== SMALL/MID CAP DISCOVERY ==========================

    def _get_russell2000_tickers(self) -> List[str]:
        """Get Russell 2000 small-cap US stocks sample."""
        logger.info("Using Russell 2000 sample (US small caps)")
        return [
            # Financial Services
            'BANR', 'CADE', 'CATY', 'CBSH', 'EWBC', 'FULT', 'GBCI', 'HOMB', 'IBOC',
            'IBTX', 'ONB', 'OZK', 'PBCT', 'PFS', 'SNV', 'TRMK', 'UBSI', 'UCBI',
            # Healthcare
            'AXSM', 'BLFS', 'CORT', 'CRVL', 'DNLI', 'DXCM', 'ENSG', 'FATE', 'FGEN',
            'HALO', 'ICUI', 'KRYS', 'LGND', 'MRNA', 'NVTA', 'NVCR', 'PDCO', 'PTCT',
            # Technology
            'ACIA', 'APPN', 'AVAV', 'CALX', 'CGNX', 'COUP', 'CVLT', 'DOMO', 'ETWO',
            'EXLS', 'GLUU', 'GTLS', 'LITE', 'MSTR', 'NEWR', 'QLYS', 'RSKD', 'SAIL',
            # Industrials
            'ATRO', 'BECN', 'BWXT', 'CSWI', 'ESE', 'GVA', 'HLIO', 'LECO', 'NOVT',
            'NWL', 'POWI', 'RUSHA', 'SSD', 'TTEK', 'TRS', 'VICR', 'WTS', 'ZIXI',
            # Consumer
            'AEO', 'BOOT', 'CAKE', 'CHUY', 'CROX', 'DDS', 'FIVE', 'GRUB', 'KIRK',
            'LULU', 'NAPA', 'OLLI', 'PLNT', 'PRDO', 'RVLV', 'TXRH', 'ULTA', 'WOLF'
        ]

    def _get_india_midcap_tickers(self) -> List[str]:
        """Get India mid-cap stocks (next tier after NIFTY 50)."""
        logger.info("Using India mid-cap stocks (.NS suffix)")
        return [
            # IT & Services
            'MPHASIS.NS', 'COFORGE.NS', 'PERSISTENT.NS', 'LTTS.NS', 'TATAELXSI.NS',
            # Finance
            'BANDHANBNK.NS', 'IDFCFIRSTB.NS', 'FEDERALBNK.NS', 'CHOLAFIN.NS', 'MUTHOOTFIN.NS',
            # Pharma
            'BIOCON.NS', 'TORNTPHARM.NS', 'LUPIN.NS', 'ALKEM.NS', 'AUROPHARMA.NS',
            # Consumer
            'TATACONSUM.NS', 'MARICO.NS', 'DABUR.NS', 'GODREJCP.NS', 'COLGATE.NS',
            # Infrastructure
            'ADANIGREEN.NS', 'ADANIPOWER.NS', 'TATAPOWER.NS', 'TORNTPOWER.NS',
            # Auto Components
            'BALKRISIND.NS', 'MRF.NS', 'MOTHERSON.NS', 'BOSCHLTD.NS',
            # Others
            'PIDILITIND.NS', 'BERGEPAINT.NS', 'PAGEIND.NS', 'ABFRL.NS'
        ]

    def _get_brazil_smallcap_tickers(self) -> List[str]:
        """Get Brazil small/mid-cap stocks beyond BOVESPA top stocks."""
        logger.info("Using Brazil small/mid-cap stocks (.SA suffix)")
        return [
            # Retail & Consumer
            'LAME4.SA', 'GRND3.SA', 'SOMA3.SA', 'VIVA3.SA', 'CEAB3.SA',
            # Financial
            'PINE4.SA', 'BPAN4.SA', 'BMOB3.SA', 'BPAC11.SA',
            # Healthcare
            'FLRY3.SA', 'HAPV3.SA', 'GNDI3.SA', 'QUAL3.SA',
            # Technology
            'TOTS3.SA', 'LWSA3.SA', 'MDIA3.SA',
            # Infrastructure
            'TAEE11.SA', 'TRPL4.SA', 'CMIG4.SA', 'SAPR11.SA',
            # Industrials
            'AZUL4.SA', 'GOLL4.SA', 'CSAN3.SA', 'POMO4.SA',
            # Real Estate
            'MULT3.SA', 'MRVE3.SA', 'CYRE3.SA', 'JHSF3.SA'
        ]

    def _get_china_smallcap_adrs(self) -> List[str]:
        """Get smaller Chinese ADRs - less known opportunities."""
        logger.info("Using smaller Chinese ADRs (hidden gems)")
        return [
            # E-commerce & Retail
            'VIPS', 'BZUN', 'MOGU', 'TOUR',
            # Education
            'EDU', 'TAL', 'GOTU', 'LAIX',
            # Entertainment & Media
            'HUYA', 'DOYU', 'YY', 'MOMO', 'FENG',
            # Technology
            'TIGR', 'FUTU', 'QTT', 'TUYA', 'KC',
            # Healthcare & Biotech
            'ZLAB', 'CBPO', 'APT', 'CASI',
            # Finance
            'YINN', 'PHUN', 'LX', 'BEKE',
            # Others
            'ATAI', 'CAAS', 'BEST', 'EH', 'RERE', 'SOHU'
        ]

    def _get_emerging_smallcap_adrs(self) -> List[str]:
        """
        Curated list of smaller emerging market ADRs - true "diamonds in the rough".
        These are less covered, smaller market cap, but US-listed for accessibility.
        """
        logger.info("Using emerging markets small-cap ADRs (hidden gems)")
        return [
            # India - Smaller IT/Services
            'SIFY', 'INFY', 'WIT', 'VEDL', 'TTM', 'RDY',
            # China - Smaller Tech/Consumer
            'TIGR', 'FUTU', 'QTT', 'MOGU', 'BZUN', 'VIPS', 'HUYA', 'YY',
            # Brazil - Smaller Players
            'ERJ', 'GGB', 'SBS', 'CBD', 'CIG',
            # South Korea - Smaller Caps
            'LPL', 'PKX', 'HIMX',
            # Taiwan - Beyond TSM
            'UMC', 'ASX', 'SPIL',
            # Southeast Asia
            'GRAB', 'SEA',
            # Latin America
            'GLOB', 'SAN', 'SUP',
            # Israel (Emerging Tech Hub)
            'TEVA', 'NICE', 'CHKP', 'CYBR', 'INMD',
            # South Africa - Miners
            'SBSW', 'HMY', 'AU', 'AUY',
            # Various
            'FRO', 'STNG', 'DHT', 'NAT'  # Shipping (global trade exposure)
        ]

    # ========================== BROADER DEVELOPED MARKETS ==========================
    # These extend beyond blue-chip indices to give the Graham screener a much
    # deeper universe. yfinance ticker suffixes: .L (LSE), .T (TSE), .KS (KRX),
    # .TW (TWSE), .SI (SGX), .SW (SIX).

    def _get_stoxx600_tickers(self) -> List[str]:
        """STOXX Europe 600 — pan-European large/mid cap sample.

        Note: full 600-name list changes frequently; this is a curated
        representative subset spanning UK, Germany, France, Switzerland,
        Netherlands, Spain, Italy, Nordics.
        """
        logger.info("Using STOXX Europe 600 curated sample")
        return [
            # UK (.L)
            'SHEL.L', 'AZN.L', 'HSBA.L', 'ULVR.L', 'DGE.L', 'GSK.L', 'RIO.L',
            'BATS.L', 'REL.L', 'BP.L', 'LSEG.L', 'CPG.L', 'NG.L',
            # Germany (.DE) — SAP & Siemens use ADRs already; here native
            'SAP.DE', 'SIE.DE', 'ALV.DE', 'DTE.DE', 'MBG.DE', 'BAS.DE',
            'BAYN.DE', 'MUV2.DE', 'DBK.DE', 'ADS.DE', 'BMW.DE',
            # France (.PA)
            'MC.PA', 'OR.PA', 'AIR.PA', 'SAN.PA', 'BNP.PA', 'CS.PA',
            'RMS.PA', 'SU.PA', 'AI.PA', 'DG.PA', 'KER.PA',
            # Switzerland (.SW)
            'NESN.SW', 'ROG.SW', 'NOVN.SW', 'UBSG.SW', 'ZURN.SW', 'ABBN.SW',
            'CFR.SW', 'GIVN.SW', 'HOLN.SW',
            # Netherlands (.AS)
            'ASML.AS', 'HEIA.AS', 'INGA.AS', 'PHIA.AS', 'AD.AS', 'REN.AS',
            # Spain (.MC)
            'ITX.MC', 'SAN.MC', 'BBVA.MC', 'IBE.MC', 'REP.MC', 'TEF.MC',
            # Italy (.MI)
            'ENI.MI', 'ISP.MI', 'UCG.MI', 'STLAM.MI', 'ENEL.MI', 'G.MI',
            # Nordics
            'NOVO-B.CO', 'VOLV-B.ST', 'ATCO-A.ST', 'HM-B.ST', 'ERIC-B.ST',
            'EQNR.OL', 'NDA-FI.HE', 'NOKIA.HE'
        ]

    def _get_ftse250_tickers(self) -> List[str]:
        """FTSE 250 UK mid-cap — sample of well-covered names."""
        logger.info("Using FTSE 250 UK mid-cap sample")
        return [
            'MKS.L', 'ITV.L', 'BAB.L', 'JD.L', 'HSX.L', 'GAW.L', 'HWDN.L',
            'MRO.L', 'BME.L', 'BYG.L', 'GNS.L', 'IWG.L', 'PSN.L', 'RMV.L',
            'SPT.L', 'TRN.L', 'WMH.L', 'WIZZ.L', 'DPLM.L', 'GNC.L',
            'AGR.L', 'HL.L', 'IHG.L', 'JMAT.L', 'MONY.L', 'PAG.L', 'PETS.L',
            'PLUS.L', 'PNN.L', 'RSW.L', 'SGRO.L', 'SVS.L', 'TATE.L'
        ]

    def _get_topix_core30_tickers(self) -> List[str]:
        """TOPIX Core 30 — Japan's 30 most liquid names on TSE (.T suffix)."""
        logger.info("Using TOPIX Core 30 (Japan large caps, .T suffix)")
        return [
            '7203.T', '6758.T', '9432.T', '8306.T', '9984.T', '6098.T',
            '6861.T', '7974.T', '8035.T', '9433.T', '6902.T', '4063.T',
            '8316.T', '8058.T', '8031.T', '8001.T', '6501.T', '7267.T',
            '4502.T', '4568.T', '4519.T', '4661.T', '6367.T', '6594.T',
            '6981.T', '7751.T', '9022.T', '9020.T', '9613.T', '9503.T'
        ]

    def _get_kospi200_tickers(self) -> List[str]:
        """KOSPI 200 — South Korea top 200 by liquidity (.KS suffix)."""
        logger.info("Using KOSPI 200 sample (South Korea, .KS suffix)")
        return [
            '005930.KS', '000660.KS', '373220.KS', '207940.KS', '005380.KS',
            '005490.KS', '051910.KS', '006400.KS', '035420.KS', '028260.KS',
            '000270.KS', '068270.KS', '035720.KS', '105560.KS', '055550.KS',
            '012330.KS', '096770.KS', '017670.KS', '032830.KS', '015760.KS',
            '003550.KS', '033780.KS', '018260.KS', '316140.KS', '011200.KS',
            '086790.KS', '009150.KS', '090430.KS', '024110.KS', '036570.KS'
        ]

    def _get_taiwan50_tickers(self) -> List[str]:
        """Taiwan 50 — top 50 TWSE names (.TW suffix)."""
        logger.info("Using Taiwan 50 sample (.TW suffix)")
        return [
            '2330.TW', '2317.TW', '2454.TW', '2308.TW', '2412.TW', '2891.TW',
            '2882.TW', '3711.TW', '2382.TW', '1301.TW', '1303.TW', '2881.TW',
            '2886.TW', '2884.TW', '2885.TW', '1216.TW', '2002.TW', '2207.TW',
            '2303.TW', '2357.TW', '3008.TW', '2379.TW', '2409.TW', '3045.TW',
            '2892.TW', '5871.TW', '2880.TW', '2887.TW', '1101.TW', '2105.TW'
        ]

    def _get_sti_singapore_tickers(self) -> List[str]:
        """Straits Times Index — Singapore's 30 largest (.SI suffix)."""
        logger.info("Using Straits Times Index (Singapore, .SI suffix)")
        return [
            'D05.SI', 'O39.SI', 'U11.SI', 'Z74.SI', 'C6L.SI', 'F34.SI',
            'S68.SI', 'C31.SI', 'C09.SI', 'BN4.SI', 'G13.SI', 'V03.SI',
            'S63.SI', 'U96.SI', 'S58.SI', 'BS6.SI', 'H78.SI', 'C38U.SI',
            'A17U.SI', 'ME8U.SI', 'M44U.SI', 'AJBU.SI', 'CJLU.SI', 'J36.SI',
            'J37.SI', 'Y92.SI', 'U14.SI', 'C07.SI', 'N2IU.SI', 'C52.SI'
        ]

    def _get_smi_swiss_tickers(self) -> List[str]:
        """Swiss Market Index — Switzerland's 20 largest (.SW suffix)."""
        logger.info("Using SMI (Swiss Market Index, .SW suffix)")
        return [
            'NESN.SW', 'ROG.SW', 'NOVN.SW', 'UBSG.SW', 'ZURN.SW', 'ABBN.SW',
            'CFR.SW', 'GIVN.SW', 'HOLN.SW', 'SIKA.SW', 'LONN.SW', 'SREN.SW',
            'ALC.SW', 'GEBN.SW', 'PGHN.SW', 'SCMN.SW', 'SLHN.SW', 'LOGN.SW',
            'KNIN.SW', 'SGSN.SW'
        ]

    # ==================== HIGH-GROWTH EMERGING ECONOMIES ====================
    # Focus: countries projected by IMF (2026–2030) to see above-average GDP
    # growth — India, Vietnam, Philippines, Bangladesh, Indonesia, Egypt,
    # Saudi Arabia (Vision 2030), UAE, Turkey, Poland, Malaysia, Pakistan.

    def _get_india_nifty_next50_tickers(self) -> List[str]:
        """NIFTY Next 50 — the tier below NIFTY 50 (.NS suffix).

        This is where India's next generation of blue chips is emerging;
        arguably the highest-signal growth-value universe on the planet.
        """
        logger.info("Using India NIFTY Next 50 (.NS suffix)")
        return [
            'ADANIENSOL.NS', 'ADANIGREEN.NS', 'ADANIPOWER.NS', 'AMBUJACEM.NS',
            'ATGL.NS', 'BAJAJHLDNG.NS', 'BANKBARODA.NS', 'BERGEPAINT.NS',
            'BOSCHLTD.NS', 'CANBK.NS', 'CGPOWER.NS', 'CHOLAFIN.NS', 'COLPAL.NS',
            'DABUR.NS', 'DLF.NS', 'DMART.NS', 'GAIL.NS', 'GODREJCP.NS',
            'HAL.NS', 'HAVELLS.NS', 'ICICIGI.NS', 'ICICIPRULI.NS', 'INDIGO.NS',
            'IOC.NS', 'IRCTC.NS', 'JINDALSTEL.NS', 'JIOFIN.NS', 'LICI.NS',
            'MARICO.NS', 'MOTHERSON.NS', 'NAUKRI.NS', 'PFC.NS', 'PIDILITIND.NS',
            'PNB.NS', 'RECLTD.NS', 'SAIL.NS', 'SIEMENS.NS', 'SRF.NS',
            'SUNTV.NS', 'TATAPOWER.NS', 'TORNTPHARM.NS', 'TRENT.NS',
            'TVSMOTOR.NS', 'UNITDSPR.NS', 'VBL.NS', 'VEDL.NS', 'ZOMATO.NS',
            'ZYDUSLIFE.NS'
        ]

    def _get_india_smallcap100_tickers(self) -> List[str]:
        """NIFTY Smallcap 100 sample — India small caps (.NS suffix).

        Where classic Graham candidates hide in a fast-growing economy.
        """
        logger.info("Using India Smallcap 100 sample (.NS suffix)")
        return [
            'AAVAS.NS', 'AEGISCHEM.NS', 'AFFLE.NS', 'AJANTPHARM.NS',
            'AMBER.NS', 'ANURAS.NS', 'APLLTD.NS', 'ASTERDM.NS', 'BALRAMCHIN.NS',
            'BEML.NS', 'BIRLACORPN.NS', 'BLUESTARCO.NS', 'BSOFT.NS', 'CAMS.NS',
            'CANFINHOME.NS', 'CARBORUNIV.NS', 'CDSL.NS', 'CENTURYTEX.NS',
            'CERA.NS', 'CHAMBLFERT.NS', 'CHENNPETRO.NS', 'CROMPTON.NS',
            'CUB.NS', 'CYIENT.NS', 'DEEPAKFERT.NS', 'DEVYANI.NS', 'ECLERX.NS',
            'ENGINERSIN.NS', 'EQUITASBNK.NS', 'FINCABLES.NS', 'FINPIPE.NS',
            'FSL.NS', 'GESHIP.NS', 'GLENMARK.NS', 'GRAPHITE.NS', 'GUJGASLTD.NS',
            'HFCL.NS', 'HINDCOPPER.NS', 'IDBI.NS', 'IIFL.NS', 'INDIAMART.NS',
            'IRB.NS', 'JBCHEPHARM.NS', 'JKCEMENT.NS', 'JYOTHYLAB.NS',
            'KEC.NS', 'KEI.NS', 'KIRLOSENG.NS', 'LTF.NS', 'MASTEK.NS'
        ]

    def _get_vietnam_vn100_tickers(self) -> List[str]:
        """Vietnam VN100 — broader than VN30.

        Note: yfinance coverage for direct HOSE tickers is patchy; symbols
        listed here are the ones with the best availability. For reliable
        Vietnam exposure, consider VNM (US-listed Vietnam ETF ADR).
        """
        logger.info("Using Vietnam VN100 sample (data availability varies)")
        return [
            'VNM.VN', 'VIC.VN', 'VHM.VN', 'GAS.VN', 'TCB.VN', 'BID.VN',
            'CTG.VN', 'VCB.VN', 'HPG.VN', 'MSN.VN', 'VPB.VN', 'PLX.VN',
            'POW.VN', 'SAB.VN', 'MWG.VN', 'VRE.VN', 'FPT.VN', 'GVR.VN',
            'VJC.VN', 'HVN.VN', 'STB.VN', 'BVH.VN', 'SSI.VN', 'PDR.VN',
            'HDB.VN', 'ACB.VN', 'MBB.VN', 'TPB.VN', 'VIB.VN', 'SHB.VN',
            'DGC.VN', 'DPM.VN', 'PVD.VN', 'PVS.VN', 'BSR.VN'
        ]

    def _get_philippines_psei_tickers(self) -> List[str]:
        """Philippine Stock Exchange PSEi — top 30 (.PS suffix)."""
        logger.info("Using Philippines PSEi 30 (.PS suffix)")
        return [
            'SM.PS', 'BDO.PS', 'ALI.PS', 'BPI.PS', 'AC.PS', 'JGS.PS',
            'AEV.PS', 'JFC.PS', 'MBT.PS', 'SMPH.PS', 'ICT.PS', 'URC.PS',
            'TEL.PS', 'GLO.PS', 'MER.PS', 'MPI.PS', 'RLC.PS', 'GTCAP.PS',
            'AGI.PS', 'DMC.PS', 'SCC.PS', 'FGEN.PS', 'PGOLD.PS', 'MEG.PS',
            'CNVRG.PS', 'CNPF.PS', 'MONDE.PS', 'BLOOM.PS', 'AP.PS', 'WLCON.PS'
        ]

    def _get_thailand_set50_tickers(self) -> List[str]:
        """Thailand SET50 — top 50 on Stock Exchange of Thailand (.BK suffix)."""
        logger.info("Using Thailand SET50 sample (.BK suffix)")
        return [
            'PTT.BK', 'AOT.BK', 'CPALL.BK', 'ADVANC.BK', 'GULF.BK', 'DELTA.BK',
            'PTTEP.BK', 'BDMS.BK', 'SCB.BK', 'KBANK.BK', 'BBL.BK', 'KTB.BK',
            'CPN.BK', 'CPF.BK', 'TRUE.BK', 'INTUCH.BK', 'EA.BK', 'HMPRO.BK',
            'LH.BK', 'MINT.BK', 'BH.BK', 'IVL.BK', 'GPSC.BK', 'BEM.BK',
            'BTS.BK', 'TU.BK', 'OSP.BK', 'TISCO.BK', 'RATCH.BK', 'BJC.BK'
        ]

    def _get_malaysia_klci_tickers(self) -> List[str]:
        """Malaysia KLCI — FTSE Bursa Malaysia Top 30 (.KL suffix)."""
        logger.info("Using Malaysia KLCI 30 (.KL suffix)")
        return [
            '1155.KL', '5347.KL', '1023.KL', '5225.KL', '6033.KL', '1295.KL',
            '5285.KL', '1961.KL', '4707.KL', '6888.KL', '4197.KL', '3816.KL',
            '4863.KL', '1082.KL', '5183.KL', '5296.KL', '6012.KL', '5168.KL',
            '5819.KL', '3182.KL', '2445.KL', '5099.KL', '8869.KL', '4715.KL',
            '5681.KL', '4197.KL', '1066.KL', '5398.KL'
        ]

    def _get_bangladesh_dse_tickers(self) -> List[str]:
        """Bangladesh DSE — top names.

        Note: yfinance coverage for Bangladesh (.DH) is very limited.
        Included for completeness; expect many symbols to return no data.
        For reliable Bangladesh exposure, use frontier-market ETFs (FM, FRN).
        """
        logger.info("Using Bangladesh DSE sample (LIMITED yfinance coverage)")
        return [
            'GP.DH', 'BATBC.DH', 'SQURPHARMA.DH', 'ROBI.DH', 'BXPHARMA.DH',
            'BEXIMCO.DH', 'RENATA.DH', 'MARICO.DH', 'BRACBANK.DH', 'DBBL.DH',
            'CITYBANK.DH', 'EBL.DH', 'PUBALIBANK.DH', 'UTTARABANK.DH'
        ]

    def _get_egypt_egx30_tickers(self) -> List[str]:
        """Egypt EGX 30 — top 30 on Egyptian Exchange (.CA suffix).

        Note: yfinance .CA coverage varies; some names may be sparse.
        """
        logger.info("Using Egypt EGX 30 (.CA suffix, coverage varies)")
        return [
            'COMI.CA', 'HRHO.CA', 'EAST.CA', 'ETEL.CA', 'EFIH.CA', 'ORWE.CA',
            'SWDY.CA', 'MNHD.CA', 'PHDC.CA', 'AMOC.CA', 'JUFO.CA', 'ALCN.CA',
            'ORAS.CA', 'ESRS.CA', 'ABUK.CA', 'CIEB.CA', 'ADIB.CA', 'CIRA.CA',
            'FWRY.CA', 'ISPH.CA', 'MFPC.CA', 'MASR.CA', 'PHAR.CA', 'RMDA.CA',
            'ADPC.CA', 'CANA.CA', 'CLHO.CA', 'CSAG.CA', 'EGCH.CA', 'EGTS.CA'
        ]

    def _get_saudi_tasi_tickers(self) -> List[str]:
        """Saudi Tadawul TASI — top Saudi names (.SR suffix).

        Vision 2030 has driven significant reforms; Aramco IPO opened
        the market to global investors.
        """
        logger.info("Using Saudi TASI top names (.SR suffix)")
        return [
            '2222.SR', '2010.SR', '1120.SR', '1180.SR', '7010.SR', '1211.SR',
            '2020.SR', '2280.SR', '1150.SR', '1050.SR', '1140.SR', '2350.SR',
            '4030.SR', '5110.SR', '4002.SR', '4190.SR', '4240.SR', '4200.SR',
            '2380.SR', '1010.SR', '1080.SR', '1060.SR', '2290.SR', '2040.SR',
            '4001.SR', '4090.SR', '4260.SR', '4322.SR', '3030.SR', '3040.SR'
        ]

    def _get_uae_tickers(self) -> List[str]:
        """UAE — Abu Dhabi (.AE) and Dubai (.DU) top names."""
        logger.info("Using UAE ADX/DFM top names (.AE / .DU suffix)")
        return [
            # Abu Dhabi (ADX)
            'IHC.AE', 'FAB.AE', 'ADCB.AE', 'ETISALAT.AE', 'ADNOCDIST.AE',
            'ALDAR.AE', 'MULTIPLY.AE', 'BOROUGE.AE', 'ADNOCGAS.AE',
            'PUREHEALTH.AE', 'TAQA.AE', 'AGTHIA.AE', 'ADIB.AE',
            # Dubai (DFM)
            'EMAAR.DU', 'EMIRATESNBD.DU', 'DEWA.DU', 'DIB.DU', 'SALIK.DU',
            'DFM.DU', 'DAMAC.DU', 'AMLAK.DU', 'MASHREQ.DU', 'ARMX.DU'
        ]

    def _get_turkey_bist100_tickers(self) -> List[str]:
        """Turkey BIST 100 — top Borsa Istanbul names (.IS suffix)."""
        logger.info("Using Turkey BIST 100 sample (.IS suffix)")
        return [
            'AKBNK.IS', 'ARCLK.IS', 'ASELS.IS', 'BIMAS.IS', 'EKGYO.IS',
            'EREGL.IS', 'FROTO.IS', 'GARAN.IS', 'HALKB.IS', 'ISCTR.IS',
            'KCHOL.IS', 'KOZAA.IS', 'KOZAL.IS', 'KRDMD.IS', 'PETKM.IS',
            'PGSUS.IS', 'SAHOL.IS', 'SASA.IS', 'SISE.IS', 'TAVHL.IS',
            'TCELL.IS', 'THYAO.IS', 'TOASO.IS', 'TSKB.IS', 'TUPRS.IS',
            'VAKBN.IS', 'VESTL.IS', 'YKBNK.IS', 'ENKAI.IS', 'MGROS.IS'
        ]

    def _get_poland_wig20_tickers(self) -> List[str]:
        """Poland WIG20 — Warsaw Stock Exchange top 20 (.WA suffix)."""
        logger.info("Using Poland WIG20 (.WA suffix)")
        return [
            'PKO.WA', 'PZU.WA', 'PKN.WA', 'PEO.WA', 'ALE.WA', 'DNP.WA',
            'CPS.WA', 'CDR.WA', 'LPP.WA', 'KGH.WA', 'PGE.WA', 'JSW.WA',
            'ORLEN.WA', 'CCC.WA', 'MBK.WA', 'ALR.WA', 'OPL.WA', 'ACP.WA',
            'ATT.WA', 'SPL.WA'
        ]

    def _get_pakistan_kse100_tickers(self) -> List[str]:
        """Pakistan KSE 100 — top Karachi names.

        Note: yfinance coverage for Pakistan is limited; expect gaps.
        """
        logger.info("Using Pakistan KSE 100 sample (coverage limited)")
        return [
            'OGDC.KA', 'PPL.KA', 'MCB.KA', 'HBL.KA', 'UBL.KA', 'ENGRO.KA',
            'LUCK.KA', 'FFC.KA', 'HUBC.KA', 'PSO.KA', 'MARI.KA', 'NBP.KA',
            'BAHL.KA', 'MEBL.KA', 'PAKT.KA', 'DGKC.KA', 'FCCL.KA', 'INDU.KA'
        ]

    def _get_growth_markets_adrs(self) -> List[str]:
        """Curated US-listed ADRs from high-growth economies.

        Easiest way to gain growth-market exposure without dealing with
        foreign exchange settlement, custody, or thin liquidity.
        Covers: India, Vietnam, Indonesia, Philippines, Egypt, Turkey,
        Saudi Arabia (via ETF), Argentina, plus growth-adjacent LatAm.
        """
        logger.info("Using high-growth-economy ADRs (US-listed)")
        return [
            # India
            'INFY', 'WIT', 'HDB', 'IBN', 'RDY', 'TTM', 'MMYT', 'AZRE',
            # Vietnam (ETF only — no single-stock ADRs)
            'VNM',
            # Indonesia (ETF + ADR)
            'EIDO', 'TLK',
            # Philippines (ETF)
            'EPHE',
            # Egypt & MENA
            'EGPT', 'MES',
            # Turkey
            'TUR', 'TKC',
            # Saudi Arabia (ETF)
            'KSA',
            # UAE / Gulf (ETF)
            'UAE', 'GULF',
            # Argentina (high growth potential post-reform)
            'YPF', 'BMA', 'GGAL', 'PAM', 'TEO', 'CEPU', 'IRS',
            # Poland
            'PLND',
            # Frontier markets (ETFs — captures Bangladesh, Kenya, Nigeria, etc.)
            'FM', 'FRN',
            # LatAm growth-adjacent (Peru, Chile, Colombia)
            'EPU', 'ECH', 'GXG', 'BAP', 'SQM', 'EC',
            # Africa broad (beyond South Africa)
            'AFK',
        ]

    def calculate_graham_number(self, ticker: str) -> Dict[str, float]:
        """
        Calculate Graham Number and related metrics.

        Graham Number = sqrt(22.5 × EPS × Book Value per Share)

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dictionary with graham_number, intrinsic_value, current_price, margin_of_safety
        """
        metrics = self.get_key_metrics(ticker)

        eps = metrics.get('eps', 0)
        book_value = metrics.get('book_value_per_share', 0)
        current_price = metrics.get('current_price', 0)

        # Default MoS to NaN when the Graham Number cannot be calculated
        # (negative EPS or negative book value). Returning 0 here was misleading —
        # downstream code read it as "priced at intrinsic value".
        result = {
            'ticker': ticker,
            'eps': eps,
            'book_value_per_share': book_value,
            'current_price': current_price,
            'graham_number': 0,
            'margin_of_safety': np.nan,
            'is_undervalued': False
        }

        # Calculate Graham Number
        if eps > 0 and book_value > 0:
            graham_number = np.sqrt(22.5 * eps * book_value)
            result['graham_number'] = graham_number

            # Calculate margin of safety
            if current_price > 0:
                margin_of_safety = (graham_number - current_price) / graham_number
                result['margin_of_safety'] = margin_of_safety
                result['is_undervalued'] = margin_of_safety > 0.33  # 33% discount

        return result
