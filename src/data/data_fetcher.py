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

    def get_index_tickers(self, index_name: str = 'SP500') -> List[str]:
        """
        Get list of tickers for various stock market indexes.

        Args:
            index_name: Name of the index (SP500, NASDAQ100, DOW30, FTSE100, DAX, CAC40, NIKKEI225, ASX200, TSX60)

        Returns:
            List of ticker symbols
        """
        index_map = {
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
        }

        fetcher = index_map.get(index_name, self._get_sp500_tickers)
        return fetcher()

    def get_sp500_tickers(self) -> List[str]:
        """Legacy method - calls get_index_tickers('SP500')"""
        return self.get_index_tickers('SP500')

    def _get_sp500_tickers(self) -> List[str]:
        """
        Get list of S&P 500 tickers.

        Returns:
            List of ticker symbols
        """
        try:
            # Get S&P 500 tickers from Wikipedia
            url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
            import ssl
            import certifi
            ssl_context = ssl.create_default_context(cafile=certifi.where())
            tables = pd.read_html(url, storage_options={'ssl': ssl_context})
            sp500_table = tables[0]
            tickers = sp500_table['Symbol'].tolist()
            # Clean tickers (remove dots for compatibility)
            tickers = [ticker.replace('.', '-') for ticker in tickers]
            return tickers
        except Exception as e:
            logger.warning(f"Error fetching S&P 500 tickers from Wikipedia: {e}")
            # Fallback to a curated list of major S&P 500 stocks
            logger.info("Using fallback list of major S&P 500 stocks")
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
