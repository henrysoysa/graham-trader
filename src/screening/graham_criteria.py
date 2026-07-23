"""
Benjamin Graham screening criteria implementation.

Includes:
- Graham Number calculation
- Margin of Safety
- Defensive Investor criteria
- Enterprising Investor criteria
"""
import copy
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import logging

from ..data.data_fetcher import DataFetcher

logger = logging.getLogger(__name__)

# Fallback criteria, used only if the project-level ``config`` module cannot be
# imported. These MUST mirror ``config.GRAHAM_CRITERIA`` (modernised thresholds)
# so behaviour is identical whether or not a config is supplied.
_FALLBACK_CRITERIA = {
    'defensive': {
        'min_earnings_stability': 10,
        'min_dividend_history': 10,
        'min_earnings_growth': 0.33,
        'max_pe_ratio': 25,
        'max_pb_ratio': 4.0,
        'min_current_ratio': 1.2,
        'max_debt_to_current_assets': 1.5,
        'margin_of_safety': 0.20,
    },
    'enterprising': {
        'min_earnings_stability': 5,
        'max_pe_ratio': 35,
        'max_pb_ratio': 5.0,
        'min_current_ratio': 1.0,
        'margin_of_safety': 0.15,
    },
    'graham_number': {
        'eps_multiplier': 15,
        'book_value_multiplier': 1.5,
        'max_multiplier': 22.5,
    },
}


class GrahamScreener:
    """
    Implements Benjamin Graham's value investing screening criteria.
    """

    def __init__(self, data_fetcher: DataFetcher, criteria_config: Optional[Dict] = None):
        """
        Initialize Graham Screener.

        Args:
            data_fetcher: DataFetcher instance for retrieving stock data
            criteria_config: Optional custom criteria configuration
        """
        self.data_fetcher = data_fetcher
        self.criteria = criteria_config or self._default_criteria()

    def _default_criteria(self) -> Dict:
        """Return default Graham criteria.

        Single source of truth is ``config.GRAHAM_CRITERIA`` so the strict and
        modernised thresholds can never silently diverge. If that module is not
        importable we fall back to an identical inline copy.
        """
        try:
            import config
            return copy.deepcopy(config.GRAHAM_CRITERIA)
        except Exception:
            logger.warning("config.GRAHAM_CRITERIA unavailable; using inline fallback criteria")
            return copy.deepcopy(_FALLBACK_CRITERIA)

    def screen_defensive(self, tickers: List[str]) -> pd.DataFrame:
        """
        Screen stocks using Benjamin Graham's Defensive Investor criteria.

        Defensive Investor Criteria:
        1. Adequate size (market cap > $2B)
        2. Strong financial condition (Current Ratio > 2.0)
        3. Earnings stability (10 years of positive earnings)
        4. Dividend record (20 years of continuous dividends)
        5. Earnings growth (33% over 10 years)
        6. Moderate P/E ratio (< 15)
        7. Moderate P/B ratio (< 1.5)
        8. P/E × P/B < 22.5

        Args:
            tickers: List of stock ticker symbols

        Returns:
            DataFrame with screening results and scores
        """
        results = []

        for ticker in tickers:
            try:
                logger.info(f"Screening {ticker} (Defensive)")
                score = self._evaluate_defensive(ticker)
                if score:
                    results.append(score)
            except Exception as e:
                logger.error(f"Error screening {ticker}: {e}")

        df = pd.DataFrame(results)
        if not df.empty:
            df = df.sort_values('total_score', ascending=False)

        return df

    def screen_enterprising(self, tickers: List[str]) -> pd.DataFrame:
        """
        Screen stocks using Benjamin Graham's Enterprising Investor criteria.

        Enterprising Investor has more relaxed criteria but still focuses on value.

        Args:
            tickers: List of stock ticker symbols

        Returns:
            DataFrame with screening results and scores
        """
        results = []

        for ticker in tickers:
            try:
                logger.info(f"Screening {ticker} (Enterprising)")
                score = self._evaluate_enterprising(ticker)
                if score:
                    results.append(score)
            except Exception as e:
                logger.error(f"Error screening {ticker}: {e}")

        df = pd.DataFrame(results)
        if not df.empty:
            df = df.sort_values('total_score', ascending=False)

        return df

    def _earnings_history_eps(self, ticker: str) -> Optional[pd.Series]:
        """Return a chronological (oldest→newest) EPS series, or None.

        Wraps DataFetcher.get_earnings_history and normalises ordering so the
        stability/growth checks don't depend on the source's row order.
        """
        try:
            hist = self.data_fetcher.get_earnings_history(ticker)
        except Exception as e:  # pragma: no cover - defensive
            logger.debug(f"earnings history unavailable for {ticker}: {e}")
            return None
        if hist is None or getattr(hist, 'empty', True) or 'EPS' not in getattr(hist, 'columns', []):
            return None
        eps = hist['EPS'].dropna().sort_index()
        return eps if len(eps) >= 1 else None

    def _earnings_stability_ok(self, ticker: str, min_years: int, current_eps: float) -> bool:
        """Earnings stability = no annual losses across available history.

        Graham asked for ``min_years`` of positive earnings. yfinance rarely
        provides more than ~4 years of statements, so we verify that every year
        we *can* see is positive (a real multi-year check when data allows) and
        fall back to the trailing-EPS sign when no history is available — the
        latter keeps sparsely-covered non-US names from being auto-failed on a
        data gap rather than on fundamentals.
        """
        eps = self._earnings_history_eps(ticker)
        if eps is not None:
            return bool((eps > 0).all())
        return bool(current_eps is not None and current_eps > 0)

    def _earnings_growth_ok(self, ticker: str, min_total_growth_10y: float, current_growth: float) -> bool:
        """Earnings growth vs Graham's target (default 33% over 10 years).

        Computes annualised EPS growth from available history and compares to
        the annualised equivalent of the 10-year target. Falls back to the
        trailing earnings-growth figure when history is too short.
        """
        eps = self._earnings_history_eps(ticker)
        if eps is not None and len(eps) >= 2:
            oldest = float(eps.iloc[0])
            newest = float(eps.iloc[-1])
            years = max(len(eps) - 1, 1)
            if oldest > 0 and newest > 0:
                annualised = (newest / oldest) ** (1.0 / years) - 1.0
                target_annualised = (1.0 + min_total_growth_10y) ** (1.0 / 10.0) - 1.0
                return bool(annualised >= target_annualised)
            # Sign change (loss→profit or profit→loss): use direction.
            return bool(newest > oldest)
        return bool(current_growth is not None and current_growth > 0)

    def _evaluate_defensive(self, ticker: str) -> Optional[Dict]:
        """
        Evaluate a stock against defensive investor criteria.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dictionary with evaluation results or None if data unavailable
        """
        metrics = self.data_fetcher.get_key_metrics(ticker)
        graham_calc = self.data_fetcher.calculate_graham_number(ticker)

        if not metrics or metrics.get('current_price', 0) == 0:
            return None

        criteria = self.criteria['defensive']
        result = {
            'ticker': ticker,
            'current_price': metrics.get('current_price', 0),
            'market_cap': metrics.get('market_cap', 0),
            'eps': metrics.get('eps', 0),
            'pe_ratio': metrics.get('pe_ratio', 0),
            'pb_ratio': metrics.get('pb_ratio', 0),
            'current_ratio': metrics.get('current_ratio', 0),
            'debt_to_current_assets': metrics.get('debt_to_current_assets', 0),
            'dividend_yield': metrics.get('dividend_yield', 0),
            'graham_number': graham_calc.get('graham_number', 0),
            'margin_of_safety': graham_calc.get('margin_of_safety', 0),
        }

        # Evaluate each criterion and assign points
        total_score = 0
        max_score = 8

        # 1. Adequate size (Market cap > $2B)
        if result['market_cap'] > 2_000_000_000:
            total_score += 1
            result['size_check'] = True
        else:
            result['size_check'] = False

        # 2. Strong financial condition (Current Ratio > 2.0)
        if result['current_ratio'] >= criteria['min_current_ratio']:
            total_score += 1
            result['financial_condition_check'] = True
        else:
            result['financial_condition_check'] = False

        # 3. Debt to Current Assets
        if result['debt_to_current_assets'] <= criteria['max_debt_to_current_assets']:
            total_score += 1
            result['debt_check'] = True
        else:
            result['debt_check'] = False

        # 4. Earnings stability — no annual losses across available history
        #    (falls back to trailing-EPS sign when history is unavailable).
        if self._earnings_stability_ok(
            ticker, criteria.get('min_earnings_stability', 10), result['eps']
        ):
            total_score += 1
            result['earnings_stability_check'] = True
        else:
            result['earnings_stability_check'] = False

        # 5. Dividend record (check if dividend yield exists)
        if result['dividend_yield'] > 0:
            total_score += 1
            result['dividend_check'] = True
        else:
            result['dividend_check'] = False

        # 6. Moderate P/E ratio
        if 0 < result['pe_ratio'] <= criteria['max_pe_ratio']:
            total_score += 1
            result['pe_check'] = True
        else:
            result['pe_check'] = False

        # 7. Moderate P/B ratio
        if 0 < result['pb_ratio'] <= criteria['max_pb_ratio']:
            total_score += 1
            result['pb_check'] = True
        else:
            result['pb_check'] = False

        # 8. Graham's multiplier: P/E × P/B ≤ 22.5
        pe_pb_product = result['pe_ratio'] * result['pb_ratio']
        if 0 < pe_pb_product <= 22.5:
            total_score += 1
            result['graham_multiplier_check'] = True
        else:
            result['graham_multiplier_check'] = False

        # 9. Earnings growth vs Graham's target (default 33% over 10 years).
        if self._earnings_growth_ok(
            ticker,
            criteria.get('min_earnings_growth', 0.33),
            metrics.get('earnings_growth', 0),
        ):
            total_score += 1
            result['earnings_growth_check'] = True
        else:
            result['earnings_growth_check'] = False

        # 10. Margin of safety vs Graham Number — core Graham principle.
        # Only meaningful when Graham Number could be computed (EPS > 0, BV > 0).
        max_score = 10
        min_mos = criteria.get('margin_of_safety', 0.20)
        if result['graham_number'] > 0 and result['margin_of_safety'] >= min_mos:
            total_score += 1
            result['margin_of_safety_check'] = True
        else:
            result['margin_of_safety_check'] = False

        result['pe_pb_product'] = pe_pb_product
        result['total_score'] = total_score
        result['max_score'] = max_score
        result['pass_percentage'] = (total_score / max_score) * 100

        # Overall pass: need 7/10 criteria AND a positive margin of safety.
        # Without the MoS gate, stocks trading well above intrinsic value can
        # still pass on other metrics — defeating the point of value screening.
        result['passes_screening'] = (
            total_score >= 7 and result['margin_of_safety_check']
        )

        return result

    def _evaluate_enterprising(self, ticker: str) -> Optional[Dict]:
        """
        Evaluate a stock against enterprising investor criteria.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dictionary with evaluation results or None if data unavailable
        """
        metrics = self.data_fetcher.get_key_metrics(ticker)
        graham_calc = self.data_fetcher.calculate_graham_number(ticker)

        if not metrics or metrics.get('current_price', 0) == 0:
            return None

        criteria = self.criteria['enterprising']
        result = {
            'ticker': ticker,
            'current_price': metrics.get('current_price', 0),
            'market_cap': metrics.get('market_cap', 0),
            'eps': metrics.get('eps', 0),
            'pe_ratio': metrics.get('pe_ratio', 0),
            'pb_ratio': metrics.get('pb_ratio', 0),
            'current_ratio': metrics.get('current_ratio', 0),
            'graham_number': graham_calc.get('graham_number', 0),
            'margin_of_safety': graham_calc.get('margin_of_safety', 0),
        }

        total_score = 0
        max_score = 5

        # 1. Adequate size
        if result['market_cap'] > 1_000_000_000:  # $1B for enterprising
            total_score += 1
            result['size_check'] = True
        else:
            result['size_check'] = False

        # 2. Financial condition
        if result['current_ratio'] >= criteria['min_current_ratio']:
            total_score += 1
            result['financial_condition_check'] = True
        else:
            result['financial_condition_check'] = False

        # 3. Earnings stability — no annual losses across available history
        #    (falls back to trailing-EPS sign when history is unavailable).
        if self._earnings_stability_ok(
            ticker, criteria.get('min_earnings_stability', 5), result['eps']
        ):
            total_score += 1
            result['earnings_check'] = True
        else:
            result['earnings_check'] = False

        # 4. P/E ratio
        if 0 < result['pe_ratio'] <= criteria['max_pe_ratio']:
            total_score += 1
            result['pe_check'] = True
        else:
            result['pe_check'] = False

        # 5. P/B ratio
        if 0 < result['pb_ratio'] <= criteria['max_pb_ratio']:
            total_score += 1
            result['pb_check'] = True
        else:
            result['pb_check'] = False

        # 6. Margin of safety vs Graham Number — core Graham principle.
        max_score = 6
        min_mos = criteria.get('margin_of_safety', 0.15)
        if result['graham_number'] > 0 and result['margin_of_safety'] >= min_mos:
            total_score += 1
            result['margin_of_safety_check'] = True
        else:
            result['margin_of_safety_check'] = False

        result['total_score'] = total_score
        result['max_score'] = max_score
        result['pass_percentage'] = (total_score / max_score) * 100
        # Require MoS gate — enterprising investors still demand a discount.
        result['passes_screening'] = (
            total_score >= 5 and result['margin_of_safety_check']
        )

        return result

    def calculate_intrinsic_value(self, ticker: str, method: str = 'graham') -> Dict:
        """
        Calculate intrinsic value using various methods.

        Args:
            ticker: Stock ticker symbol
            method: Calculation method ('graham', 'dcf', 'earnings_multiplier')

        Returns:
            Dictionary with intrinsic value calculation
        """
        if method == 'graham':
            return self.data_fetcher.calculate_graham_number(ticker)
        else:
            raise NotImplementedError(f"Method '{method}' not yet implemented")

    def get_margin_of_safety(self, ticker: str) -> float:
        """
        Calculate margin of safety for a stock.

        Margin of Safety = (Intrinsic Value - Current Price) / Intrinsic Value

        Args:
            ticker: Stock ticker symbol

        Returns:
            Margin of safety as a decimal (0.33 = 33% discount)
        """
        graham_calc = self.data_fetcher.calculate_graham_number(ticker)
        return graham_calc.get('margin_of_safety', 0)
