"""
Benjamin Graham screening criteria implementation.

Includes:
- Graham Number calculation
- Margin of Safety
- Defensive Investor criteria
- Enterprising Investor criteria
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import logging

from ..data.data_fetcher import DataFetcher

logger = logging.getLogger(__name__)


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
        """Return default Graham criteria."""
        return {
            'defensive': {
                'min_earnings_stability': 10,
                'min_dividend_history': 20,
                'min_earnings_growth': 0.33,
                'max_pe_ratio': 15,
                'max_pb_ratio': 1.5,
                'min_current_ratio': 2.0,
                'max_debt_to_current_assets': 1.1,
                'margin_of_safety': 0.33,
            },
            'enterprising': {
                'min_earnings_stability': 5,
                'max_pe_ratio': 25,
                'max_pb_ratio': 2.5,
                'min_current_ratio': 1.5,
                'margin_of_safety': 0.25,
            }
        }

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

        # 4. Earnings stability (simplified - check if EPS is positive)
        if result['eps'] > 0:
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

        # 9. Margin of safety vs Graham Number — core Graham principle.
        # Only meaningful when Graham Number could be computed (EPS > 0, BV > 0).
        max_score = 9
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

        # Overall pass: need 6/9 criteria AND a positive margin of safety.
        # Without the MoS gate, stocks trading well above intrinsic value can
        # still pass on other metrics — defeating the point of value screening.
        result['passes_screening'] = (
            total_score >= 6 and result['margin_of_safety_check']
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

        # 3. Positive earnings
        if result['eps'] > 0:
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
