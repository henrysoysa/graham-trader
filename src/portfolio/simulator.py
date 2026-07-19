"""Portfolio simulation and backtesting engine"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging

from .portfolio import Portfolio
from ..data.data_fetcher import DataFetcher
from ..screening.graham_criteria import GrahamScreener

logger = logging.getLogger(__name__)


class PortfolioSimulator:
    """
    Simulates portfolio performance using Benjamin Graham criteria.
    """

    def __init__(
        self,
        data_fetcher: DataFetcher,
        screener: GrahamScreener,
        initial_capital: float = 100000,
        max_positions: int = 20,
        position_size: float = 0.05,
        rebalance_frequency: str = 'quarterly'
    ):
        """
        Initialize portfolio simulator.

        Args:
            data_fetcher: DataFetcher instance
            screener: GrahamScreener instance
            initial_capital: Starting capital
            max_positions: Maximum number of positions
            position_size: Target position size as fraction of portfolio (0.05 = 5%)
            rebalance_frequency: How often to rebalance ('monthly', 'quarterly', 'annually')
        """
        self.data_fetcher = data_fetcher
        self.screener = screener
        self.initial_capital = initial_capital
        self.max_positions = max_positions
        self.position_size = position_size
        self.rebalance_frequency = rebalance_frequency
        self.portfolio = Portfolio(initial_capital=initial_capital)

    def backtest(
        self,
        universe: List[str],
        start_date: str,
        end_date: str,
        strategy: str = 'defensive'
    ) -> Dict:
        """
        Backtest the Graham strategy over a historical period.

        Args:
            universe: List of tickers to screen from
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            strategy: 'defensive' or 'enterprising'

        Returns:
            Dictionary with backtest results and metrics
        """
        logger.info(f"Starting backtest from {start_date} to {end_date}")
        logger.info(f"Universe: {len(universe)} stocks, Strategy: {strategy}")

        # Parse dates
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)

        # Generate rebalance dates
        rebalance_dates = self._generate_rebalance_dates(start, end)

        # Track benchmark (S&P 500)
        benchmark_values = []

        # Run simulation
        for rebalance_date in rebalance_dates:
            logger.info(f"\n=== Rebalancing on {rebalance_date.date()} ===")

            # Screen universe for qualifying stocks
            if strategy == 'defensive':
                screening_results = self.screener.screen_defensive(universe)
            else:
                screening_results = self.screener.screen_enterprising(universe)

            if screening_results.empty:
                logger.warning(f"No stocks passed screening on {rebalance_date.date()}")
                continue

            # Get top candidates
            top_candidates = screening_results[
                screening_results['passes_screening'] == True
            ].head(self.max_positions)

            logger.info(f"Found {len(top_candidates)} qualifying stocks")

            # Rebalance portfolio
            self._rebalance_portfolio(top_candidates, rebalance_date)

            # Record portfolio snapshot
            current_prices = self._get_current_prices(
                list(self.portfolio.holdings.keys()),
                rebalance_date
            )
            self.portfolio.record_snapshot(rebalance_date, current_prices)

        # Calculate final metrics
        results = self._calculate_backtest_metrics(start_date, end_date)

        return results

    def _generate_rebalance_dates(
        self,
        start: pd.Timestamp,
        end: pd.Timestamp
    ) -> List[pd.Timestamp]:
        """Generate rebalance dates based on frequency."""
        dates = []

        if self.rebalance_frequency == 'monthly':
            freq = pd.offsets.MonthEnd(1)
        elif self.rebalance_frequency == 'quarterly':
            freq = pd.offsets.QuarterEnd(1)
        elif self.rebalance_frequency == 'annually':
            freq = pd.offsets.YearEnd(1)
        else:
            freq = pd.offsets.QuarterEnd(1)

        current = start
        while current <= end:
            dates.append(current)
            current += freq

        return dates

    def _rebalance_portfolio(
        self,
        target_stocks: pd.DataFrame,
        date: pd.Timestamp
    ):
        """
        Rebalance portfolio to match target stocks.

        Args:
            target_stocks: DataFrame of stocks that passed screening
            date: Rebalance date
        """
        # Get current holdings
        current_holdings = set(self.portfolio.holdings.keys())
        target_tickers = set(target_stocks['ticker'].tolist())

        # Stocks to sell (in portfolio but not in target)
        to_sell = current_holdings - target_tickers

        # Stocks to buy (in target but not in portfolio)
        to_buy = target_tickers - current_holdings

        # Get current prices
        all_tickers = list(current_holdings.union(target_tickers))
        current_prices = self._get_current_prices(all_tickers, date)

        # Sell positions no longer in target
        for ticker in to_sell:
            if ticker in self.portfolio.holdings:
                shares = self.portfolio.holdings[ticker]['shares']
                price = current_prices.get(ticker, 0)
                if price > 0:
                    self.portfolio.sell(ticker, shares, price, date)

        # Calculate target allocation per position
        total_value = self.portfolio.get_total_value(current_prices)
        target_allocation = total_value * self.position_size

        # Buy new positions
        for ticker in to_buy:
            price = current_prices.get(ticker, 0)
            if price > 0:
                shares = int(target_allocation / price)
                if shares > 0:
                    self.portfolio.buy(ticker, shares, price, date)

    def _get_current_prices(
        self,
        tickers: List[str],
        date: pd.Timestamp
    ) -> Dict[str, float]:
        """
        Get historical prices for tickers at a specific date.

        Args:
            tickers: List of ticker symbols
            date: Date to get prices for

        Returns:
            Dictionary of {ticker: price}
        """
        prices = {}

        for ticker in tickers:
            try:
                # Get historical data around the date
                start_date = (date - timedelta(days=7)).strftime('%Y-%m-%d')
                end_date = (date + timedelta(days=1)).strftime('%Y-%m-%d')

                hist_data = self.data_fetcher.get_historical_prices(
                    ticker,
                    start_date=start_date,
                    end_date=end_date
                )

                if not hist_data.empty and 'Close' in hist_data.columns:
                    # Get closest price to target date
                    closest_date = hist_data.index[
                        hist_data.index.get_indexer([date], method='nearest')[0]
                    ]
                    prices[ticker] = hist_data.loc[closest_date, 'Close']
                else:
                    logger.warning(f"No price data for {ticker} on {date.date()}")
                    prices[ticker] = 0
            except Exception as e:
                logger.error(f"Error fetching price for {ticker}: {e}")
                prices[ticker] = 0

        return prices

    def _calculate_backtest_metrics(
        self,
        start_date: str,
        end_date: str
    ) -> Dict:
        """Calculate performance metrics from backtest."""
        history_df = self.portfolio.get_history_df()

        if history_df.empty:
            return {}

        # Calculate returns
        history_df['returns'] = history_df['total_value'].pct_change()

        # Total return
        final_value = history_df['total_value'].iloc[-1]
        total_return = final_value - self.initial_capital
        total_return_pct = (total_return / self.initial_capital) * 100

        # Calculate time period in years
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        years = (end - start).days / 365.25

        # Annualized return
        annualized_return = ((final_value / self.initial_capital) ** (1 / years) - 1) * 100

        # Volatility (annualized)
        daily_vol = history_df['returns'].std()
        annualized_vol = daily_vol * np.sqrt(252)  # Assuming quarterly rebalancing

        # Sharpe ratio (assuming 0% risk-free rate)
        sharpe_ratio = annualized_return / (annualized_vol * 100) if annualized_vol > 0 else 0

        # Max drawdown
        cumulative = (1 + history_df['returns']).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min() * 100

        # Number of trades
        transactions_df = self.portfolio.get_transactions_df()
        num_trades = len(transactions_df)

        # Win rate
        if not transactions_df.empty:
            sell_transactions = transactions_df[transactions_df['action'] == 'SELL']
            if not sell_transactions.empty and 'gain_loss' in sell_transactions.columns:
                wins = (sell_transactions['gain_loss'] > 0).sum()
                win_rate = (wins / len(sell_transactions)) * 100 if len(sell_transactions) > 0 else 0
            else:
                win_rate = 0
        else:
            win_rate = 0

        results = {
            'start_date': start_date,
            'end_date': end_date,
            'years': round(years, 2),
            'initial_capital': self.initial_capital,
            'final_value': final_value,
            'total_return': total_return,
            'total_return_pct': total_return_pct,
            'annualized_return': annualized_return,
            'annualized_volatility': annualized_vol * 100,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'num_trades': num_trades,
            'win_rate': win_rate,
            'portfolio_history': history_df,
            'transactions': transactions_df,
            'final_holdings': self.portfolio.get_holdings_summary(
                self._get_current_prices(
                    list(self.portfolio.holdings.keys()),
                    pd.to_datetime(end_date)
                )
            ) if self.portfolio.holdings else pd.DataFrame()
        }

        return results

    def simulate_strategy(
        self,
        tickers: List[str],
        strategy: str = 'defensive'
    ) -> pd.DataFrame:
        """
        Simulate current strategy on given tickers.

        Args:
            tickers: List of tickers to evaluate
            strategy: 'defensive' or 'enterprising'

        Returns:
            DataFrame with simulation results
        """
        # Screen stocks
        if strategy == 'defensive':
            results = self.screener.screen_defensive(tickers)
        else:
            results = self.screener.screen_enterprising(tickers)

        # Get top candidates
        candidates = results[results['passes_screening'] == True].head(self.max_positions)

        # Calculate position sizes
        if not candidates.empty:
            target_allocation = self.initial_capital * self.position_size
            candidates['recommended_shares'] = (
                target_allocation / candidates['current_price']
            ).astype(int)
            candidates['position_value'] = (
                candidates['recommended_shares'] * candidates['current_price']
            )

        return candidates
