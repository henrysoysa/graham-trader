"""Portfolio tracking and management"""
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class Portfolio:
    """
    Tracks portfolio holdings, transactions, and performance.
    """

    def __init__(self, initial_capital: float = 100000, name: str = "Graham Portfolio"):
        """
        Initialize portfolio.

        Args:
            initial_capital: Starting cash balance
            name: Portfolio name
        """
        self.name = name
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.holdings = {}  # {ticker: {'shares': int, 'avg_cost': float}}
        self.transactions = []  # List of transaction dictionaries
        self.history = []  # Portfolio value over time

    def buy(self, ticker: str, shares: int, price: float, date: Optional[datetime] = None):
        """
        Buy shares of a stock.

        Args:
            ticker: Stock ticker symbol
            shares: Number of shares to buy
            price: Price per share
            date: Transaction date (defaults to now)
        """
        total_cost = shares * price

        if total_cost > self.cash:
            logger.warning(f"Insufficient funds to buy {shares} shares of {ticker}")
            return False

        # Update cash
        self.cash -= total_cost

        # Update holdings
        if ticker in self.holdings:
            # Calculate new average cost
            current_shares = self.holdings[ticker]['shares']
            current_avg = self.holdings[ticker]['avg_cost']
            new_shares = current_shares + shares
            new_avg = ((current_shares * current_avg) + (shares * price)) / new_shares

            self.holdings[ticker]['shares'] = new_shares
            self.holdings[ticker]['avg_cost'] = new_avg
        else:
            self.holdings[ticker] = {
                'shares': shares,
                'avg_cost': price
            }

        # Record transaction
        transaction = {
            'date': date or datetime.now(),
            'ticker': ticker,
            'action': 'BUY',
            'shares': shares,
            'price': price,
            'total': total_cost,
            'cash_balance': self.cash
        }
        self.transactions.append(transaction)

        logger.info(f"Bought {shares} shares of {ticker} at ${price:.2f}")
        return True

    def sell(self, ticker: str, shares: int, price: float, date: Optional[datetime] = None):
        """
        Sell shares of a stock.

        Args:
            ticker: Stock ticker symbol
            shares: Number of shares to sell
            price: Price per share
            date: Transaction date (defaults to now)
        """
        if ticker not in self.holdings:
            logger.warning(f"Cannot sell {ticker} - not in portfolio")
            return False

        if self.holdings[ticker]['shares'] < shares:
            logger.warning(f"Cannot sell {shares} shares of {ticker} - only have {self.holdings[ticker]['shares']}")
            return False

        # Update holdings
        self.holdings[ticker]['shares'] -= shares
        if self.holdings[ticker]['shares'] == 0:
            del self.holdings[ticker]

        # Update cash
        total_proceeds = shares * price
        self.cash += total_proceeds

        # Calculate gain/loss
        avg_cost = self.holdings.get(ticker, {}).get('avg_cost', 0)
        gain_loss = (price - avg_cost) * shares

        # Record transaction
        transaction = {
            'date': date or datetime.now(),
            'ticker': ticker,
            'action': 'SELL',
            'shares': shares,
            'price': price,
            'total': total_proceeds,
            'gain_loss': gain_loss,
            'cash_balance': self.cash
        }
        self.transactions.append(transaction)

        logger.info(f"Sold {shares} shares of {ticker} at ${price:.2f} (Gain/Loss: ${gain_loss:.2f})")
        return True

    def get_holdings_value(self, current_prices: Dict[str, float]) -> float:
        """
        Calculate current value of all holdings.

        Args:
            current_prices: Dictionary of {ticker: current_price}

        Returns:
            Total value of holdings
        """
        total_value = 0
        for ticker, holding in self.holdings.items():
            if ticker in current_prices:
                total_value += holding['shares'] * current_prices[ticker]
        return total_value

    def get_total_value(self, current_prices: Dict[str, float]) -> float:
        """
        Calculate total portfolio value (cash + holdings).

        Args:
            current_prices: Dictionary of {ticker: current_price}

        Returns:
            Total portfolio value
        """
        return self.cash + self.get_holdings_value(current_prices)

    def get_returns(self, current_prices: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate portfolio returns.

        Args:
            current_prices: Dictionary of {ticker: current_price}

        Returns:
            Dictionary with total_return, total_return_pct, annualized_return
        """
        current_value = self.get_total_value(current_prices)
        total_return = current_value - self.initial_capital
        total_return_pct = (total_return / self.initial_capital) * 100

        return {
            'initial_capital': self.initial_capital,
            'current_value': current_value,
            'total_return': total_return,
            'total_return_pct': total_return_pct,
            'cash': self.cash,
            'holdings_value': self.get_holdings_value(current_prices)
        }

    def get_holdings_summary(self, current_prices: Dict[str, float]) -> pd.DataFrame:
        """
        Get summary of current holdings.

        Args:
            current_prices: Dictionary of {ticker: current_price}

        Returns:
            DataFrame with holdings information
        """
        holdings_list = []
        for ticker, holding in self.holdings.items():
            current_price = current_prices.get(ticker, 0)
            market_value = holding['shares'] * current_price
            cost_basis = holding['shares'] * holding['avg_cost']
            gain_loss = market_value - cost_basis
            gain_loss_pct = (gain_loss / cost_basis * 100) if cost_basis > 0 else 0

            holdings_list.append({
                'ticker': ticker,
                'shares': holding['shares'],
                'avg_cost': holding['avg_cost'],
                'current_price': current_price,
                'cost_basis': cost_basis,
                'market_value': market_value,
                'gain_loss': gain_loss,
                'gain_loss_pct': gain_loss_pct
            })

        return pd.DataFrame(holdings_list)

    def get_transactions_df(self) -> pd.DataFrame:
        """
        Get all transactions as a DataFrame.

        Returns:
            DataFrame with transaction history
        """
        return pd.DataFrame(self.transactions)

    def record_snapshot(self, date: datetime, current_prices: Dict[str, float]):
        """
        Record a snapshot of portfolio value at a given date.

        Args:
            date: Snapshot date
            current_prices: Current prices for all holdings
        """
        total_value = self.get_total_value(current_prices)
        holdings_value = self.get_holdings_value(current_prices)

        snapshot = {
            'date': date,
            'total_value': total_value,
            'cash': self.cash,
            'holdings_value': holdings_value,
            'num_positions': len(self.holdings)
        }
        self.history.append(snapshot)

    def get_history_df(self) -> pd.DataFrame:
        """
        Get portfolio value history as DataFrame.

        Returns:
            DataFrame with historical portfolio values
        """
        return pd.DataFrame(self.history)
