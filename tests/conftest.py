"""Shared pytest fixtures and path setup."""
import os
import sys

# Make the project root importable (so `import src...` and `import config` work
# regardless of where pytest is invoked from).
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


class StubFetcher:
    """A DataFetcher stand-in that returns canned metrics.

    Lets us test the GrahamScreener scoring logic deterministically without
    any network access or yfinance dependency.

    ``earnings_history`` is an optional list of annual EPS values in
    chronological (oldest→newest) order; ``None`` means "no history available".
    """

    def __init__(self, metrics, graham, earnings_history=None):
        self._metrics = metrics
        self._graham = graham
        self._earnings_history = earnings_history

    def get_key_metrics(self, ticker):
        return dict(self._metrics)

    def calculate_graham_number(self, ticker):
        return dict(self._graham)

    def calculate_graham_number_from_metrics(self, ticker, metrics):
        return dict(self._graham)

    def get_earnings_history(self, ticker, years=10):
        import pandas as pd
        if self._earnings_history is None:
            return pd.DataFrame()
        return pd.DataFrame({'EPS': list(self._earnings_history)})
