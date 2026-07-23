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
    """

    def __init__(self, metrics, graham):
        self._metrics = metrics
        self._graham = graham

    def get_key_metrics(self, ticker):
        return dict(self._metrics)

    def calculate_graham_number(self, ticker):
        return dict(self._graham)
