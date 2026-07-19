"""
Configuration file for Graham Trader application
"""
import os

# API Keys (set these as environment variables)
FINANCIAL_MODELING_PREP_API_KEY = os.getenv('FMP_API_KEY', '')
ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY', '')

# Benjamin Graham Criteria Defaults
# NOTE: These have been modernized for 2026 markets while maintaining Graham's value investing philosophy
GRAHAM_CRITERIA = {
    # Defensive Investor Criteria (Updated for modern markets)
    'defensive': {
        'min_earnings_stability': 10,  # years of positive earnings
        'min_dividend_history': 10,  # years of continuous dividends (reduced from 20)
        'min_earnings_growth': 0.33,  # 33% growth over 10 years (3% annualized)
        'max_pe_ratio': 25,  # Increased from 15 to reflect modern valuations
        'max_pb_ratio': 4.0,  # Increased from 1.5 to reflect modern valuations
        'min_current_ratio': 1.2,  # Reduced from 2.0 (companies are more efficient now)
        'max_debt_to_current_assets': 1.5,  # Slightly more lenient
        'margin_of_safety': 0.20,  # 20% discount to intrinsic value (reduced from 33%)
    },
    # Enterprising Investor Criteria (more relaxed)
    'enterprising': {
        'min_earnings_stability': 5,
        'max_pe_ratio': 35,  # Increased from 25
        'max_pb_ratio': 5.0,  # Increased from 2.5
        'min_current_ratio': 1.0,  # Reduced from 1.5
        'margin_of_safety': 0.15,  # 15% discount to intrinsic value (reduced from 25%)
    },
    # Graham Number calculation
    'graham_number': {
        'eps_multiplier': 15,
        'book_value_multiplier': 1.5,
        'max_multiplier': 22.5,  # sqrt(22.5 * EPS * BVPS)
    }
}

# Portfolio Simulation Settings
PORTFOLIO_SETTINGS = {
    'initial_capital': 100000,  # $100,000 starting capital
    'max_position_size': 0.1,  # 10% max per position
    'rebalance_frequency': 'quarterly',  # or 'monthly', 'annually'
    'commission': 0,  # commission per trade (0 for modern brokers)
}

# Data caching settings
CACHE_ENABLED = True
CACHE_EXPIRY_HOURS = 24

# Screening universe
DEFAULT_UNIVERSE = 'SP500'  # or 'NASDAQ100', 'DOW30', 'RUSSELL2000', 'ALL'
