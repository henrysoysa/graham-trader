# Graham Trader

A comprehensive web application for Benjamin Graham value investing, featuring stock screening, portfolio simulation, and backtesting capabilities.

![Graham Trader](https://img.shields.io/badge/Python-3.9+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## Features

### 1. Stock Screener
- **Defensive Investor Criteria**: Conservative screening for passive investors
- **Enterprising Investor Criteria**: Active investor screening with relaxed parameters
- **Graham Number Calculation**: Automatic intrinsic value calculation
- **Margin of Safety Analysis**: Identify undervalued stocks

### 2. Portfolio Simulator
- Build portfolios based on screening results
- Automatic position sizing
- Real-time portfolio allocation
- Cash management

### 3. Backtesting Engine
- Test strategies on historical data
- Performance metrics: Total return, annualized return, Sharpe ratio, max drawdown
- Transaction history and win rate analysis
- Interactive charts and visualizations

## Installation

### Prerequisites
- Python 3.9 or higher
- pip package manager

### Quick Start

1. **Clone or navigate to the project directory**
```bash
cd graham-trader
```

2. **Create a virtual environment (recommended)**
```bash
python -m venv venv

# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the application**
```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

## Usage Guide

### Stock Screener

1. Navigate to the **Stock Screener** page
2. Select your investment strategy:
   - **Defensive Investor**: For conservative, passive investors
   - **Enterprising Investor**: For active investors willing to do more analysis
3. Choose your stock universe:
   - **S&P 500**: Screen all S&P 500 stocks
   - **Custom Tickers**: Enter specific tickers to analyze
4. Click **Run Screening** to analyze stocks
5. Review results showing stocks that pass Benjamin Graham's criteria

### Portfolio Simulator

1. Go to the **Portfolio Simulator** page
2. Select your strategy and initial capital
3. Choose your stock universe
4. Click **Simulate Portfolio** to get recommendations
5. Review the recommended portfolio with position sizes and allocations

### Backtesting

1. Navigate to the **Backtesting** page
2. Configure backtest parameters:
   - Strategy (Defensive or Enterprising)
   - Date range (start and end dates)
   - Initial capital
   - Rebalance frequency (monthly, quarterly, or annually)
3. Click **Run Backtest**
4. Analyze performance metrics and charts

## Benjamin Graham's Investment Principles

### Defensive Investor Criteria

The defensive investor seeks safety and freedom from effort:

1. **Adequate Size**: Market capitalization > $2 billion
2. **Strong Financial Condition**: Current Ratio > 2.0
3. **Earnings Stability**: 10 years of positive earnings
4. **Dividend Record**: 20+ years of continuous dividends
5. **Earnings Growth**: 33% growth over 10 years (minimum)
6. **Moderate P/E Ratio**: P/E < 15
7. **Moderate P/B Ratio**: P/B < 1.5
8. **Graham Multiplier**: P/E × P/B ≤ 22.5

### Enterprising Investor Criteria

The enterprising investor is willing to devote time and effort:

1. **Adequate Size**: Market cap > $1 billion
2. **Financial Condition**: Current Ratio > 1.5
3. **Positive Earnings**: Consistent positive earnings
4. **Reasonable P/E**: P/E < 25
5. **Reasonable P/B**: P/B < 2.5
6. **Margin of Safety**: At least 25% discount to intrinsic value

### Graham Number

The Graham Number provides an estimate of a stock's intrinsic value:

```
Graham Number = √(22.5 × EPS × Book Value per Share)
```

A stock trading below its Graham Number may be undervalued.

### Margin of Safety

```
Margin of Safety = (Intrinsic Value - Current Price) / Intrinsic Value
```

Graham recommended a minimum 33% margin of safety for defensive investors.

## Data Sources

Graham Trader integrates multiple financial data sources:

- **yfinance (Yahoo Finance)**: Historical prices, basic fundamentals - **Free**
- **fundamentalanalysis**: Detailed financial statements - **Requires API key**
- **edgartools**: SEC EDGAR filings - **Free**
- **openbb**: Comprehensive financial data - **Free with registration**

### API Keys (Optional)

For enhanced data access, you can set environment variables:

```bash
export FMP_API_KEY="your_financial_modeling_prep_key"
export ALPHA_VANTAGE_API_KEY="your_alpha_vantage_key"
```

The application works without API keys using free data sources.

## Configuration

Edit `config.py` to customize:

- **Screening criteria**: Adjust thresholds for P/E, P/B, etc.
- **Portfolio settings**: Initial capital, position sizes, rebalance frequency
- **Data settings**: Cache expiration, default universe

## Project Structure

```
graham-trader/
├── app.py                      # Main Streamlit application
├── config.py                   # Configuration settings
├── requirements.txt            # Python dependencies
├── README.md                   # This file
└── src/
    ├── data/
    │   └── data_fetcher.py    # Multi-source data integration
    ├── screening/
    │   └── graham_criteria.py # Benjamin Graham screening logic
    ├── portfolio/
    │   ├── portfolio.py       # Portfolio tracking
    │   └── simulator.py       # Backtesting engine
    └── visualization/
        └── charts.py          # Plotly chart builders
```

## Example Workflow

### 1. Screen for Value Stocks

```python
from src.data.data_fetcher import DataFetcher
from src.screening.graham_criteria import GrahamScreener

# Initialize
fetcher = DataFetcher()
screener = GrahamScreener(fetcher)

# Screen S&P 500 for defensive investor stocks
tickers = fetcher.get_sp500_tickers()
results = screener.screen_defensive(tickers)

# Get top candidates
top_stocks = results[results['passes_screening'] == True].head(10)
print(top_stocks[['ticker', 'graham_number', 'margin_of_safety']])
```

### 2. Backtest a Strategy

```python
from src.portfolio.simulator import PortfolioSimulator

# Create simulator
simulator = PortfolioSimulator(fetcher, screener, initial_capital=100000)

# Run backtest
results = simulator.backtest(
    universe=tickers,
    start_date='2020-01-01',
    end_date='2023-12-31',
    strategy='defensive'
)

print(f"Total Return: {results['total_return_pct']:.2f}%")
print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
```

## Performance Considerations

- **Initial screening**: Screening 500 stocks may take 5-10 minutes due to API rate limits
- **Caching**: Enable caching in `config.py` to speed up repeated queries
- **Parallel processing**: Consider implementing concurrent requests for large universes

## Limitations

- Historical data quality depends on data source availability
- Some stocks may have incomplete fundamental data
- Backtesting assumes perfect execution and ignores transaction costs
- Past performance does not guarantee future results

## Contributing

Contributions are welcome! Areas for improvement:

- Additional screening criteria (e.g., NCAV, Magic Formula)
- More data sources integration
- Portfolio optimization algorithms
- Tax-loss harvesting
- Real-time alerts

## Disclaimer

This application is for educational and research purposes only. It does not constitute financial advice. Always conduct your own research and consult with qualified financial advisors before making investment decisions.

## Resources

- **The Intelligent Investor** by Benjamin Graham
- **Security Analysis** by Benjamin Graham and David Dodd
- [Benjamin Graham's investment principles](https://www.investopedia.com/terms/b/bengraham.asp)

## License

MIT License - see LICENSE file for details

## Support

For issues, questions, or contributions, please open an issue on GitHub.

---

*"The intelligent investor is a realist who sells to optimists and buys from pessimists."* - Benjamin Graham
