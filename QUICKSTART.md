# Quick Start Guide

Get up and running with Graham Trader in 5 minutes!

## Step 1: Install Dependencies

```bash
cd graham-trader
pip install -r requirements.txt
```

This will install all necessary packages including:
- streamlit (web interface)
- yfinance (stock data)
- plotly (charts)
- pandas, numpy (data analysis)
- and more

## Step 2: Launch the Application

```bash
streamlit run app.py
```

The app will automatically open in your browser at `http://localhost:8501`

## Step 3: Try the Features

### Stock Screener

1. Go to **Stock Screener** in the sidebar
2. Select "Defensive Investor" strategy
3. Choose "Custom Tickers"
4. Enter: `JNJ, PG, KO, WMT, JPM`
5. Click **Run Screening**
6. Review stocks that pass Graham's criteria

### Portfolio Simulator

1. Go to **Portfolio Simulator**
2. Set initial capital: `$100,000`
3. Enter some blue-chip tickers: `JNJ, PG, KO, WMT, V, MA`
4. Click **Simulate Portfolio**
5. See recommended positions and allocations

### Backtesting

1. Go to **Backtesting**
2. Set date range: Last 2-3 years
3. Choose "Defensive Investor" strategy
4. Click **Run Backtest** (this will take a few minutes)
5. Review performance metrics and charts

## Alternative: Run Examples Programmatically

Instead of using the web interface, you can run the example script:

```bash
python example_usage.py
```

This will demonstrate:
- Stock screening
- Intrinsic value calculation
- Portfolio simulation
- Strategy comparison

## Troubleshooting

### Installation Issues

If you encounter issues installing dependencies:

```bash
# Create a virtual environment first
python -m venv venv

# Activate it
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate  # On Windows

# Then install
pip install -r requirements.txt
```

### Data Issues

If you get API rate limit errors:
- Wait a few minutes between requests
- Enable caching in `config.py`
- Use smaller ticker lists for testing

### Import Errors

Make sure you're in the `graham-trader` directory when running commands.

## Next Steps

1. **Read the documentation**: Check `README.md` for detailed information
2. **Customize criteria**: Edit `config.py` to adjust screening thresholds
3. **Try different strategies**: Compare Defensive vs Enterprising approaches
4. **Analyze results**: Use the visualization tools to understand performance

## Tips for Best Results

1. **Start small**: Test with 5-10 tickers first
2. **Use quality stocks**: Blue-chip stocks often have better data
3. **Be patient**: Screening many stocks takes time
4. **Understand the criteria**: Read about Benjamin Graham's principles
5. **Combine strategies**: Use both screening and backtesting

## Getting Help

- Check `README.md` for comprehensive documentation
- Review `example_usage.py` for code examples
- Read about Benjamin Graham's principles in the "About" page

## Quick Reference: Command Cheat Sheet

```bash
# Install dependencies
pip install -r requirements.txt

# Run web app
streamlit run app.py

# Run examples
python example_usage.py

# Activate virtual environment (if using one)
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows
```

---

Happy investing! Remember: *"Price is what you pay. Value is what you get."* - Benjamin Graham
