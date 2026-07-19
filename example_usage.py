"""
Example usage of Graham Trader components

This script demonstrates how to use the Graham Trader components
programmatically without the web interface.
"""
from src.data.data_fetcher import DataFetcher
from src.screening.graham_criteria import GrahamScreener
from src.portfolio.simulator import PortfolioSimulator
import config


def example_1_screen_stocks():
    """Example 1: Screen stocks using defensive investor criteria."""
    print("=" * 60)
    print("Example 1: Screening Stocks (Defensive Investor)")
    print("=" * 60)

    # Initialize data fetcher and screener
    fetcher = DataFetcher()
    screener = GrahamScreener(fetcher, config.GRAHAM_CRITERIA)

    # Define tickers to screen
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'JNJ', 'PG', 'KO', 'WMT', 'JPM', 'V', 'MA']

    print(f"\nScreening {len(tickers)} stocks: {', '.join(tickers)}")
    print("This may take a few moments...\n")

    # Run defensive screening
    results = screener.screen_defensive(tickers)

    # Display results
    if not results.empty:
        print(f"Found {len(results)} stocks with data\n")

        # Filter passing stocks
        passing = results[results['passes_screening'] == True]

        if not passing.empty:
            print(f"✅ {len(passing)} stocks passed the screening:\n")

            for _, stock in passing.iterrows():
                print(f"  {stock['ticker']}:")
                print(f"    Price: ${stock['current_price']:.2f}")
                print(f"    Graham Number: ${stock['graham_number']:.2f}")
                print(f"    Margin of Safety: {stock['margin_of_safety']*100:.1f}%")
                print(f"    P/E: {stock['pe_ratio']:.2f} | P/B: {stock['pb_ratio']:.2f}")
                print(f"    Score: {stock['total_score']}/{stock['max_score']}")
                print()
        else:
            print("❌ No stocks passed the defensive investor criteria")

        # Show top candidates by margin of safety
        print("\nTop 3 candidates by Margin of Safety:")
        top_3 = results.nlargest(3, 'margin_of_safety')
        for idx, stock in enumerate(top_3.itertuples(), 1):
            print(f"  {idx}. {stock.ticker}: {stock.margin_of_safety*100:.1f}% margin")

    else:
        print("No results returned from screening")


def example_2_calculate_intrinsic_value():
    """Example 2: Calculate intrinsic value for specific stocks."""
    print("\n" + "=" * 60)
    print("Example 2: Calculate Intrinsic Value (Graham Number)")
    print("=" * 60)

    fetcher = DataFetcher()

    tickers = ['JNJ', 'PG', 'KO']

    print(f"\nCalculating intrinsic values for: {', '.join(tickers)}\n")

    for ticker in tickers:
        result = fetcher.calculate_graham_number(ticker)

        print(f"{ticker}:")
        print(f"  Current Price: ${result['current_price']:.2f}")
        print(f"  EPS: ${result['eps']:.2f}")
        print(f"  Book Value/Share: ${result['book_value_per_share']:.2f}")
        print(f"  Graham Number: ${result['graham_number']:.2f}")
        print(f"  Margin of Safety: {result['margin_of_safety']*100:.1f}%")

        if result['is_undervalued']:
            print(f"  ✅ UNDERVALUED by {result['margin_of_safety']*100:.1f}%")
        else:
            print(f"  ❌ Not undervalued")
        print()


def example_3_simulate_portfolio():
    """Example 3: Simulate a portfolio with recommended allocations."""
    print("\n" + "=" * 60)
    print("Example 3: Portfolio Simulation")
    print("=" * 60)

    fetcher = DataFetcher()
    screener = GrahamScreener(fetcher, config.GRAHAM_CRITERIA)
    simulator = PortfolioSimulator(
        fetcher,
        screener,
        initial_capital=100000,
        max_positions=10,
        position_size=0.10  # 10% per position
    )

    tickers = ['JNJ', 'PG', 'KO', 'WMT', 'JPM', 'V', 'MA', 'MSFT', 'AAPL', 'GOOGL',
               'UNH', 'HD', 'CVX', 'MRK', 'PFE']

    print(f"\nSimulating portfolio with initial capital: $100,000")
    print(f"Maximum positions: 10")
    print(f"Position size: 10% each\n")

    # Simulate defensive strategy
    recommendations = simulator.simulate_strategy(tickers, strategy='defensive')

    if not recommendations.empty:
        print(f"✅ Portfolio Recommendations ({len(recommendations)} stocks):\n")

        total_investment = 0

        for _, stock in recommendations.iterrows():
            position_value = stock['recommended_shares'] * stock['current_price']
            total_investment += position_value

            print(f"{stock['ticker']}:")
            print(f"  Buy {stock['recommended_shares']} shares at ${stock['current_price']:.2f}")
            print(f"  Position Value: ${position_value:,.2f}")
            print(f"  Graham Number: ${stock['graham_number']:.2f}")
            print(f"  Margin of Safety: {stock['margin_of_safety']*100:.1f}%")
            print()

        print(f"Total Investment: ${total_investment:,.2f}")
        print(f"Cash Remaining: ${100000 - total_investment:,.2f}")
    else:
        print("❌ No stocks met the criteria for portfolio inclusion")


def example_4_compare_strategies():
    """Example 4: Compare defensive vs enterprising strategies."""
    print("\n" + "=" * 60)
    print("Example 4: Compare Defensive vs Enterprising Strategies")
    print("=" * 60)

    fetcher = DataFetcher()
    screener = GrahamScreener(fetcher, config.GRAHAM_CRITERIA)

    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'JNJ', 'PG', 'KO']

    print(f"\nScreening {len(tickers)} stocks with both strategies...\n")

    # Run both screenings
    defensive_results = screener.screen_defensive(tickers)
    enterprising_results = screener.screen_enterprising(tickers)

    # Count passing stocks
    defensive_pass = defensive_results[defensive_results['passes_screening'] == True]
    enterprising_pass = enterprising_results[enterprising_results['passes_screening'] == True]

    print("Defensive Investor Results:")
    print(f"  Stocks analyzed: {len(defensive_results)}")
    print(f"  Stocks passed: {len(defensive_pass)}")
    if not defensive_pass.empty:
        print(f"  Passing stocks: {', '.join(defensive_pass['ticker'].tolist())}")

    print("\nEnterprising Investor Results:")
    print(f"  Stocks analyzed: {len(enterprising_results)}")
    print(f"  Stocks passed: {len(enterprising_pass)}")
    if not enterprising_pass.empty:
        print(f"  Passing stocks: {', '.join(enterprising_pass['ticker'].tolist())}")

    # Find stocks that pass both
    if not defensive_pass.empty and not enterprising_pass.empty:
        both_pass = set(defensive_pass['ticker']).intersection(set(enterprising_pass['ticker']))
        if both_pass:
            print(f"\n⭐ Stocks passing BOTH strategies: {', '.join(both_pass)}")
        else:
            print("\n❌ No stocks passed both strategies")


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("GRAHAM TRADER - Example Usage")
    print("=" * 60)
    print("\nThis script demonstrates the core functionality of Graham Trader")
    print("without using the web interface.\n")

    try:
        # Run examples
        example_1_screen_stocks()
        example_2_calculate_intrinsic_value()
        example_3_simulate_portfolio()
        example_4_compare_strategies()

        print("\n" + "=" * 60)
        print("Examples completed successfully!")
        print("=" * 60)
        print("\nTo use the web interface, run: streamlit run app.py")

    except Exception as e:
        print(f"\n❌ Error running examples: {str(e)}")
        print("\nMake sure you have installed all dependencies:")
        print("  pip install -r requirements.txt")


if __name__ == "__main__":
    main()
