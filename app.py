"""
Graham Trader - Benjamin Graham Value Investing Application
Main Streamlit Dashboard
"""
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta

from src.data.data_fetcher import DataFetcher
from src.screening.graham_criteria import GrahamScreener
from src.portfolio.simulator import PortfolioSimulator
from src.visualization.charts import ChartBuilder
from src.universe import (
    UNIVERSE_OPTIONS,
    UNIVERSE_MAPPING,
    UNIVERSE_HELP,
    resolve_index_key,
)
import config

# Page configuration
st.set_page_config(
    page_title="Graham Trader",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #2E86AB;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #555;
        text-align: center;
        padding-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
if 'data_fetcher' not in st.session_state:
    st.session_state.data_fetcher = DataFetcher()

if 'screener' not in st.session_state:
    st.session_state.screener = GrahamScreener(
        st.session_state.data_fetcher,
        config.GRAHAM_CRITERIA
    )

if 'simulator' not in st.session_state:
    st.session_state.simulator = PortfolioSimulator(
        st.session_state.data_fetcher,
        st.session_state.screener,
        initial_capital=config.PORTFOLIO_SETTINGS['initial_capital'],
        max_positions=20,
        position_size=config.PORTFOLIO_SETTINGS['max_position_size'],
        rebalance_frequency=config.PORTFOLIO_SETTINGS['rebalance_frequency']
    )

if 'screening_results' not in st.session_state:
    st.session_state.screening_results = None

if 'backtest_results' not in st.session_state:
    st.session_state.backtest_results = None


# ------------------------------------------------------------------
# Shared stock-universe selector
# ------------------------------------------------------------------
# The universe options/mapping live in the pure ``src.universe`` module (so
# they can be unit-tested without Streamlit). This wrapper renders the picker
# and is shared by EVERY page (Screener, Portfolio Simulator, Backtesting) so
# each one offers — and actually honours — the same markets. Previously the
# Simulator and Backtesting pages hardcoded the S&P 500.


def select_universe(widget_key, default_custom="AAPL, MSFT, GOOGL, JNJ, PG"):
    """Render the shared universe selector.

    Returns a tuple ``(universe_option, tickers, index_key)`` where:
      * ``tickers`` is a concrete list when the user typed custom tickers,
        an empty list when a separator row is selected, or ``None`` when a
        market index was chosen (fetch it via ``index_key``).
      * ``index_key`` is the data_fetcher key for the chosen index, or
        ``None`` for custom/separator selections.
    """
    universe_option = st.selectbox(
        "Stock Universe",
        UNIVERSE_OPTIONS,
        key=f"universe_{widget_key}",
        help=UNIVERSE_HELP,
    )

    if universe_option == "Custom Tickers":
        custom_tickers_input = st.text_input(
            "Enter ticker symbols (comma-separated)",
            default_custom,
            key=f"custom_{widget_key}",
            help="Example: AAPL, MSFT, GOOGL",
        )
        tickers = [t.strip().upper() for t in custom_tickers_input.split(',') if t.strip()]
        return universe_option, tickers, None

    if universe_option.startswith("---"):
        return universe_option, [], None

    # A real market index was selected — resolve it explicitly. If a label is
    # ever missing from the mapping we surface it rather than silently using US.
    index_key = UNIVERSE_MAPPING.get(universe_option)
    if index_key is None:
        st.error(f"Internal error: no index mapping for '{universe_option}'.")
    return universe_option, None, index_key


def main():
    """Main application function."""

    # Header
    st.markdown('<div class="main-header">📈 Graham Trader</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Benjamin Graham Value Investing Application</div>', unsafe_allow_html=True)

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select a page:",
        ["Stock Screener", "Portfolio Simulator", "Backtesting", "About"]
    )

    if page == "Stock Screener":
        show_stock_screener()
    elif page == "Portfolio Simulator":
        show_portfolio_simulator()
    elif page == "Backtesting":
        show_backtesting()
    elif page == "About":
        show_about()


def show_stock_screener():
    """Stock screening page."""
    st.header("🔍 Stock Screener")
    st.write("Screen stocks using Benjamin Graham's value investing criteria")

    # Screening options
    col1, col2, col3 = st.columns(3)

    with col1:
        strategy = st.selectbox(
            "Investment Strategy",
            ["Defensive Investor", "Enterprising Investor"],
            help="Defensive: Conservative criteria for passive investors. Enterprising: More relaxed for active investors."
        )

    with col2:
        universe_option, tickers, index_key = select_universe("screener")

    with col3:
        max_stocks = st.number_input(
            "Max Results",
            min_value=5,
            max_value=50,
            value=20,
            help="Maximum number of stocks to display"
        )

    if universe_option.startswith("---"):
        st.info("Please select a specific market index from the dropdown.")

    # Screen button
    if st.button("🚀 Run Screening", type="primary"):
        if universe_option.startswith("---"):
            st.error("Please select a specific market index from the dropdown, not a separator.")
        elif tickers is not None and len(tickers) == 0:
            st.error("Please enter at least one ticker symbol.")
        else:
            with st.spinner("Screening stocks... This may take a few minutes."):
                # Get tickers
                if tickers is None:
                    with st.status(f"Fetching {universe_option} tickers..."):
                        tickers = st.session_state.data_fetcher.get_index_tickers(index_key)
                        st.write(f"Found {len(tickers)} stocks in {universe_option}")

                # Run screening
                with st.status(f"Screening {len(tickers)} stocks..."):
                    if strategy == "Defensive Investor":
                        results = st.session_state.screener.screen_defensive(tickers)
                    else:
                        results = st.session_state.screener.screen_enterprising(tickers)

                    st.session_state.screening_results = results

    # Display results
    if st.session_state.screening_results is not None and not st.session_state.screening_results.empty:
        results = st.session_state.screening_results
        passing_stocks = results[results['passes_screening'] == True]

        st.success(f"✅ Screening complete! Analyzed {len(results)} stocks.")

        # Show summary
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Stocks Analyzed", len(results))
        with col2:
            st.metric("Passing Graham Criteria", len(passing_stocks))
        with col3:
            if len(results) > 0:
                st.metric("Pass Rate", f"{len(passing_stocks)/len(results)*100:.1f}%")

        # Allow user to filter view
        view_option = st.radio(
            "View:",
            ["All Stocks (Ranked by Score)", "Only Passing Stocks"],
            horizontal=True
        )

        if view_option == "Only Passing Stocks":
            display_stocks = passing_stocks
        else:
            display_stocks = results

        st.subheader(f"📊 {view_option}")

        if not display_stocks.empty:
            # Display top stocks
            display_df = display_stocks.head(max_stocks).copy()

            # Format for display
            display_columns = {
                'ticker': 'Ticker',
                'current_price': 'Price',
                'pe_ratio': 'P/E',
                'pb_ratio': 'P/B',
                'current_ratio': 'Current Ratio',
                'graham_number': 'Graham Number',
                'margin_of_safety': 'Margin of Safety',
                'total_score': 'Score',
                'pass_percentage': 'Pass %'
            }

            # Select and rename columns
            available_cols = [col for col in display_columns.keys() if col in display_df.columns]
            display_df = display_df[available_cols].copy()

            # Format numeric columns
            for col in ['current_price', 'graham_number']:
                if col in display_df.columns:
                    display_df[col] = display_df[col].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "N/A")

            for col in ['pe_ratio', 'pb_ratio', 'current_ratio']:
                if col in display_df.columns:
                    display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")

            if 'margin_of_safety' in display_df.columns:
                display_df['margin_of_safety'] = display_df['margin_of_safety'].apply(
                    lambda x: f"{x*100:.1f}%" if pd.notna(x) else "N/A"
                )

            if 'pass_percentage' in display_df.columns:
                display_df['pass_percentage'] = display_df['pass_percentage'].apply(
                    lambda x: f"{x:.0f}%" if pd.notna(x) else "N/A"
                )

            # Rename columns
            display_df.rename(columns=display_columns, inplace=True)

            # Display table
            st.dataframe(display_df, use_container_width=True, hide_index=True)

            # Visualizations
            st.subheader("📈 Visualizations")

            col1, col2 = st.columns(2)

            with col1:
                # Margin of Safety comparison
                fig1 = ChartBuilder.create_stock_comparison_chart(
                    display_stocks,
                    metric='margin_of_safety',
                    top_n=min(10, len(display_stocks))
                )
                st.plotly_chart(fig1, use_container_width=True)

            with col2:
                # Score comparison
                if 'total_score' in display_stocks.columns:
                    fig2 = ChartBuilder.create_stock_comparison_chart(
                        display_stocks,
                        metric='total_score',
                        top_n=min(10, len(display_stocks))
                    )
                    st.plotly_chart(fig2, use_container_width=True)

        else:
            st.info("No stocks in the selected view. Try switching to 'All Stocks' to see all analyzed stocks.")

    elif st.session_state.screening_results is not None:
        st.warning("No stocks passed the screening criteria.")


def show_portfolio_simulator():
    """Portfolio simulation page."""
    st.header("💼 Portfolio Simulator")
    st.write("Simulate a Graham-based portfolio with current stock picks")

    # Strategy selection
    col1, col2 = st.columns(2)

    with col1:
        strategy = st.selectbox(
            "Strategy",
            ["Defensive Investor", "Enterprising Investor"]
        )

    with col2:
        initial_capital = st.number_input(
            "Initial Capital ($)",
            min_value=10000,
            max_value=10000000,
            value=100000,
            step=10000
        )

    # Universe selection (shared full market picker)
    universe_option, tickers, index_key = select_universe(
        "simulator", default_custom="AAPL, MSFT, GOOGL, JNJ, PG, KO, WMT, JPM"
    )

    if universe_option.startswith("---"):
        st.info("Please select a specific market index from the dropdown.")

    # Simulate button
    if st.button("🎯 Simulate Portfolio", type="primary"):
        if universe_option.startswith("---"):
            st.error("Please select a specific market index, not a separator.")
        elif tickers is not None and len(tickers) == 0:
            st.error("Please enter at least one ticker symbol.")
        else:
          with st.spinner("Running simulation..."):
            # Update simulator capital
            st.session_state.simulator.initial_capital = initial_capital
            st.session_state.simulator.portfolio.initial_capital = initial_capital
            st.session_state.simulator.portfolio.cash = initial_capital

            # Get tickers from the selected market index
            if tickers is None:
                tickers = st.session_state.data_fetcher.get_index_tickers(index_key)

            # Run simulation
            strategy_key = 'defensive' if strategy == "Defensive Investor" else 'enterprising'
            results = st.session_state.simulator.simulate_strategy(tickers, strategy_key)

            # Display results
            if not results.empty:
                st.success(f"✅ Found {len(results)} suitable stocks for your portfolio!")

                st.subheader("📋 Recommended Portfolio")

                # Format for display
                display_df = results[[
                    'ticker', 'current_price', 'recommended_shares',
                    'position_value', 'graham_number', 'margin_of_safety'
                ]].copy()

                display_df['current_price'] = display_df['current_price'].apply(lambda x: f"${x:.2f}")
                display_df['position_value'] = display_df['position_value'].apply(lambda x: f"${x:,.2f}")
                display_df['graham_number'] = display_df['graham_number'].apply(lambda x: f"${x:.2f}")
                display_df['margin_of_safety'] = display_df['margin_of_safety'].apply(lambda x: f"{x*100:.1f}%")

                display_df.columns = [
                    'Ticker', 'Price', 'Shares', 'Position Value',
                    'Graham Number', 'Margin of Safety'
                ]

                st.dataframe(display_df, use_container_width=True, hide_index=True)

                # Portfolio summary
                total_investment = results['position_value'].sum()
                num_positions = len(results)

                col1, col2, col3 = st.columns(3)
                col1.metric("Total Investment", f"${total_investment:,.2f}")
                col2.metric("Number of Positions", num_positions)
                col3.metric("Cash Remaining", f"${initial_capital - total_investment:,.2f}")

            else:
                st.warning("No stocks met the criteria for this strategy.")


def show_backtesting():
    """Backtesting page."""
    st.header("⏮️ Backtesting")
    st.write("Test the Graham strategy on historical data")

    # Backtest parameters
    col1, col2, col3 = st.columns(3)

    with col1:
        strategy = st.selectbox(
            "Strategy",
            ["Defensive Investor", "Enterprising Investor"],
            key="backtest_strategy"
        )

    with col2:
        start_date = st.date_input(
            "Start Date",
            value=datetime.now() - timedelta(days=365*3),
            max_value=datetime.now()
        )

    with col3:
        end_date = st.date_input(
            "End Date",
            value=datetime.now(),
            max_value=datetime.now()
        )

    # Additional parameters
    col1, col2 = st.columns(2)

    with col1:
        initial_capital = st.number_input(
            "Initial Capital ($)",
            min_value=10000,
            max_value=10000000,
            value=100000,
            step=10000,
            key="backtest_capital"
        )

    with col2:
        rebalance_freq = st.selectbox(
            "Rebalance Frequency",
            ["quarterly", "monthly", "annually"]
        )

    # Universe selection (shared full market picker)
    universe_option, tickers, index_key = select_universe("backtest")
    max_backtest_stocks = st.number_input(
        "Max stocks to screen (for performance)",
        min_value=10,
        max_value=200,
        value=50,
        step=10,
        help="Backtesting fetches data for every stock at each rebalance date, so large universes can be slow."
    )

    if universe_option.startswith("---"):
        st.info("Please select a specific market index from the dropdown.")

    # Run backtest
    if st.button("🚀 Run Backtest", type="primary"):
        if universe_option.startswith("---"):
            st.error("Please select a specific market index, not a separator.")
        elif tickers is not None and len(tickers) == 0:
            st.error("Please enter at least one ticker symbol.")
        else:
          with st.spinner("Running backtest... This may take several minutes."):
            # Update simulator settings
            st.session_state.simulator.initial_capital = initial_capital
            st.session_state.simulator.rebalance_frequency = rebalance_freq

            # Get tickers from the selected market index (capped for performance)
            if tickers is None:
                tickers = st.session_state.data_fetcher.get_index_tickers(index_key)
            tickers = tickers[:int(max_backtest_stocks)]

            # Run backtest
            strategy_key = 'defensive' if strategy == "Defensive Investor" else 'enterprising'

            try:
                results = st.session_state.simulator.backtest(
                    universe=tickers,
                    start_date=start_date.strftime('%Y-%m-%d'),
                    end_date=end_date.strftime('%Y-%m-%d'),
                    strategy=strategy_key
                )

                st.session_state.backtest_results = results

            except Exception as e:
                st.error(f"Error running backtest: {str(e)}")
                st.session_state.backtest_results = None

    # Display backtest results
    if st.session_state.backtest_results:
        results = st.session_state.backtest_results

        st.success("✅ Backtest complete!")

        # Performance metrics
        st.subheader("📊 Performance Metrics")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Total Return",
                f"{results['total_return_pct']:.2f}%",
                f"${results['total_return']:,.2f}"
            )

        with col2:
            st.metric(
                "Annualized Return",
                f"{results['annualized_return']:.2f}%"
            )

        with col3:
            st.metric(
                "Sharpe Ratio",
                f"{results['sharpe_ratio']:.2f}"
            )

        with col4:
            st.metric(
                "Max Drawdown",
                f"{results['max_drawdown']:.2f}%"
            )

        # Additional metrics
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Number of Trades", results['num_trades'])

        with col2:
            st.metric("Win Rate", f"{results['win_rate']:.1f}%")

        with col3:
            st.metric("Final Value", f"${results['final_value']:,.2f}")

        # Charts
        st.subheader("📈 Portfolio Performance")

        if 'portfolio_history' in results and not results['portfolio_history'].empty:
            # Portfolio value over time
            fig1 = ChartBuilder.create_portfolio_value_chart(results['portfolio_history'])
            st.plotly_chart(fig1, use_container_width=True)

            # Drawdown chart
            fig2 = ChartBuilder.create_drawdown_chart(results['portfolio_history'])
            st.plotly_chart(fig2, use_container_width=True)

        # Transactions
        if 'transactions' in results and not results['transactions'].empty:
            st.subheader("📝 Transaction History")
            st.dataframe(
                results['transactions'].tail(20),
                use_container_width=True,
                hide_index=True
            )


def show_about():
    """About page."""
    st.header("ℹ️ About Graham Trader")

    st.markdown("""
    ### Benjamin Graham Value Investing

    **Graham Trader** is a comprehensive application for implementing Benjamin Graham's value investing principles.
    Benjamin Graham, known as the "father of value investing," developed systematic approaches for identifying
    undervalued stocks with strong fundamentals.

    #### Key Features:

    1. **Stock Screening**
       - Defensive Investor criteria for conservative, passive investors
       - Enterprising Investor criteria for active investors
       - Graham Number calculation for intrinsic value
       - Margin of Safety analysis

    2. **Portfolio Simulation**
       - Simulate portfolios based on screening results
       - Position sizing and allocation
       - Real-time stock data integration

    3. **Backtesting**
       - Test strategies on historical data
       - Performance metrics (returns, Sharpe ratio, drawdown)
       - Transaction history and win rate analysis

    #### Benjamin Graham's Investment Criteria:

    **Defensive Investor:**
    - Adequate size (market cap > $2B)
    - Strong financial condition (Current Ratio > 2.0)
    - Earnings stability (10 years positive earnings)
    - Dividend record (20+ years continuous)
    - Moderate P/E ratio (< 15)
    - Moderate P/B ratio (< 1.5)
    - P/E × P/B ≤ 22.5

    **Enterprising Investor:**
    - More relaxed criteria for active investors
    - Focus on margin of safety
    - Willingness to do deeper analysis

    #### Data Sources:

    This application integrates multiple data sources:
    - **Yahoo Finance (yfinance)**: Historical prices and basic fundamentals
    - **FundamentalAnalysis**: Detailed financial statements
    - **edgartools**: SEC EDGAR filings
    - **OpenBB**: Comprehensive financial data

    #### Technology Stack:

    - **Python**: Core language
    - **Streamlit**: Web dashboard
    - **Plotly**: Interactive visualizations
    - **Pandas/NumPy**: Data analysis
    - **Multiple APIs**: Financial data

    #### Getting Started:

    1. Navigate to **Stock Screener** to find value stocks
    2. Use **Portfolio Simulator** to build a portfolio
    3. Run **Backtesting** to validate the strategy

    ---

    *"Price is what you pay. Value is what you get."* - Benjamin Graham
    """)

    st.info("💡 **Tip**: Start with the Defensive Investor strategy if you're new to value investing!")


if __name__ == "__main__":
    main()
