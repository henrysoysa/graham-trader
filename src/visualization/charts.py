"""
Chart and visualization builder using Plotly.
"""
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from typing import Optional


class ChartBuilder:
    """
    Creates interactive charts for portfolio analysis and stock screening.
    """

    @staticmethod
    def create_portfolio_value_chart(history_df: pd.DataFrame) -> go.Figure:
        """
        Create portfolio value over time chart.

        Args:
            history_df: DataFrame with portfolio history

        Returns:
            Plotly figure
        """
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=history_df['date'],
            y=history_df['total_value'],
            mode='lines',
            name='Total Value',
            line=dict(color='#2E86AB', width=3),
            fill='tozeroy',
            fillcolor='rgba(46, 134, 171, 0.1)'
        ))

        fig.update_layout(
            title='Portfolio Value Over Time',
            xaxis_title='Date',
            yaxis_title='Portfolio Value ($)',
            hovermode='x unified',
            template='plotly_white',
            height=500
        )

        return fig

    @staticmethod
    def create_holdings_pie_chart(holdings_df: pd.DataFrame) -> go.Figure:
        """
        Create pie chart of current holdings allocation.

        Args:
            holdings_df: DataFrame with holdings information

        Returns:
            Plotly figure
        """
        if holdings_df.empty:
            fig = go.Figure()
            fig.add_annotation(
                text="No holdings to display",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=16)
            )
            return fig

        fig = px.pie(
            holdings_df,
            values='market_value',
            names='ticker',
            title='Portfolio Allocation',
            hole=0.4
        )

        fig.update_traces(
            textposition='inside',
            textinfo='percent+label'
        )

        fig.update_layout(
            template='plotly_white',
            height=500
        )

        return fig

    @staticmethod
    def create_returns_distribution_chart(history_df: pd.DataFrame) -> go.Figure:
        """
        Create histogram of returns distribution.

        Args:
            history_df: DataFrame with portfolio history

        Returns:
            Plotly figure
        """
        if 'returns' not in history_df.columns or history_df['returns'].isna().all():
            returns = history_df['total_value'].pct_change()
        else:
            returns = history_df['returns']

        returns = returns.dropna()

        fig = go.Figure()

        fig.add_trace(go.Histogram(
            x=returns * 100,
            nbinsx=30,
            name='Returns',
            marker=dict(color='#2E86AB')
        ))

        fig.update_layout(
            title='Returns Distribution',
            xaxis_title='Returns (%)',
            yaxis_title='Frequency',
            template='plotly_white',
            height=400
        )

        return fig

    @staticmethod
    def create_drawdown_chart(history_df: pd.DataFrame) -> go.Figure:
        """
        Create drawdown chart.

        Args:
            history_df: DataFrame with portfolio history

        Returns:
            Plotly figure
        """
        # Calculate drawdown
        values = history_df['total_value']
        running_max = values.cummax()
        drawdown = (values - running_max) / running_max * 100

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=history_df['date'],
            y=drawdown,
            mode='lines',
            name='Drawdown',
            line=dict(color='#A23B72', width=2),
            fill='tozeroy',
            fillcolor='rgba(162, 59, 114, 0.2)'
        ))

        fig.update_layout(
            title='Portfolio Drawdown',
            xaxis_title='Date',
            yaxis_title='Drawdown (%)',
            hovermode='x unified',
            template='plotly_white',
            height=400
        )

        return fig

    @staticmethod
    def create_screening_results_table(screening_df: pd.DataFrame, max_rows: int = 20) -> go.Figure:
        """
        Create table of screening results.

        Args:
            screening_df: DataFrame with screening results
            max_rows: Maximum number of rows to display

        Returns:
            Plotly figure
        """
        if screening_df.empty:
            fig = go.Figure()
            fig.add_annotation(
                text="No stocks passed screening",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=16)
            )
            return fig

        # Select key columns
        display_columns = [
            'ticker', 'current_price', 'pe_ratio', 'pb_ratio',
            'graham_number', 'margin_of_safety', 'total_score', 'passes_screening'
        ]

        # Filter to available columns
        available_cols = [col for col in display_columns if col in screening_df.columns]
        table_df = screening_df[available_cols].head(max_rows)

        # Format values
        formatted_df = table_df.copy()
        for col in formatted_df.columns:
            if col in ['current_price', 'graham_number']:
                formatted_df[col] = formatted_df[col].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "N/A")
            elif col in ['pe_ratio', 'pb_ratio']:
                formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
            elif col == 'margin_of_safety':
                formatted_df[col] = formatted_df[col].apply(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "N/A")
            elif col == 'passes_screening':
                formatted_df[col] = formatted_df[col].apply(lambda x: "✓" if x else "✗")

        fig = go.Figure(data=[go.Table(
            header=dict(
                values=[col.replace('_', ' ').title() for col in formatted_df.columns],
                fill_color='#2E86AB',
                font=dict(color='white', size=12),
                align='left'
            ),
            cells=dict(
                values=[formatted_df[col] for col in formatted_df.columns],
                fill_color=[['#f0f0f0' if i % 2 == 0 else 'white' for i in range(len(formatted_df))]],
                align='left',
                font=dict(size=11)
            )
        )])

        fig.update_layout(
            title='Stock Screening Results',
            height=min(600, 50 + 30 * len(table_df)),
            margin=dict(l=0, r=0, t=40, b=0)
        )

        return fig

    @staticmethod
    def create_performance_metrics_chart(metrics: dict) -> go.Figure:
        """
        Create visual display of performance metrics.

        Args:
            metrics: Dictionary of performance metrics

        Returns:
            Plotly figure
        """
        # Create metrics cards
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=(
                'Total Return',
                'Annualized Return',
                'Sharpe Ratio',
                'Max Drawdown',
                'Win Rate',
                'Number of Trades'
            ),
            specs=[[{'type': 'indicator'}] * 3] * 2
        )

        # Total Return
        fig.add_trace(go.Indicator(
            mode="number+delta",
            value=metrics.get('total_return_pct', 0),
            number={'suffix': '%', 'font': {'size': 40}},
            delta={'reference': 0},
        ), row=1, col=1)

        # Annualized Return
        fig.add_trace(go.Indicator(
            mode="number",
            value=metrics.get('annualized_return', 0),
            number={'suffix': '%', 'font': {'size': 40}},
        ), row=1, col=2)

        # Sharpe Ratio
        fig.add_trace(go.Indicator(
            mode="number",
            value=metrics.get('sharpe_ratio', 0),
            number={'font': {'size': 40}},
        ), row=1, col=3)

        # Max Drawdown
        fig.add_trace(go.Indicator(
            mode="number",
            value=metrics.get('max_drawdown', 0),
            number={'suffix': '%', 'font': {'size': 40}},
        ), row=2, col=1)

        # Win Rate
        fig.add_trace(go.Indicator(
            mode="number",
            value=metrics.get('win_rate', 0),
            number={'suffix': '%', 'font': {'size': 40}},
        ), row=2, col=2)

        # Number of Trades
        fig.add_trace(go.Indicator(
            mode="number",
            value=metrics.get('num_trades', 0),
            number={'font': {'size': 40}},
        ), row=2, col=3)

        fig.update_layout(
            height=400,
            showlegend=False,
            template='plotly_white'
        )

        return fig

    @staticmethod
    def create_stock_comparison_chart(
        screening_df: pd.DataFrame,
        metric: str = 'margin_of_safety',
        top_n: int = 10
    ) -> go.Figure:
        """
        Create bar chart comparing stocks on a specific metric.

        Args:
            screening_df: DataFrame with screening results
            metric: Metric to compare
            top_n: Number of top stocks to show

        Returns:
            Plotly figure
        """
        if screening_df.empty or metric not in screening_df.columns:
            fig = go.Figure()
            fig.add_annotation(
                text=f"No data available for {metric}",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=16)
            )
            return fig

        # Get top N stocks by metric
        top_stocks = screening_df.nlargest(top_n, metric)

        # Format metric for display
        if metric == 'margin_of_safety':
            y_values = top_stocks[metric] * 100
            y_title = 'Margin of Safety (%)'
        else:
            y_values = top_stocks[metric]
            y_title = metric.replace('_', ' ').title()

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=top_stocks['ticker'],
            y=y_values,
            marker=dict(
                color=y_values,
                colorscale='Viridis',
                showscale=True
            ),
            text=[f"{val:.1f}" for val in y_values],
            textposition='auto',
        ))

        fig.update_layout(
            title=f'Top {top_n} Stocks by {y_title}',
            xaxis_title='Ticker',
            yaxis_title=y_title,
            template='plotly_white',
            height=500
        )

        return fig
