"""
Stock Analysis Dashboard - Streamlit Application
================================================
Interactive financial analysis application for stocks from Yahoo Finance.
Includes price analysis, risk metrics, correlations, and comparisons.

Author: Financial Analysis Course
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="Stock Analysis Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CUSTOM CSS STYLING
# =============================================================================
st.markdown("""
<style>
    /* Main header styling */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A8A;
        text-align: center;
        padding: 1rem 0;
        border-bottom: 3px solid #D91023;
        margin-bottom: 2rem;
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    /* Sidebar styling */
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }

    /* Info boxes */
    .info-box {
        background-color: #e7f3ff;
        border-left: 5px solid #2196F3;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 5px 5px 0;
    }

    /* Success boxes */
    .success-box {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 5px 5px 0;
    }

    /* Warning boxes */
    .warning-box {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 5px 5px 0;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 10px 20px;
        background-color: #f0f2f6;
        border-radius: 5px 5px 0 0;
    }

    .stTabs [aria-selected="true"] {
        background-color: #1E3A8A;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

@st.cache_data(ttl=3600)  # Cache for 1 hour
def download_stock_data(tickers: list, start_date: datetime, end_date: datetime) -> tuple:
    """Download stock data from Yahoo Finance with caching."""
    try:
        data = yf.download(tickers, start=start_date, end=end_date, progress=False)

        if data.empty:
            return None, None, "No data available for the selected tickers and date range."

        # Handle MultiIndex columns
        if isinstance(data.columns, pd.MultiIndex):
            prices = data['Close'].copy()
            volume = data['Volume'].copy()
        else:
            prices = data[['Close']].copy()
            volume = data[['Volume']].copy()
            prices.columns = tickers
            volume.columns = tickers

        return prices, volume, None
    except Exception as e:
        return None, None, str(e)


def calculate_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Calculate daily returns from prices."""
    return prices.pct_change().dropna()


def calculate_cumulative_returns(returns: pd.DataFrame) -> pd.DataFrame:
    """Calculate cumulative returns."""
    return (1 + returns).cumprod() - 1


def normalize_prices(prices: pd.DataFrame, base: float = 100) -> pd.DataFrame:
    """Normalize prices to a base value."""
    return prices / prices.iloc[0] * base


def calculate_risk_metrics(returns: pd.Series, risk_free_rate: float = 0.05) -> dict:
    """Calculate comprehensive risk metrics for a return series."""
    trading_days = 252

    # Annualized return and volatility
    annual_return = returns.mean() * trading_days
    annual_volatility = returns.std() * np.sqrt(trading_days)

    # Sharpe Ratio
    sharpe = (annual_return - risk_free_rate) / annual_volatility if annual_volatility > 0 else 0

    # Sortino Ratio
    negative_returns = returns[returns < 0]
    downside_std = negative_returns.std() * np.sqrt(trading_days) if len(negative_returns) > 0 else 0
    sortino = (annual_return - risk_free_rate) / downside_std if downside_std > 0 else 0

    # Maximum Drawdown
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()

    # Value at Risk (95%)
    var_95 = returns.quantile(0.05)

    # Conditional VaR (Expected Shortfall)
    cvar_95 = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else var_95

    return {
        'Annual Return': annual_return,
        'Annual Volatility': annual_volatility,
        'Sharpe Ratio': sharpe,
        'Sortino Ratio': sortino,
        'Max Drawdown': max_drawdown,
        'VaR (95%)': var_95,
        'CVaR (95%)': cvar_95,
        'Skewness': returns.skew(),
        'Kurtosis': returns.kurtosis()
    }


def calculate_drawdown(returns: pd.Series) -> pd.Series:
    """Calculate drawdown series from returns."""
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max * 100
    return drawdown


# =============================================================================
# PRESET STOCK LISTS
# =============================================================================
STOCK_PRESETS = {
    "Peruvian ADRs": {
        "tickers": ["BAP", "SCCO", "BVN", "IFS"],
        "names": {
            "BAP": "Credicorp",
            "SCCO": "Southern Copper",
            "BVN": "Buenaventura",
            "IFS": "Intercorp Financial"
        }
    },
    "US Tech Giants": {
        "tickers": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA"],
        "names": {
            "AAPL": "Apple",
            "MSFT": "Microsoft",
            "GOOGL": "Alphabet",
            "AMZN": "Amazon",
            "META": "Meta",
            "NVDA": "NVIDIA"
        }
    },
    "US Banks": {
        "tickers": ["JPM", "BAC", "WFC", "GS", "MS"],
        "names": {
            "JPM": "JPMorgan Chase",
            "BAC": "Bank of America",
            "WFC": "Wells Fargo",
            "GS": "Goldman Sachs",
            "MS": "Morgan Stanley"
        }
    },
    "Commodities ETFs": {
        "tickers": ["GLD", "SLV", "USO", "COPX"],
        "names": {
            "GLD": "Gold ETF",
            "SLV": "Silver ETF",
            "USO": "Oil ETF",
            "COPX": "Copper Miners ETF"
        }
    },
    "Latin America": {
        "tickers": ["BAP", "BSBR", "ITUB", "VALE", "AMX"],
        "names": {
            "BAP": "Credicorp (Peru)",
            "BSBR": "Banco Santander Brasil",
            "ITUB": "Itau Unibanco (Brazil)",
            "VALE": "Vale (Brazil)",
            "AMX": "America Movil (Mexico)"
        }
    },
    "Global Indices ETFs": {
        "tickers": ["SPY", "EEM", "EFA", "VWO"],
        "names": {
            "SPY": "S&P 500 ETF",
            "EEM": "Emerging Markets ETF",
            "EFA": "EAFE ETF",
            "VWO": "Vanguard EM ETF"
        }
    }
}


# =============================================================================
# SIDEBAR CONFIGURATION
# =============================================================================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/stocks.png", width=80)
    st.title("Configuration")

    st.markdown("---")

    # Stock Selection Method
    selection_method = st.radio(
        "Stock Selection Method",
        ["Preset Lists", "Custom Tickers"],
        help="Choose preset lists or enter your own stock tickers"
    )

    if selection_method == "Preset Lists":
        preset_choice = st.selectbox(
            "Select Preset",
            list(STOCK_PRESETS.keys())
        )
        selected_tickers = STOCK_PRESETS[preset_choice]["tickers"]
        ticker_names = STOCK_PRESETS[preset_choice]["names"]

        st.info(f"**Selected:** {', '.join(selected_tickers)}")

    else:
        custom_input = st.text_area(
            "Enter Tickers",
            value="AAPL, MSFT, GOOGL",
            help="Enter stock tickers separated by commas (e.g., AAPL, MSFT, GOOGL)"
        )
        selected_tickers = [t.strip().upper() for t in custom_input.split(",") if t.strip()]
        ticker_names = {t: t for t in selected_tickers}

    st.markdown("---")

    # Date Range Selection
    st.subheader("Date Range")

    date_preset = st.selectbox(
        "Quick Select",
        ["Custom", "1 Month", "3 Months", "6 Months", "1 Year", "2 Years", "3 Years", "5 Years"]
    )

    if date_preset == "Custom":
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=datetime.now() - timedelta(days=365)
            )
        with col2:
            end_date = st.date_input(
                "End Date",
                value=datetime.now()
            )
    else:
        end_date = datetime.now()
        days_map = {
            "1 Month": 30,
            "3 Months": 90,
            "6 Months": 180,
            "1 Year": 365,
            "2 Years": 730,
            "3 Years": 1095,
            "5 Years": 1825
        }
        start_date = end_date - timedelta(days=days_map[date_preset])

    st.markdown("---")

    # Benchmark Selection
    st.subheader("Benchmark")
    benchmark_options = {
        "S&P 500": "^GSPC",
        "NASDAQ": "^IXIC",
        "Dow Jones": "^DJI",
        "Emerging Markets": "EEM",
        "None": None
    }
    benchmark_choice = st.selectbox(
        "Compare with",
        list(benchmark_options.keys())
    )
    benchmark_ticker = benchmark_options[benchmark_choice]

    st.markdown("---")

    # Risk-free rate
    risk_free_rate = st.slider(
        "Risk-Free Rate (%)",
        min_value=0.0,
        max_value=10.0,
        value=5.0,
        step=0.25,
        help="Annual risk-free rate for Sharpe/Sortino calculations"
    ) / 100

    st.markdown("---")

    # Download button
    analyze_button = st.button("🔍 Analyze Stocks", type="primary", use_container_width=True)


# =============================================================================
# MAIN CONTENT
# =============================================================================

# Header
st.markdown('<h1 class="main-header">📈 Stock Analysis Dashboard</h1>', unsafe_allow_html=True)

# Check if we have tickers
if not selected_tickers:
    st.warning("Please enter at least one stock ticker to analyze.")
    st.stop()

# Download data
with st.spinner("Downloading stock data from Yahoo Finance..."):
    prices, volume, error = download_stock_data(selected_tickers, start_date, end_date)

    # Download benchmark if selected
    benchmark_prices = None
    if benchmark_ticker:
        bench_data, _, _ = download_stock_data([benchmark_ticker], start_date, end_date)
        if bench_data is not None:
            benchmark_prices = bench_data

if error:
    st.error(f"Error downloading data: {error}")
    st.stop()

if prices is None or prices.empty:
    st.error("No data available. Please check your tickers and date range.")
    st.stop()

# Rename columns to company names
if selection_method == "Preset Lists":
    prices.columns = [ticker_names.get(col, col) for col in prices.columns]
    if volume is not None:
        volume.columns = [ticker_names.get(col, col) for col in volume.columns]

# Calculate metrics
returns = calculate_returns(prices)
cumulative_returns = calculate_cumulative_returns(returns)
normalized_prices = normalize_prices(prices)

# =============================================================================
# SUMMARY METRICS ROW
# =============================================================================
st.markdown("### 📊 Quick Summary")

# Create metric columns
metric_cols = st.columns(len(prices.columns))

for idx, stock in enumerate(prices.columns):
    with metric_cols[idx]:
        current_price = prices[stock].iloc[-1]
        start_price = prices[stock].iloc[0]
        total_return = (current_price / start_price - 1) * 100
        daily_return = returns[stock].iloc[-1] * 100 if len(returns) > 0 else 0

        st.metric(
            label=stock,
            value=f"${current_price:,.2f}",
            delta=f"{total_return:+.1f}% total"
        )

st.markdown("---")

# =============================================================================
# MAIN TABS
# =============================================================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Price Analysis",
    "📊 Returns & Performance",
    "🔗 Correlation",
    "⚠️ Risk Metrics",
    "📉 Drawdown",
    "📋 Dashboard"
])

# =============================================================================
# TAB 1: PRICE ANALYSIS
# =============================================================================
with tab1:
    st.subheader("Price Evolution")

    col1, col2 = st.columns([3, 1])

    with col1:
        # Price chart
        fig_prices = go.Figure()

        colors = px.colors.qualitative.Set2
        for idx, stock in enumerate(prices.columns):
            fig_prices.add_trace(go.Scatter(
                x=prices.index,
                y=prices[stock],
                name=stock,
                line=dict(color=colors[idx % len(colors)], width=2),
                hovertemplate=f"{stock}<br>Date: %{{x}}<br>Price: $%{{y:.2f}}<extra></extra>"
            ))

        fig_prices.update_layout(
            title="Stock Prices Over Time",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=500
        )

        st.plotly_chart(fig_prices, use_container_width=True)

    with col2:
        st.markdown("**Price Statistics**")

        for stock in prices.columns:
            with st.expander(stock, expanded=False):
                st.write(f"**Current:** ${prices[stock].iloc[-1]:,.2f}")
                st.write(f"**High:** ${prices[stock].max():,.2f}")
                st.write(f"**Low:** ${prices[stock].min():,.2f}")
                st.write(f"**Mean:** ${prices[stock].mean():,.2f}")

    st.markdown("---")

    # Normalized prices
    st.subheader("Normalized Performance (Base = 100)")

    fig_norm = go.Figure()

    for idx, stock in enumerate(normalized_prices.columns):
        fig_norm.add_trace(go.Scatter(
            x=normalized_prices.index,
            y=normalized_prices[stock],
            name=stock,
            line=dict(color=colors[idx % len(colors)], width=2)
        ))

    # Add benchmark if available
    if benchmark_prices is not None and not benchmark_prices.empty:
        bench_norm = normalize_prices(benchmark_prices)
        fig_norm.add_trace(go.Scatter(
            x=bench_norm.index,
            y=bench_norm.iloc[:, 0],
            name=benchmark_choice,
            line=dict(color='gray', width=2, dash='dash')
        ))

    fig_norm.add_hline(y=100, line_dash="dash", line_color="gray", opacity=0.5)

    fig_norm.update_layout(
        title="Normalized Performance Comparison",
        xaxis_title="Date",
        yaxis_title="Normalized Value",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=500
    )

    st.plotly_chart(fig_norm, use_container_width=True)

# =============================================================================
# TAB 2: RETURNS & PERFORMANCE
# =============================================================================
with tab2:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Total Returns")

        total_returns = ((prices.iloc[-1] - prices.iloc[0]) / prices.iloc[0] * 100).sort_values(ascending=True)

        fig_returns = go.Figure()

        colors_bar = ['#10B981' if x > 0 else '#EF4444' for x in total_returns.values]

        fig_returns.add_trace(go.Bar(
            y=total_returns.index,
            x=total_returns.values,
            orientation='h',
            marker_color=colors_bar,
            text=[f"{x:.1f}%" for x in total_returns.values],
            textposition='outside'
        ))

        fig_returns.add_vline(x=0, line_color="black", line_width=1)

        fig_returns.update_layout(
            title="Total Return by Stock",
            xaxis_title="Return (%)",
            yaxis_title="",
            height=400,
            showlegend=False
        )

        st.plotly_chart(fig_returns, use_container_width=True)

    with col2:
        st.subheader("Cumulative Returns")

        fig_cumret = go.Figure()

        for idx, stock in enumerate(cumulative_returns.columns):
            fig_cumret.add_trace(go.Scatter(
                x=cumulative_returns.index,
                y=cumulative_returns[stock] * 100,
                name=stock,
                fill='tonexty' if idx > 0 else 'tozeroy',
                line=dict(color=colors[idx % len(colors)], width=2)
            ))

        fig_cumret.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

        fig_cumret.update_layout(
            title="Cumulative Returns Over Time",
            xaxis_title="Date",
            yaxis_title="Cumulative Return (%)",
            hovermode="x unified",
            height=400
        )

        st.plotly_chart(fig_cumret, use_container_width=True)

    st.markdown("---")

    # Return distribution
    st.subheader("Return Distribution")

    fig_dist = make_subplots(rows=1, cols=2, subplot_titles=("Daily Return Distribution", "Box Plot"))

    for idx, stock in enumerate(returns.columns):
        fig_dist.add_trace(
            go.Histogram(x=returns[stock] * 100, name=stock, opacity=0.7,
                        marker_color=colors[idx % len(colors)]),
            row=1, col=1
        )

        fig_dist.add_trace(
            go.Box(y=returns[stock] * 100, name=stock,
                  marker_color=colors[idx % len(colors)]),
            row=1, col=2
        )

    fig_dist.update_layout(
        height=400,
        barmode='overlay',
        showlegend=True
    )
    fig_dist.update_xaxes(title_text="Daily Return (%)", row=1, col=1)
    fig_dist.update_yaxes(title_text="Frequency", row=1, col=1)
    fig_dist.update_yaxes(title_text="Daily Return (%)", row=1, col=2)

    st.plotly_chart(fig_dist, use_container_width=True)

# =============================================================================
# TAB 3: CORRELATION
# =============================================================================
with tab3:
    st.subheader("Correlation Analysis")

    if len(prices.columns) > 1:
        corr_matrix = returns.corr()

        col1, col2 = st.columns([2, 1])

        with col1:
            fig_corr = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.index,
                colorscale='RdYlGn',
                zmid=0,
                text=np.round(corr_matrix.values, 2),
                texttemplate="%{text}",
                textfont={"size": 12},
                hovertemplate="Correlation between %{x} and %{y}: %{z:.2f}<extra></extra>"
            ))

            fig_corr.update_layout(
                title="Correlation Matrix (Daily Returns)",
                height=500
            )

            st.plotly_chart(fig_corr, use_container_width=True)

        with col2:
            st.markdown("**Correlation Interpretation**")
            st.markdown("""
            - **+1.0**: Perfect positive correlation
            - **+0.5 to +1.0**: Strong positive
            - **0 to +0.5**: Weak positive
            - **0**: No correlation
            - **-0.5 to 0**: Weak negative
            - **-1.0 to -0.5**: Strong negative
            - **-1.0**: Perfect negative correlation
            """)

            st.markdown("---")

            st.markdown("**Key Pairs**")

            # Find highest and lowest correlations
            corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_pairs.append({
                        'Pair': f"{corr_matrix.columns[i]} - {corr_matrix.columns[j]}",
                        'Correlation': corr_matrix.iloc[i, j]
                    })

            if corr_pairs:
                corr_df = pd.DataFrame(corr_pairs).sort_values('Correlation', ascending=False)
                st.dataframe(corr_df, hide_index=True)

        # Rolling correlation (if 2 stocks)
        if len(prices.columns) >= 2:
            st.markdown("---")
            st.subheader("Rolling Correlation (30-day window)")

            stock_pairs = [(prices.columns[i], prices.columns[j])
                          for i in range(len(prices.columns))
                          for j in range(i+1, len(prices.columns))]

            if len(stock_pairs) <= 6:  # Only show if not too many pairs
                fig_rolling = go.Figure()

                for pair in stock_pairs:
                    rolling_corr = returns[pair[0]].rolling(30).corr(returns[pair[1]])
                    fig_rolling.add_trace(go.Scatter(
                        x=rolling_corr.index,
                        y=rolling_corr,
                        name=f"{pair[0]} vs {pair[1]}",
                        mode='lines'
                    ))

                fig_rolling.add_hline(y=0, line_dash="dash", line_color="gray")
                fig_rolling.update_layout(
                    title="30-Day Rolling Correlation",
                    xaxis_title="Date",
                    yaxis_title="Correlation",
                    height=400
                )

                st.plotly_chart(fig_rolling, use_container_width=True)
    else:
        st.info("Correlation analysis requires at least 2 stocks.")

# =============================================================================
# TAB 4: RISK METRICS
# =============================================================================
with tab4:
    st.subheader("Risk & Performance Metrics")

    # Calculate metrics for all stocks
    metrics_data = {}
    for stock in returns.columns:
        metrics_data[stock] = calculate_risk_metrics(returns[stock], risk_free_rate)

    metrics_df = pd.DataFrame(metrics_data).T

    # Format for display
    display_df = metrics_df.copy()
    display_df['Annual Return'] = (display_df['Annual Return'] * 100).round(2).astype(str) + '%'
    display_df['Annual Volatility'] = (display_df['Annual Volatility'] * 100).round(2).astype(str) + '%'
    display_df['Max Drawdown'] = (display_df['Max Drawdown'] * 100).round(2).astype(str) + '%'
    display_df['VaR (95%)'] = (display_df['VaR (95%)'] * 100).round(2).astype(str) + '%'
    display_df['CVaR (95%)'] = (display_df['CVaR (95%)'] * 100).round(2).astype(str) + '%'
    display_df['Sharpe Ratio'] = display_df['Sharpe Ratio'].round(2)
    display_df['Sortino Ratio'] = display_df['Sortino Ratio'].round(2)
    display_df['Skewness'] = display_df['Skewness'].round(2)
    display_df['Kurtosis'] = display_df['Kurtosis'].round(2)

    st.dataframe(display_df, use_container_width=True)

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        # Risk-Return scatter
        st.subheader("Risk vs Return")

        fig_rr = go.Figure()

        for idx, stock in enumerate(metrics_df.index):
            fig_rr.add_trace(go.Scatter(
                x=[metrics_df.loc[stock, 'Annual Volatility'] * 100],
                y=[metrics_df.loc[stock, 'Annual Return'] * 100],
                mode='markers+text',
                name=stock,
                marker=dict(size=20, color=colors[idx % len(colors)]),
                text=[stock],
                textposition="top center"
            ))

        fig_rr.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

        fig_rr.update_layout(
            title="Risk-Return Profile",
            xaxis_title="Annual Volatility (%)",
            yaxis_title="Annual Return (%)",
            height=400,
            showlegend=False
        )

        st.plotly_chart(fig_rr, use_container_width=True)

    with col2:
        # Sharpe Ratio comparison
        st.subheader("Sharpe Ratio Comparison")

        sharpe_sorted = metrics_df['Sharpe Ratio'].sort_values()

        fig_sharpe = go.Figure()

        colors_sharpe = ['#10B981' if x > 0 else '#EF4444' for x in sharpe_sorted.values]

        fig_sharpe.add_trace(go.Bar(
            y=sharpe_sorted.index,
            x=sharpe_sorted.values,
            orientation='h',
            marker_color=colors_sharpe,
            text=[f"{x:.2f}" for x in sharpe_sorted.values],
            textposition='outside'
        ))

        fig_sharpe.add_vline(x=0, line_color="black", line_width=1)
        fig_sharpe.add_vline(x=1, line_dash="dash", line_color="green", opacity=0.5)

        fig_sharpe.update_layout(
            title="Sharpe Ratio by Stock",
            xaxis_title="Sharpe Ratio",
            yaxis_title="",
            height=400,
            showlegend=False
        )

        st.plotly_chart(fig_sharpe, use_container_width=True)

    # VaR comparison
    st.markdown("---")
    st.subheader("Value at Risk Comparison")

    var_df = pd.DataFrame({
        'VaR (95%)': metrics_df['VaR (95%)'] * 100,
        'CVaR (95%)': metrics_df['CVaR (95%)'] * 100
    })

    fig_var = go.Figure()

    fig_var.add_trace(go.Bar(
        name='VaR (95%)',
        x=var_df.index,
        y=var_df['VaR (95%)'],
        marker_color='#F59E0B'
    ))

    fig_var.add_trace(go.Bar(
        name='CVaR (95%)',
        x=var_df.index,
        y=var_df['CVaR (95%)'],
        marker_color='#EF4444'
    ))

    fig_var.update_layout(
        title="Value at Risk - Expected Daily Loss",
        xaxis_title="Stock",
        yaxis_title="Daily Loss (%)",
        barmode='group',
        height=400
    )

    st.plotly_chart(fig_var, use_container_width=True)

    st.info("""
    **Metric Definitions:**
    - **Sharpe Ratio**: Risk-adjusted return (higher is better, >1 is good)
    - **Sortino Ratio**: Like Sharpe but only considers downside volatility
    - **Max Drawdown**: Worst peak-to-trough decline
    - **VaR (95%)**: Expected daily loss in the worst 5% of days
    - **CVaR (95%)**: Average loss when VaR is breached
    """)

# =============================================================================
# TAB 5: DRAWDOWN
# =============================================================================
with tab5:
    st.subheader("Drawdown Analysis")

    # Calculate drawdowns
    drawdowns = returns.apply(calculate_drawdown)

    fig_dd = go.Figure()

    for idx, stock in enumerate(drawdowns.columns):
        fig_dd.add_trace(go.Scatter(
            x=drawdowns.index,
            y=drawdowns[stock],
            name=stock,
            fill='tozeroy',
            line=dict(color=colors[idx % len(colors)], width=1),
            opacity=0.7
        ))

    fig_dd.update_layout(
        title="Historical Drawdown",
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        hovermode="x unified",
        height=500
    )

    fig_dd.update_yaxes(autorange="reversed")

    st.plotly_chart(fig_dd, use_container_width=True)

    # Drawdown statistics
    st.markdown("---")
    st.subheader("Drawdown Statistics")

    dd_stats = pd.DataFrame({
        'Current Drawdown': drawdowns.iloc[-1],
        'Max Drawdown': drawdowns.min(),
        'Avg Drawdown': drawdowns.mean(),
        'Days in Drawdown': (drawdowns < 0).sum()
    }).T

    dd_stats = dd_stats.round(2)
    st.dataframe(dd_stats, use_container_width=True)

# =============================================================================
# TAB 6: DASHBOARD
# =============================================================================
with tab6:
    st.subheader("Complete Dashboard View")

    # Create a comprehensive dashboard
    fig_dashboard = make_subplots(
        rows=3, cols=3,
        subplot_titles=(
            'Normalized Prices', 'Total Returns', 'Correlation Matrix',
            'Sharpe Ratio', 'Risk vs Return', 'Return Distribution',
            'Drawdown', 'Max Drawdown', 'Rolling Volatility'
        ),
        specs=[
            [{"colspan": 1}, {"colspan": 1}, {"colspan": 1}],
            [{"colspan": 1}, {"colspan": 1}, {"colspan": 1}],
            [{"colspan": 1}, {"colspan": 1}, {"colspan": 1}]
        ],
        vertical_spacing=0.1,
        horizontal_spacing=0.08
    )

    # 1. Normalized prices
    for idx, stock in enumerate(normalized_prices.columns):
        fig_dashboard.add_trace(
            go.Scatter(x=normalized_prices.index, y=normalized_prices[stock],
                      name=stock, line=dict(color=colors[idx % len(colors)], width=1.5),
                      showlegend=True, legendgroup="prices"),
            row=1, col=1
        )

    # 2. Total Returns
    total_returns = ((prices.iloc[-1] - prices.iloc[0]) / prices.iloc[0] * 100).sort_values()
    colors_bar = ['#10B981' if x > 0 else '#EF4444' for x in total_returns.values]
    fig_dashboard.add_trace(
        go.Bar(y=total_returns.index, x=total_returns.values, orientation='h',
               marker_color=colors_bar, showlegend=False),
        row=1, col=2
    )

    # 3. Correlation heatmap
    if len(returns.columns) > 1:
        corr_matrix = returns.corr()
        fig_dashboard.add_trace(
            go.Heatmap(z=corr_matrix.values, x=corr_matrix.columns, y=corr_matrix.index,
                      colorscale='RdYlGn', zmid=0, showscale=False,
                      text=np.round(corr_matrix.values, 2), texttemplate="%{text}"),
            row=1, col=3
        )

    # 4. Sharpe Ratio
    sharpe_sorted = metrics_df['Sharpe Ratio'].sort_values()
    colors_sharpe = ['#10B981' if x > 0 else '#EF4444' for x in sharpe_sorted.values]
    fig_dashboard.add_trace(
        go.Bar(y=sharpe_sorted.index, x=sharpe_sorted.values, orientation='h',
               marker_color=colors_sharpe, showlegend=False),
        row=2, col=1
    )

    # 5. Risk vs Return
    for idx, stock in enumerate(metrics_df.index):
        fig_dashboard.add_trace(
            go.Scatter(
                x=[metrics_df.loc[stock, 'Annual Volatility'] * 100],
                y=[metrics_df.loc[stock, 'Annual Return'] * 100],
                mode='markers', marker=dict(size=12, color=colors[idx % len(colors)]),
                showlegend=False
            ),
            row=2, col=2
        )

    # 6. Return Distribution
    for idx, stock in enumerate(returns.columns):
        fig_dashboard.add_trace(
            go.Histogram(x=returns[stock]*100, opacity=0.6, showlegend=False,
                        marker_color=colors[idx % len(colors)]),
            row=2, col=3
        )

    # 7. Drawdown
    for idx, stock in enumerate(drawdowns.columns):
        fig_dashboard.add_trace(
            go.Scatter(x=drawdowns.index, y=drawdowns[stock], fill='tozeroy',
                      showlegend=False, line=dict(color=colors[idx % len(colors)], width=1)),
            row=3, col=1
        )

    # 8. Max Drawdown
    max_dd = (metrics_df['Max Drawdown'] * 100).sort_values()
    fig_dashboard.add_trace(
        go.Bar(y=max_dd.index, x=max_dd.values, orientation='h',
               marker_color='#EF4444', showlegend=False),
        row=3, col=2
    )

    # 9. Rolling Volatility
    rolling_vol = returns.rolling(30).std() * np.sqrt(252) * 100
    for idx, stock in enumerate(rolling_vol.columns):
        fig_dashboard.add_trace(
            go.Scatter(x=rolling_vol.index, y=rolling_vol[stock],
                      showlegend=False, line=dict(color=colors[idx % len(colors)], width=1)),
            row=3, col=3
        )

    fig_dashboard.update_layout(
        height=1000,
        title_text="Stock Analysis Dashboard",
        title_font_size=20,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig_dashboard, use_container_width=True)

    # Download data option
    st.markdown("---")
    st.subheader("Export Data")

    col1, col2, col3 = st.columns(3)

    with col1:
        csv_prices = prices.to_csv()
        st.download_button(
            label="📥 Download Prices (CSV)",
            data=csv_prices,
            file_name="stock_prices.csv",
            mime="text/csv"
        )

    with col2:
        csv_returns = returns.to_csv()
        st.download_button(
            label="📥 Download Returns (CSV)",
            data=csv_returns,
            file_name="stock_returns.csv",
            mime="text/csv"
        )

    with col3:
        csv_metrics = metrics_df.to_csv()
        st.download_button(
            label="📥 Download Metrics (CSV)",
            data=csv_metrics,
            file_name="risk_metrics.csv",
            mime="text/csv"
        )

# =============================================================================
# FOOTER
# =============================================================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray; font-size: 0.9rem;">
    <p>Stock Analysis Dashboard | Built with Streamlit & Plotly</p>
    <p>Data Source: Yahoo Finance | Updated: {}</p>
</div>
""".format(datetime.now().strftime("%Y-%m-%d %H:%M")), unsafe_allow_html=True)
