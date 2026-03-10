"""Streamlit dashboard for stock market analysis and visualization."""

from __future__ import annotations

import sys
from datetime import date, timedelta
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── resolve project root so `src.*` imports work when launched from any cwd ──
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.analyzer import (
    calculate_moving_averages,
    calculate_returns,
    calculate_sharpe_ratio,
    calculate_volatility,
)
from src.data_fetcher import get_stock_data

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stock Analysis Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# GLOBAL DARK THEME CSS
# ─────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── app background ── */
    html, body, [data-testid="stAppViewContainer"],
    [data-testid="stMain"], section.main {
        background-color: #0e1117;
        color: #e0e0e0;
    }
    /* ── sidebar ── */
    [data-testid="stSidebar"] {
        background-color: #161b22;
    }
    [data-testid="stSidebar"] * { color: #c9d1d9 !important; }

    /* ── metric cards ── */
    [data-testid="stMetric"] {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 16px 20px;
    }
    [data-testid="stMetricLabel"]  { color: #8b949e !important; font-size: 0.78rem; }
    [data-testid="stMetricValue"]  { color: #e0e0e0 !important; font-size: 1.4rem; }
    [data-testid="stMetricDelta"]  { font-size: 0.85rem; }

    /* ── tab strip ── */
    [data-testid="stTabs"] button {
        color: #8b949e;
        font-weight: 600;
        font-size: 0.9rem;
    }
    [data-testid="stTabs"] button[aria-selected="true"] {
        color: #58a6ff !important;
        border-bottom: 2px solid #58a6ff;
    }

    /* ── dividers & misc ── */
    hr { border-color: #30363d; }
    .section-header {
        font-size: 1.05rem;
        font-weight: 600;
        color: #8b949e;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-bottom: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────
PLOTLY_TEMPLATE = "plotly_dark"
PLOTLY_BG = "#0e1117"
PLOTLY_PAPER = "#161b22"

ALL_TICKERS = ["VOO", "SPY", "QQQ", "QQQM", "NVDA", "AAPL", "MSFT", "TSLA", "GOOGL", "AMZN"]
DEFAULT_TICKERS = ["VOO", "QQQM", "NVDA", "AAPL"]

TICKER_COLORS = [
    "#58a6ff", "#3fb950", "#f78166", "#d2a8ff",
    "#ffa657", "#79c0ff", "#56d364", "#ff7b72", "#b083f0",
]


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
def _date_range_to_period(start: date, end: date) -> str:
    """Map a date range into the closest yfinance period string."""
    days = max((end - start).days, 1)
    if days <= 32:
        return "1mo"
    if days <= 95:
        return "3mo"
    if days <= 185:
        return "6mo"
    if days <= 370:
        return "1y"
    if days <= 740:
        return "2y"
    if days <= 1830:
        return "5y"
    return "10y"


def _get_close(df: pd.DataFrame) -> pd.DataFrame:
    """Return the (Adj Close or Close) sub-DataFrame from a multi-ticker df."""
    lvl0 = df.columns.get_level_values(0)
    field = "Adj Close" if "Adj Close" in lvl0 else "Close"
    return df[field]


def _slice_by_dates(df: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    """Trim a DatetimeIndex DataFrame to [start, end] inclusive."""
    mask = (df.index.date >= start) & (df.index.date <= end)
    return df.loc[mask]


def _color_for(ticker: str, ticker_list: list[str]) -> str:
    idx = ticker_list.index(ticker) if ticker in ticker_list else 0
    return TICKER_COLORS[idx % len(TICKER_COLORS)]


def _plotly_layout(title: str = "", height: int = 400, **extra) -> dict:
    """Shared plotly layout with dark theme applied."""
    return dict(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor=PLOTLY_PAPER,
        plot_bgcolor=PLOTLY_BG,
        title=dict(text=title, font=dict(size=15, color="#c9d1d9")),
        height=height,
        margin=dict(l=50, r=20, t=50, b=40),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color="#c9d1d9"),
        ),
        xaxis=dict(gridcolor="#21262d", zerolinecolor="#30363d"),
        yaxis=dict(gridcolor="#21262d", zerolinecolor="#30363d"),
        **extra,
    )


def _fmt_pct(val: float) -> str:
    return f"{val:+.2f}%"


def _fmt_price(val: float) -> str:
    return f"${val:,.2f}"


# ─────────────────────────────────────────────────────────────
# CACHED DATA LOADER
# ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _load_data(tickers: tuple[str, ...], period: str) -> pd.DataFrame:
    """Fetch and cache OHLCV data. Key is (tickers tuple, period string)."""
    return get_stock_data(tickers=list(tickers), period=period)


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📈 Stock Dashboard")
    st.markdown("---")

    st.markdown('<p class="section-header">Tickers</p>', unsafe_allow_html=True)
    dropdown_tickers: list[str] = st.multiselect(
        label="Select tickers",
        options=ALL_TICKERS,
        default=DEFAULT_TICKERS,
        label_visibility="collapsed",
    )

    custom_input: str = st.text_input(
        label="Add custom tickers",
        placeholder="e.g. AMD, META, COST",
        help="Enter additional ticker symbols separated by commas.",
    )

    # Parse and normalise the free-text tickers
    custom_tickers: list[str] = [
        t.strip().upper()
        for t in custom_input.split(",")
        if t.strip()
    ]

    # Merge, preserving order and removing duplicates
    seen: set[str] = set()
    combined_tickers: list[str] = []
    for t in dropdown_tickers + custom_tickers:
        if t not in seen:
            seen.add(t)
            combined_tickers.append(t)

    # Safe fallback when nothing is selected
    selected_tickers = combined_tickers if combined_tickers else DEFAULT_TICKERS
    if not combined_tickers:
        st.warning(f"No tickers selected — defaulting to {', '.join(DEFAULT_TICKERS)}.")

    st.markdown('<p class="section-header">Date Range</p>', unsafe_allow_html=True)
    today = date.today()
    default_start = today - timedelta(days=365)

    start_date: date = st.date_input("Start", value=default_start, max_value=today - timedelta(days=7))
    end_date: date = st.date_input("End", value=today, min_value=start_date + timedelta(days=7), max_value=today)

    st.markdown("---")
    refresh = st.button("🔄 Refresh Data", use_container_width=True)

    if refresh:
        st.cache_data.clear()
        st.success("Cache cleared — reloading…")

    st.markdown("---")
    st.caption(f"Data via yfinance · Last loaded: {today.strftime('%b %d, %Y')}")

# ─────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────
period = _date_range_to_period(start_date, end_date)

with st.spinner("Fetching market data…"):
    try:
        raw_df = _load_data(tuple(sorted(selected_tickers)), period)
    except Exception as err:
        st.error(f"Data fetch failed: {err}")
        st.stop()

df = _slice_by_dates(raw_df, start_date, end_date)

if df.empty:
    st.error("No data returned for the selected tickers and date range.")
    st.stop()

close_prices = _get_close(df)
available_tickers = close_prices.columns.tolist()

# ── report which tickers loaded vs failed ──────────────────────
_requested = set(selected_tickers)
_loaded    = set(available_tickers)
_failed    = _requested - _loaded

if _loaded:
    st.sidebar.success(f"Loaded ({len(_loaded)}): {', '.join(sorted(_loaded))}")
if _failed:
    st.sidebar.warning(f"Not found ({len(_failed)}): {', '.join(sorted(_failed))}")

# ─────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────
tab_overview, tab_deep_dive, tab_risk = st.tabs(
    ["📊  Overview", "🔍  Deep Dive", "⚠️  Risk Analysis"]
)


# ══════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════
with tab_overview:

    # ── metric cards ──────────────────────────────────────────
    daily_returns_df = close_prices.pct_change()
    ytd_start = date(today.year, 1, 1)

    st.markdown("### Snapshot")
    cols = st.columns(len(available_tickers))
    for i, ticker in enumerate(available_tickers):
        series = close_prices[ticker].dropna()
        current_price = series.iloc[-1]
        daily_chg = daily_returns_df[ticker].iloc[-1] * 100

        # YTD: first price on or after Jan 1
        ytd_mask = series.index.date >= ytd_start
        ytd_series = series[ytd_mask]
        ytd_return = (
            ((ytd_series.iloc[-1] / ytd_series.iloc[0]) - 1) * 100
            if len(ytd_series) >= 2
            else 0.0
        )

        with cols[i]:
            st.metric(
                label=ticker,
                value=_fmt_price(current_price),
                delta=f"{_fmt_pct(daily_chg)} today  |  {_fmt_pct(ytd_return)} YTD",
            )

    st.markdown("---")

    # ── normalized price chart ─────────────────────────────────
    st.markdown("### Relative Performance (Base = 100)")

    normalized = close_prices / close_prices.iloc[0] * 100
    fig_norm = go.Figure()
    for ticker in available_tickers:
        fig_norm.add_trace(
            go.Scatter(
                x=normalized.index,
                y=normalized[ticker].round(2),
                mode="lines",
                name=ticker,
                line=dict(width=2, color=_color_for(ticker, available_tickers)),
                hovertemplate=f"<b>{ticker}</b><br>Date: %{{x|%b %d, %Y}}<br>Value: %{{y:.2f}}<extra></extra>",
            )
        )
    fig_norm.add_hline(
        y=100,
        line_dash="dot",
        line_color="#30363d",
        annotation_text="Baseline",
        annotation_font_color="#8b949e",
    )
    fig_norm.update_layout(**_plotly_layout("", height=430))
    st.plotly_chart(fig_norm, use_container_width=True)

    # ── summary table ──────────────────────────────────────────
    st.markdown("### Summary Statistics")

    sharpe = calculate_sharpe_ratio(df)
    vol_df = calculate_volatility(df)
    summary_rows = []
    for ticker in available_tickers:
        s = close_prices[ticker].dropna()
        dr = daily_returns_df[ticker].dropna()
        row = {
            "Ticker": ticker,
            "Current Price": _fmt_price(s.iloc[-1]),
            "Period Return": _fmt_pct(((s.iloc[-1] / s.iloc[0]) - 1) * 100),
            "Ann. Volatility": _fmt_pct(dr.std() * sqrt(252) * 100),
            "Sharpe Ratio": f"{sharpe[ticker]:.2f}" if pd.notna(sharpe.get(ticker)) else "—",
        }
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows).set_index("Ticker")
    st.dataframe(summary_df, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# TAB 2 — DEEP DIVE
# ══════════════════════════════════════════════════════════════
with tab_deep_dive:

    st.markdown("### Single-Stock Deep Dive")

    selected_single = st.selectbox(
        "Select a ticker",
        options=available_tickers,
        index=available_tickers.index("NVDA") if "NVDA" in available_tickers else 0,
    )

    # Extract flat OHLCV for the chosen ticker
    ticker_data = df.xs(selected_single, axis=1, level=1).copy()
    open_col   = ticker_data["Open"]
    high_col   = ticker_data["High"]
    low_col    = ticker_data["Low"]
    close_col  = ticker_data["Adj Close"] if "Adj Close" in ticker_data else ticker_data["Close"]
    volume_col = ticker_data["Volume"]

    sma_20 = close_col.rolling(20).mean()
    sma_50 = close_col.rolling(50).mean()

    # ── candlestick + SMAs ─────────────────────────────────────
    fig_candle = go.Figure()

    fig_candle.add_trace(
        go.Candlestick(
            x=ticker_data.index,
            open=open_col,
            high=high_col,
            low=low_col,
            close=close_col,
            name=selected_single,
            increasing_line_color="#3fb950",
            decreasing_line_color="#f78166",
        )
    )
    fig_candle.add_trace(
        go.Scatter(
            x=ticker_data.index,
            y=sma_20,
            mode="lines",
            name="SMA 20",
            line=dict(color="#ffa657", width=1.5, dash="dot"),
        )
    )
    fig_candle.add_trace(
        go.Scatter(
            x=ticker_data.index,
            y=sma_50,
            mode="lines",
            name="SMA 50",
            line=dict(color="#d2a8ff", width=1.5, dash="dash"),
        )
    )
    fig_candle.update_layout(
        **_plotly_layout(f"{selected_single} — Candlestick with SMAs", height=450),
        xaxis_rangeslider_visible=False,
    )
    st.plotly_chart(fig_candle, use_container_width=True)

    # ── volume bar chart ───────────────────────────────────────
    volume_colors = np.where(
        close_col.diff() >= 0,
        "#3fb950",
        "#f78166",
    )

    fig_vol = go.Figure()
    fig_vol.add_trace(
        go.Bar(
            x=ticker_data.index,
            y=volume_col,
            name="Volume",
            marker_color=volume_colors.tolist(),
            opacity=0.75,
        )
    )
    fig_vol.update_layout(
        **_plotly_layout(f"{selected_single} — Volume", height=230),
        bargap=0.1,
        showlegend=False,
    )
    st.plotly_chart(fig_vol, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# TAB 3 — RISK ANALYSIS
# ══════════════════════════════════════════════════════════════
with tab_risk:

    daily_ret = close_prices.pct_change().dropna(how="all")
    ann_ret   = daily_ret.mean() * 252
    ann_vol   = daily_ret.std() * sqrt(252)
    sharpe_s  = calculate_sharpe_ratio(df)

    col_left, col_right = st.columns(2)

    # ── correlation heatmap ────────────────────────────────────
    with col_left:
        st.markdown("#### Correlation Matrix")
        corr = daily_ret.corr()
        fig_corr = go.Figure(
            go.Heatmap(
                z=corr.values,
                x=corr.columns.tolist(),
                y=corr.index.tolist(),
                colorscale="RdBu",
                zmid=0,
                zmin=-1,
                zmax=1,
                text=corr.round(2).values,
                texttemplate="%{text}",
                textfont=dict(size=12, color="white"),
                hoverongaps=False,
                showscale=True,
            )
        )
        fig_corr.update_layout(**_plotly_layout("", height=380))
        fig_corr.update_layout(xaxis=dict(side="bottom"))
        st.plotly_chart(fig_corr, use_container_width=True)

    # ── risk vs return scatter ─────────────────────────────────
    with col_right:
        st.markdown("#### Risk vs Return")
        bubble_sizes = sharpe_s.clip(lower=0.01).fillna(0.01) * 30

        fig_rr = go.Figure()
        for ticker in available_tickers:
            fig_rr.add_trace(
                go.Scatter(
                    x=[ann_vol[ticker] * 100],
                    y=[ann_ret[ticker] * 100],
                    mode="markers+text",
                    name=ticker,
                    text=[ticker],
                    textposition="top center",
                    textfont=dict(color="#c9d1d9"),
                    marker=dict(
                        size=max(float(bubble_sizes.get(ticker, 10)), 8),
                        color=_color_for(ticker, available_tickers),
                        line=dict(width=1, color="#30363d"),
                        opacity=0.85,
                    ),
                    hovertemplate=(
                        f"<b>{ticker}</b><br>"
                        "Volatility: %{x:.1f}%<br>"
                        "Return: %{y:.1f}%<br>"
                        f"Sharpe: {sharpe_s.get(ticker, float('nan')):.2f}"
                        "<extra></extra>"
                    ),
                )
            )
        fig_rr.update_layout(**_plotly_layout("", height=380))
        fig_rr.update_layout(
            xaxis=dict(title="Annualized Volatility (%)", gridcolor="#21262d"),
            yaxis=dict(title="Annualized Return (%)", gridcolor="#21262d"),
            showlegend=False,
        )
        st.plotly_chart(fig_rr, use_container_width=True)

    # ── rolling volatility ─────────────────────────────────────
    st.markdown("#### Rolling 30-Day Volatility (Annualized)")
    rolling_vol = daily_ret.rolling(30, min_periods=30).std() * sqrt(252) * 100
    rolling_vol = rolling_vol.dropna(how="all")

    fig_rv = go.Figure()
    for ticker in available_tickers:
        if ticker in rolling_vol.columns:
            fig_rv.add_trace(
                go.Scatter(
                    x=rolling_vol.index,
                    y=rolling_vol[ticker].round(2),
                    mode="lines",
                    name=ticker,
                    line=dict(width=2, color=_color_for(ticker, available_tickers)),
                    hovertemplate=f"<b>{ticker}</b><br>Date: %{{x|%b %d, %Y}}<br>Vol: %{{y:.2f}}%<extra></extra>",
                )
            )
    fig_rv.update_layout(**_plotly_layout("", height=360))
    fig_rv.update_layout(yaxis=dict(title="Volatility (%)", gridcolor="#21262d"))
    st.plotly_chart(fig_rv, use_container_width=True)

    # ── risk stats table ───────────────────────────────────────
    st.markdown("#### Risk Metrics Table")
    risk_rows = []
    for ticker in available_tickers:
        risk_rows.append({
            "Ticker": ticker,
            "Ann. Return": _fmt_pct(ann_ret[ticker] * 100),
            "Ann. Volatility": _fmt_pct(ann_vol[ticker] * 100),
            "Sharpe Ratio": f"{sharpe_s[ticker]:.2f}" if pd.notna(sharpe_s.get(ticker)) else "—",
            "Max Daily Gain": _fmt_pct(daily_ret[ticker].max() * 100),
            "Max Daily Loss": _fmt_pct(daily_ret[ticker].min() * 100),
        })
    st.dataframe(pd.DataFrame(risk_rows).set_index("Ticker"), use_container_width=True)

# ─────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Data sourced from Yahoo Finance via yfinance · "
    "For educational purposes only — not financial advice."
)
