"""Analytical functions for stock market datasets."""

from __future__ import annotations

from math import sqrt

import pandas as pd

from src.data_fetcher import get_stock_data


TRADING_DAYS_PER_YEAR = 252


def _validate_dataframe(df: pd.DataFrame) -> None:
    """Validate that input is a non-empty pandas DataFrame."""
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    if df.empty:
        raise ValueError("Input DataFrame is empty.")


def _extract_close_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Extract close-price matrix from single or multi-ticker DataFrame."""
    _validate_dataframe(df)

    if isinstance(df.columns, pd.MultiIndex):
        first_level = df.columns.get_level_values(0)
        price_field = "Adj Close" if "Adj Close" in first_level else "Close"
        if price_field not in first_level:
            raise ValueError("Could not find 'Adj Close' or 'Close' columns.")
        close_prices = df[price_field].copy()
        return close_prices.sort_index()

    if "Adj Close" in df.columns:
        return df[["Adj Close"]].copy()
    if "Close" in df.columns:
        return df[["Close"]].copy()
    raise ValueError("Could not find 'Adj Close' or 'Close' columns.")


def calculate_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Add daily and cumulative return columns to stock price data.

    Args:
        df: Price DataFrame from `get_stock_data`.

    Returns:
        DataFrame with added return columns:
            - For multi-ticker data: ('Daily Return', ticker) and
              ('Cumulative Return', ticker)
            - For single-ticker data: 'Daily Return' and 'Cumulative Return'
    """
    _validate_dataframe(df)
    result = df.copy()
    close_prices = _extract_close_prices(df)

    daily_return = close_prices.pct_change().fillna(0.0)
    cumulative_return = (1.0 + daily_return).cumprod() - 1.0

    if isinstance(result.columns, pd.MultiIndex):
        for ticker in close_prices.columns:
            result[("Daily Return", ticker)] = daily_return[ticker]
            result[("Cumulative Return", ticker)] = cumulative_return[ticker]
        result = result.sort_index(axis=1)
        return result

    result["Daily Return"] = daily_return.iloc[:, 0]
    result["Cumulative Return"] = cumulative_return.iloc[:, 0]
    return result


def calculate_volatility(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate rolling 30-day annualized volatility from close prices.

    Args:
        df: Price DataFrame from `get_stock_data`.

    Returns:
        DataFrame where each column is a ticker (or single close column)
        containing rolling 30-day annualized volatility.
    """
    close_prices = _extract_close_prices(df)
    daily_return = close_prices.pct_change()
    volatility = daily_return.rolling(window=30, min_periods=30).std() * sqrt(
        TRADING_DAYS_PER_YEAR
    )
    return volatility.dropna(how="all")


def calculate_sharpe_ratio(
    df: pd.DataFrame, risk_free_rate: float = 0.05
) -> pd.Series:
    """Calculate annualized Sharpe ratio for each stock.

    Args:
        df: Price DataFrame from `get_stock_data`.
        risk_free_rate: Annual risk-free rate as a decimal (default 0.05).

    Returns:
        pandas Series of annualized Sharpe ratio values by ticker/column.
    """
    close_prices = _extract_close_prices(df)
    daily_return = close_prices.pct_change().dropna(how="all")

    annual_return = daily_return.mean() * TRADING_DAYS_PER_YEAR
    annual_volatility = daily_return.std() * sqrt(TRADING_DAYS_PER_YEAR)

    sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility
    return sharpe_ratio.replace([float("inf"), float("-inf")], pd.NA)


def calculate_correlation(tickers: list[str]) -> pd.DataFrame:
    """Compute correlation matrix between stock daily returns.

    Args:
        tickers: List of ticker symbols to compare.

    Returns:
        Correlation matrix DataFrame based on daily close-price returns.
    """
    data = get_stock_data(tickers=tickers, period="1y")
    close_prices = _extract_close_prices(data)
    daily_return = close_prices.pct_change().dropna(how="all")
    return daily_return.corr()


def calculate_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    """Add 20-day and 50-day simple moving average columns.

    Args:
        df: Price DataFrame from `get_stock_data`.

    Returns:
        DataFrame with moving average columns:
            - For multi-ticker data: ('SMA 20', ticker) and ('SMA 50', ticker)
            - For single-ticker data: 'SMA 20' and 'SMA 50'
    """
    _validate_dataframe(df)
    result = df.copy()
    close_prices = _extract_close_prices(df)

    sma_20 = close_prices.rolling(window=20, min_periods=20).mean()
    sma_50 = close_prices.rolling(window=50, min_periods=50).mean()

    if isinstance(result.columns, pd.MultiIndex):
        for ticker in close_prices.columns:
            result[("SMA 20", ticker)] = sma_20[ticker]
            result[("SMA 50", ticker)] = sma_50[ticker]
        result = result.sort_index(axis=1)
        return result.dropna(how="any")

    result["SMA 20"] = sma_20.iloc[:, 0]
    result["SMA 50"] = sma_50.iloc[:, 0]
    return result.dropna(how="any")
