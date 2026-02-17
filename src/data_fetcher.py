"""Utilities for fetching and cleaning stock market data."""

from __future__ import annotations

from typing import Any

import pandas as pd
import yfinance as yf


def _validate_tickers(tickers: list[str]) -> list[str]:
    """Validate and normalize a list of ticker symbols.

    Args:
        tickers: Raw list of ticker symbols.

    Returns:
        A cleaned list of uppercase ticker symbols with duplicates removed.

    Raises:
        ValueError: If tickers is empty or contains invalid values.
    """
    if not isinstance(tickers, list) or not tickers:
        raise ValueError("tickers must be a non-empty list of stock symbols.")

    cleaned: list[str] = []
    for ticker in tickers:
        if not isinstance(ticker, str) or not ticker.strip():
            raise ValueError("Each ticker must be a non-empty string.")
        normalized = ticker.strip().upper()
        if normalized not in cleaned:
            cleaned.append(normalized)

    return cleaned


def _clean_ohlcv_data(raw_data: pd.DataFrame) -> pd.DataFrame:
    """Clean OHLCV data and enforce a no-missing-values output.

    Args:
        raw_data: Raw DataFrame returned by yfinance download.

    Returns:
        A cleaned DataFrame with DateTime index, sorted rows, and no missing values.

    Raises:
        ValueError: If data is empty before or after cleaning.
    """
    if raw_data is None or raw_data.empty:
        raise ValueError("No data returned from yfinance.")

    data = raw_data.copy()

    if isinstance(data.columns, pd.MultiIndex):
        data = data.sort_index(axis=1)
        ticker_level = data.columns.nlevels - 1
        tickers = data.columns.get_level_values(ticker_level).unique().tolist()
        failed_tickers: list[str] = []

        for ticker in tickers:
            ticker_slice = data.xs(ticker, axis=1, level=ticker_level, drop_level=False)
            if ticker_slice.isna().all().all():
                failed_tickers.append(str(ticker))

        if failed_tickers:
            data = data.drop(columns=failed_tickers, level=ticker_level)

    data = data.sort_index()
    data = data.dropna(how="all")
    data = data.ffill().bfill()
    data = data.dropna(how="any")

    if data.empty:
        raise ValueError(
            "Fetched data contains only missing values after cleaning. "
            "Try a different period or ticker set."
        )

    return data


def get_stock_data(tickers: list[str], period: str = "1y") -> pd.DataFrame:
    """Download historical OHLCV data for multiple ticker symbols.

    Args:
        tickers: List of stock ticker symbols (for example: ['VOO', 'NVDA']).
        period: yfinance period string (for example: '1y', '5y', 'max').

    Returns:
        A cleaned pandas DataFrame of OHLCV data with no missing values.
        For multiple tickers, columns are a MultiIndex in the form:
        (PriceField, Ticker), such as ('Close', 'NVDA').

    Raises:
        ValueError: If input arguments are invalid or data is not usable.
        RuntimeError: If yfinance download fails.
    """
    cleaned_tickers = _validate_tickers(tickers)

    if not isinstance(period, str) or not period.strip():
        raise ValueError("period must be a non-empty string, e.g. '1y' or '5y'.")

    try:
        data = yf.download(
            tickers=cleaned_tickers,
            period=period.strip(),
            auto_adjust=False,
            progress=False,
            group_by="column",
            threads=False,
        )
    except Exception as exc:
        message = (
            f"Failed to download stock data for tickers {cleaned_tickers} "
            f"with period '{period}': {exc}"
        )
        raise RuntimeError(message) from exc

    try:
        return _clean_ohlcv_data(data)
    except ValueError as exc:
        raise ValueError(
            f"Unable to prepare stock data for {cleaned_tickers}: {exc}"
        ) from exc


def get_stock_info(ticker: str) -> dict[str, Any]:
    """Fetch key company metadata for a single ticker.

    Args:
        ticker: Stock ticker symbol.

    Returns:
        Dictionary containing:
            - ticker: normalized ticker symbol
            - company_name: company long name (if available)
            - sector: company sector (if available)
            - market_cap: market capitalization (if available)
            - pe_ratio: trailing P/E ratio (if available)

    Raises:
        ValueError: If ticker input is invalid.
        RuntimeError: If metadata fetch fails.
    """
    if not isinstance(ticker, str) or not ticker.strip():
        raise ValueError("ticker must be a non-empty string.")

    normalized = ticker.strip().upper()

    try:
        stock = yf.Ticker(normalized)
        info = stock.info or {}
    except Exception as exc:
        raise RuntimeError(
            f"Failed to fetch stock info for ticker '{normalized}': {exc}"
        ) from exc

    return {
        "ticker": normalized,
        "company_name": info.get("longName"),
        "sector": info.get("sector"),
        "market_cap": info.get("marketCap"),
        "pe_ratio": info.get("trailingPE"),
    }
