from __future__ import annotations

import numpy as np
import pandas as pd


WINSORIZE_CHOICES = {"none", "p01_p99"}


def drop_tickers_with_missing(
    prices: pd.DataFrame,
    max_missing_frac: float = 0.05,
) -> pd.DataFrame:
    """Drop tickers exceeding allowed missing fraction."""
    missing_frac = prices.isna().mean(axis=0)
    keep = missing_frac <= max_missing_frac
    filtered = prices.loc[:, keep]
    if filtered.shape[1] == 0:
        raise ValueError("All tickers were dropped due to missing data")
    return filtered


def forward_fill_small_gaps(prices: pd.DataFrame, fill_limit: int = 3) -> pd.DataFrame:
    """Forward-fill short gaps only; larger gaps remain NaN."""
    return prices.ffill(limit=fill_limit)


def preprocess_prices(
    prices: pd.DataFrame,
    max_missing_frac: float = 0.05,
    fill_limit: int = 3,
) -> pd.DataFrame:
    """Clean and align price data."""
    if prices.empty:
        raise ValueError("Empty price frame")

    clean = prices.copy().sort_index()
    clean = drop_tickers_with_missing(clean, max_missing_frac=max_missing_frac)
    clean = forward_fill_small_gaps(clean, fill_limit=fill_limit)
    clean = clean.dropna(axis=0, how="any")

    if clean.shape[0] < 2:
        raise ValueError("Not enough data after preprocessing")
    return clean


def compute_returns(
    prices: pd.DataFrame,
    method: str = "log",
    winsorize: str = "none",
) -> pd.DataFrame:
    """Compute daily returns from cleaned prices."""
    if method not in {"log", "simple"}:
        raise ValueError("method must be 'log' or 'simple'")
    if winsorize not in WINSORIZE_CHOICES:
        raise ValueError(f"winsorize must be one of {sorted(WINSORIZE_CHOICES)}")

    if method == "log":
        rets = np.log(prices / prices.shift(1))
    else:
        rets = prices.pct_change()

    rets = rets.dropna(axis=0, how="any")

    if winsorize == "p01_p99":
        low = rets.quantile(0.01)
        high = rets.quantile(0.99)
        rets = rets.clip(lower=low, upper=high, axis=1)

    return rets
