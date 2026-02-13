from __future__ import annotations

from typing import Iterable

import pandas as pd
import yfinance as yf

DEFAULT_TICKERS = [
    "AAPL",
    "MSFT",
    "AMZN",
    "NVDA",
    "GOOGL",
    "GOOG",
    "META",
    "TSLA",
    "BRK-B",
    "JPM",
    "UNH",
    "XOM",
    "V",
    "LLY",
    "AVGO",
    "MA",
    "JNJ",
    "PG",
    "HD",
    "COST",
    "MRK",
    "ABBV",
    "ADBE",
    "CRM",
    "PEP",
    "NFLX",
    "KO",
    "AMD",
    "BAC",
    "CVX",
    "WMT",
    "TMO",
    "MCD",
    "CSCO",
    "ACN",
    "DIS",
    "DHR",
    "VZ",
    "ABT",
    "PFE",
]


def parse_tickers(tickers: str | Iterable[str] | None) -> list[str]:
    """Parse tickers from CLI string or iterable and normalize casing."""
    if tickers is None:
        return DEFAULT_TICKERS.copy()
    if isinstance(tickers, str):
        values = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    else:
        values = [str(t).strip().upper() for t in tickers if str(t).strip()]
    unique = list(dict.fromkeys(values))
    return unique or DEFAULT_TICKERS.copy()


def fetch_adj_close(
    tickers: list[str],
    start: str,
    end: str,
) -> pd.DataFrame:
    """Download adjusted close prices from yfinance."""
    if not tickers:
        raise ValueError("Ticker list is empty")

    data = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        progress=False,
        auto_adjust=False,
        actions=False,
        group_by="column",
        threads=True,
    )

    if data.empty:
        raise ValueError("No data returned from yfinance")

    if "Adj Close" in data.columns:
        prices = data["Adj Close"].copy()
    else:
        prices = data.copy()

    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=tickers[0])

    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    prices = prices.sort_index()
    prices = prices.loc[:, ~prices.columns.duplicated()]
    return prices
