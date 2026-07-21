from __future__ import annotations

from pathlib import Path
import re
from typing import Iterable
import warnings

import numpy as np
import pandas as pd
import yfinance as yf

SP500_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
SP500_PROXY_ETF = "SPY"
OMXS30_WIKI_URL = "https://en.wikipedia.org/wiki/OMX_Stockholm_30"
DAX40_WIKI_URL = "https://en.wikipedia.org/wiki/DAX"
MIN_SP500_CONSTITUENT_COUNT = 300

DEFAULT_TICKERS = [
    "AAPL",  # Apple
    "MSFT",  # Microsoft
    "AMZN",  # Amazon
    "NVDA",  # Nvidia
    "GOOGL",  # Alphabet Class A (Google)
    "GOOG",  # Alphabet Class C (Google)
    "META",  # Meta Platforms
    "TSLA",  # Tesla
    "BRK-B",  # Berkshire Hathaway Class B
    "JPM",  # JPMorgan Chase
    "UNH",  # UnitedHealth Group
    "XOM",  # Exxon Mobil
    "V",  # Visa
    "LLY",  # Eli Lilly
    "AVGO",  # Broadcom
    "MA",  # Mastercard
    "JNJ",  # Johnson & Johnson
    "PG",  # Procter & Gamble
    "HD",  # Home Depot
    "COST",  # Costco
    "MRK",  # Merck
    "ABBV",  # AbbVie
    "ADBE",  # Adobe
    "CRM",  # Salesforce
    "PEP",  # PepsiCo
    "NFLX",  # Netflix
    "KO",  # Coca-Cola
    "AMD",  # Advanced Micro Devices
    "BAC",  # Bank of America
    "CVX",  # Chevron
    "WMT",  # Walmart
    "TMO",  # Thermo Fisher Scientific
    "MCD",  # McDonald's
    "CSCO",  # Cisco Systems
    "ACN",  # Accenture
    "DIS",  # Walt Disney
    "DHR",  # Danaher
    "VZ",  # Verizon
    "ABT",  # Abbott Laboratories
    "PFE",  # Pfizer
]

DEFAULT_OMXS30_TICKERS = [
    "ABB.ST",  # ABB
    "ALFA.ST",  # Alfa Laval
    "ALIV-SDB.ST",  # Autoliv SDR
    "ASSA-B.ST",  # Assa Abloy B
    "ATCO-A.ST",  # Atlas Copco A
    "ATCO-B.ST",  # Atlas Copco B
    "AZN.ST",  # AstraZeneca
    "BOL.ST",  # Boliden
    "ELUX-B.ST",  # Electrolux B
    "EQT.ST",  # EQT
    "ERIC-B.ST",  # Ericsson B
    "ESSITY-B.ST",  # Essity B
    "EVO.ST",  # Evolution
    "GETI-B.ST",  # Getinge B
    "HEXA-B.ST",  # Hexagon B
    "HM-B.ST",  # H&M B
    "INVE-B.ST",  # Investor B
    "KINV-B.ST",  # Kinnevik B
    "NDA-SE.ST",  # Nordea
    "NIBE-B.ST",  # Nibe B
    "SAND.ST",  # Sandvik
    "SAAB-B.ST",  # Saab B
    "SCA-B.ST",  # SCA B
    "SEB-A.ST",  # SEB A
    "SHB-A.ST",  # Handelsbanken A
    "SKF-B.ST",  # SKF B
    "SWED-A.ST",  # Swedbank A
    "TEL2-B.ST",  # Tele2 B
    "TELIA.ST",  # Telia
    "VOLV-B.ST",  # Volvo B
]

DEFAULT_DAX40_TICKERS = [
    "ADS.DE",  # Adidas
    "AIR.DE",  # Airbus
    "ALV.DE",  # Allianz
    "BAS.DE",  # BASF
    "BAYN.DE",  # Bayer
    "BEI.DE",  # Beiersdorf
    "BMW.DE",  # BMW
    "BNR.DE",  # Brenntag
    "CBK.DE",  # Commerzbank
    "CON.DE",  # Continental
    "1COV.DE",  # Covestro
    "DB1.DE",  # Deutsche Boerse
    "DBK.DE",  # Deutsche Bank
    "DHL.DE",  # DHL Group
    "DTE.DE",  # Deutsche Telekom
    "ENR.DE",  # Siemens Energy
    "EOAN.DE",  # E.ON
    "FRE.DE",  # Fresenius
    "FME.DE",  # Fresenius Medical Care
    "HEN3.DE",  # Henkel Pref
    "HNR1.DE",  # Hannover Rueck
    "IFX.DE",  # Infineon
    "LIN.DE",  # Linde
    "MBG.DE",  # Mercedes-Benz Group
    "MRK.DE",  # Merck KGaA
    "MTX.DE",  # MTU Aero Engines
    "MUV2.DE",  # Munich Re
    "P911.DE",  # Porsche AG
    "PAH3.DE",  # Porsche SE
    "PUM.DE",  # Puma
    "QIA.DE",  # Qiagen
    "RHM.DE",  # Rheinmetall
    "RWE.DE",  # RWE
    "SAP.DE",  # SAP
    "SIE.DE",  # Siemens
    "SHL.DE",  # Siemens Healthineers
    "SY1.DE",  # Symrise
    "VNA.DE",  # Vonovia
    "VOW3.DE",  # Volkswagen Pref
    "ZAL.DE",  # Zalando
]

# Offline fallback when both yfinance holdings and Wikipedia are unavailable.
# Intended for the default `sp500_top_n=50` path.
DEFAULT_SP500_TOP50_TICKERS = [
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "GOOGL",
    "GOOG",
    "META",
    "BRK-B",
    "AVGO",
    "TSLA",
    "LLY",
    "JPM",
    "V",
    "XOM",
    "MA",
    "UNH",
    "COST",
    "WMT",
    "JNJ",
    "PG",
    "NFLX",
    "HD",
    "ABBV",
    "BAC",
    "KO",
    "CRM",
    "CVX",
    "MRK",
    "AMD",
    "PEP",
    "ADBE",
    "TMO",
    "ORCL",
    "LIN",
    "CSCO",
    "ACN",
    "MCD",
    "ABT",
    "DHR",
    "WFC",
    "DIS",
    "PM",
    "VZ",
    "TXN",
    "INTU",
    "QCOM",
    "AMGN",
    "NOW",
    "IBM",
    "GE",
]

KNOWN_EXCHANGE_SUFFIXES = {
    "ST",
    "DE",
    "AS",
    "PA",
    "L",
    "MI",
    "HE",
    "CO",
    "OL",
    "BR",
    "HK",
    "T",
    "AX",
    "NZ",
    "SI",
    "VX",
    "SW",
    "TO",
}


def _normalize_symbol_general(symbol: str) -> str:
    """
    Normalize symbols while preserving known exchange suffixes like .DE / .ST.
    Also repairs cached forms like SAP-DE back to SAP.DE.
    """
    s = str(symbol).strip().upper().replace(" ", "").replace("/", "-")
    if not s:
        return s

    # Repair trailing -EXCH to .EXCH for known exchange suffixes.
    m_dash = re.match(r"^([A-Z0-9\-]+)-([A-Z]{1,4})$", s)
    if m_dash and m_dash.group(2) in KNOWN_EXCHANGE_SUFFIXES:
        return f"{m_dash.group(1)}.{m_dash.group(2)}"

    # Keep exchange suffix format unchanged when already valid.
    m_dot = re.match(r"^([A-Z0-9\-]+)\.([A-Z]{1,4})$", s)
    if m_dot and m_dot.group(2) in KNOWN_EXCHANGE_SUFFIXES:
        return s

    # Non-exchange dots (e.g. BRK.B) are Yahoo class separators -> dash.
    return s.replace(".", "-")


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


def load_tickers_from_csv(path: str | Path) -> list[str]:
    """Load tickers from a CSV file (expects 'Symbol' column or first column)."""
    csv_path = Path(path)
    if not csv_path.exists():
        raise ValueError(f"Ticker file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"Ticker file is empty: {csv_path}")

    if "Symbol" in df.columns:
        series = df["Symbol"]
    else:
        series = df.iloc[:, 0]

    tickers = series.astype(str).map(_normalize_symbol_general)
    tickers = [t for t in tickers.tolist() if t]
    if not tickers:
        raise ValueError(f"No tickers found in file: {csv_path}")
    return list(dict.fromkeys(tickers))


def _normalize_ticker_values(values: pd.Series) -> list[str]:
    tickers = (
        values.astype(str).str.strip().str.upper().str.replace(".", "-", regex=False)
    )
    parsed = [t for t in tickers.tolist() if t]
    return list(dict.fromkeys(parsed))


def _extract_tickers_from_holdings_df(df: pd.DataFrame) -> list[str]:
    if df is None or df.empty:
        return []

    lower_cols = {c.lower(): c for c in df.columns}
    for key in ["symbol", "ticker"]:
        if key in lower_cols:
            return _normalize_ticker_values(df[lower_cols[key]])

    idx = pd.Series(df.index, dtype="object")
    idx_norm = _normalize_ticker_values(idx)
    # Keep only plausible ticker tokens from index.
    plausible = [t for t in idx_norm if 1 <= len(t) <= 8 and t.replace("-", "").isalnum()]
    return plausible


def fetch_sp500_tickers_from_yfinance(etf_ticker: str = SP500_PROXY_ETF) -> list[str]:
    """Fetch S&P 500 constituents using SPY holdings data from yfinance."""
    ticker_obj = yf.Ticker(etf_ticker)
    funds = ticker_obj.get_funds_data()

    best_candidates: list[str] = []
    for attr in ["equity_holdings", "top_holdings"]:
        df = getattr(funds, attr, None)
        if isinstance(df, pd.DataFrame):
            candidates = _extract_tickers_from_holdings_df(df)
            if len(candidates) > len(best_candidates):
                best_candidates = candidates
            if len(candidates) >= MIN_SP500_CONSTITUENT_COUNT:
                return candidates

    if best_candidates:
        raise ValueError(
            f"Parsed only {len(best_candidates)} symbols from yfinance ETF holdings; expected >= {MIN_SP500_CONSTITUENT_COUNT}."
        )
    raise ValueError("Unable to parse S&P 500 tickers from yfinance ETF holdings.")


def fetch_sp500_tickers(
    cache_path: str | Path | None = None,
    refresh: bool = False,
    min_count: int = MIN_SP500_CONSTITUENT_COUNT,
) -> list[str]:
    """Fetch current S&P 500 constituents; yfinance first, then Wikipedia, then cache."""
    min_count = max(1, int(min_count))
    cache_file = Path(cache_path) if cache_path is not None else None
    if cache_file is not None and cache_file.exists() and not refresh:
        cached = load_tickers_from_csv(cache_file)
        if len(cached) >= min_count:
            return cached
        warnings.warn(
            f"Ignoring cached S&P 500 list with only {len(cached)} symbols at {cache_file}; expected >= {min_count}; attempting refresh."
        )

    try:
        parsed = fetch_sp500_tickers_from_yfinance()
        if cache_file is not None:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"Symbol": parsed}).to_csv(cache_file, index=False)
        return parsed
    except Exception as y_exc:
        wiki_exc: Exception | None = None
        err_msg = f"yfinance error: {y_exc}"
        try:
            tables = pd.read_html(SP500_WIKI_URL)
            if not tables:
                raise ValueError("Failed to load S&P 500 constituents table")

            constituents = tables[0]
            if "Symbol" not in constituents.columns:
                raise ValueError("Could not find 'Symbol' column in S&P 500 table")

            parsed = _normalize_ticker_values(constituents["Symbol"])
            if len(parsed) < min_count:
                raise ValueError(
                    f"Wikipedia parse returned only {len(parsed)} symbols; expected >= {min_count}."
                )
            if cache_file is not None:
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame({"Symbol": parsed}).to_csv(cache_file, index=False)
            return parsed
        except Exception as exc:
            wiki_exc = exc
            err_msg = f"yfinance error: {y_exc}; wikipedia error: {wiki_exc}"

        if cache_file is not None and cache_file.exists():
            cached = load_tickers_from_csv(cache_file)
            if len(cached) >= min_count:
                warnings.warn(
                    f"Failed to refresh S&P 500 tickers ({err_msg}); using cached list from {cache_file}."
                )
                return cached
            raise ValueError(
                f"Failed to refresh S&P 500 tickers ({err_msg}) and cached file has only {len(cached)} symbols; expected >= {min_count}."
            )
        if min_count <= len(DEFAULT_SP500_TOP50_TICKERS):
            warnings.warn(
                f"Failed to fetch full S&P 500 tickers ({err_msg}); using built-in top-50 fallback list."
            )
            fallback = DEFAULT_SP500_TOP50_TICKERS.copy()
            if cache_file is not None:
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame({"Symbol": fallback}).to_csv(cache_file, index=False)
            return fallback
        raise ValueError(
            "Failed to fetch S&P 500 tickers and requested minimum count exceeds available built-in fallback size."
        )


def _normalize_omxs30_symbols(values: pd.Series) -> list[str]:
    symbols = values.astype(str).str.strip().str.upper()
    symbols = symbols.str.replace(" ", "-", regex=False)
    symbols = symbols.str.replace(r"[^A-Z0-9\\-]", "", regex=True)

    out: list[str] = []
    for sym in symbols.tolist():
        if not sym:
            continue
        if sym.endswith("-ST"):
            sym = f"{sym[:-3]}.ST"
        elif not sym.endswith(".ST"):
            sym = f"{sym}.ST"
        out.append(sym)
    return list(dict.fromkeys(out))


def fetch_omxs30_tickers(
    cache_path: str | Path | None = None,
    refresh: bool = False,
) -> list[str]:
    """Fetch current OMXS30 constituents from Wikipedia; fallback to built-in list and cache."""
    cache_file = Path(cache_path) if cache_path is not None else None
    if cache_file is not None and cache_file.exists() and not refresh:
        return load_tickers_from_csv(cache_file)

    try:
        tables = pd.read_html(OMXS30_WIKI_URL)
        if not tables:
            raise ValueError("No tables found on OMXS30 Wikipedia page")

        candidates: list[str] = []
        for tbl in tables:
            cols = {str(c).strip().lower(): c for c in tbl.columns}
            symbol_col = None
            for c in ["symbol", "ticker", "ticker symbol"]:
                if c in cols:
                    symbol_col = cols[c]
                    break
            if symbol_col is None:
                continue
            parsed = _normalize_omxs30_symbols(tbl[symbol_col])
            if len(parsed) >= 20:
                candidates = parsed
                break

        if len(candidates) < 20:
            raise ValueError("Could not parse OMXS30 constituent symbols from Wikipedia tables")

        if cache_file is not None:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"Symbol": candidates}).to_csv(cache_file, index=False)
        return candidates
    except Exception as exc:
        if cache_file is not None and cache_file.exists():
            warnings.warn(
                f"Failed to refresh OMXS30 tickers ({exc}); using cached list from {cache_file}."
            )
            return load_tickers_from_csv(cache_file)

        warnings.warn(
            "Falling back to built-in OMXS30 ticker list; it may lag index reconstitutions."
        )
        fallback = DEFAULT_OMXS30_TICKERS.copy()
        if cache_file is not None:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"Symbol": fallback}).to_csv(cache_file, index=False)
        return fallback


def _normalize_dax_symbols(values: pd.Series) -> list[str]:
    symbols = values.astype(str).str.strip().str.upper()
    symbols = symbols.str.replace(" ", "", regex=False)
    symbols = symbols.str.replace(r"[^A-Z0-9\\-]", "", regex=True)

    out: list[str] = []
    for sym in symbols.tolist():
        if not sym:
            continue
        if sym.endswith("-DE"):
            sym = f"{sym[:-3]}.DE"
        elif not sym.endswith(".DE"):
            sym = f"{sym}.DE"
        out.append(sym)
    return list(dict.fromkeys(out))


def fetch_dax40_tickers(
    cache_path: str | Path | None = None,
    refresh: bool = False,
) -> list[str]:
    """Fetch current DAX40 constituents from Wikipedia; fallback to built-in list and cache."""
    cache_file = Path(cache_path) if cache_path is not None else None
    if cache_file is not None and cache_file.exists() and not refresh:
        return load_tickers_from_csv(cache_file)

    try:
        tables = pd.read_html(DAX40_WIKI_URL)
        if not tables:
            raise ValueError("No tables found on DAX page")

        candidates: list[str] = []
        for tbl in tables:
            cols = {str(c).strip().lower(): c for c in tbl.columns}
            symbol_col = None
            for c in ["ticker symbol", "ticker", "symbol"]:
                if c in cols:
                    symbol_col = cols[c]
                    break
            if symbol_col is None:
                continue
            parsed = _normalize_dax_symbols(tbl[symbol_col])
            if len(parsed) >= 30:
                candidates = parsed
                break

        if len(candidates) < 30:
            raise ValueError("Could not parse DAX40 constituent symbols from Wikipedia tables")

        if cache_file is not None:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"Symbol": candidates}).to_csv(cache_file, index=False)
        return candidates
    except Exception as exc:
        if cache_file is not None and cache_file.exists():
            warnings.warn(
                f"Failed to refresh DAX40 tickers ({exc}); using cached list from {cache_file}."
            )
            return load_tickers_from_csv(cache_file)

        warnings.warn(
            "Falling back to built-in DAX40 ticker list; it may lag index reconstitutions."
        )
        fallback = DEFAULT_DAX40_TICKERS.copy()
        if cache_file is not None:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"Symbol": fallback}).to_csv(cache_file, index=False)
        return fallback


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


def _to_float(value: object) -> float:
    try:
        if value is None:
            return float("nan")
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _fetch_single_ticker_meta(ticker: str) -> dict[str, float | str]:
    yft = yf.Ticker(ticker)

    market_cap = float("nan")

    info: dict[str, object] = {}
    try:
        info = yft.get_info() or {}
    except Exception:
        info = {}

    market_cap = _to_float(info.get("marketCap"))

    try:
        fast_info = yft.fast_info
    except Exception:
        fast_info = {}

    if not np.isfinite(market_cap):
        market_cap = _to_float(getattr(fast_info, "market_cap", None))
        if not np.isfinite(market_cap) and isinstance(fast_info, dict):
            market_cap = _to_float(fast_info.get("market_cap"))

    return {
        "ticker": ticker,
        "market_cap": market_cap,
    }


def fetch_market_cap_metadata(tickers: list[str]) -> pd.DataFrame:
    """Fetch market-cap metadata needed for cap-weighted benchmark weighting."""
    rows = [_fetch_single_ticker_meta(t) for t in tickers]
    meta = pd.DataFrame(rows).set_index("ticker")
    return meta


def load_or_fetch_market_cap_metadata(
    tickers: list[str],
    cache_path: str | Path | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """Load cached market-cap metadata and fetch missing rows from yfinance."""
    tickers = list(dict.fromkeys([t.upper() for t in tickers]))
    cached = pd.DataFrame(columns=["market_cap"])

    if cache_path is not None:
        cache_file = Path(cache_path)
        if cache_file.exists() and not refresh:
            cached = pd.read_csv(cache_file, index_col=0)
            cached.index = cached.index.astype(str).str.upper()

    missing = [t for t in tickers if t not in cached.index]
    fetched = pd.DataFrame(columns=cached.columns)
    if missing:
        fetched = fetch_market_cap_metadata(missing)

    merged = pd.concat([cached, fetched], axis=0)
    merged = merged[~merged.index.duplicated(keep="last")]
    merged = merged.reindex(tickers)

    if cache_path is not None:
        cache_file = Path(cache_path)
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        merged.sort_index().to_csv(cache_file)

    return merged
