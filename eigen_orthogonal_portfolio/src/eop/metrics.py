from __future__ import annotations

import numpy as np
import pandas as pd


def mean_abs_offdiag_corr(data: pd.DataFrame | np.ndarray) -> float:
    """Mean absolute off-diagonal of correlation matrix."""
    if isinstance(data, pd.DataFrame):
        arr = data.values
    else:
        arr = np.asarray(data, dtype=float)

    if arr.ndim != 2 or arr.shape[1] < 2:
        return 0.0

    corr = np.corrcoef(arr, rowvar=False)
    if corr.ndim == 0:
        return 0.0

    mask = ~np.eye(corr.shape[0], dtype=bool)
    vals = np.abs(corr[mask])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0
    return float(np.mean(vals))


def max_drawdown(returns: pd.Series) -> float:
    wealth = (1.0 + returns).cumprod()
    peak = wealth.cummax()
    dd = wealth / peak - 1.0
    return float(dd.min())


def performance_metrics(returns: pd.Series, periods_per_year: int = 252) -> dict[str, float]:
    returns = returns.dropna()
    if returns.empty:
        return {
            "total_return": 0.0,
            "cagr": 0.0,
            "vol": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
        }

    wealth = (1.0 + returns).cumprod()
    total_return = float(wealth.iloc[-1] - 1.0)

    n = len(returns)
    years = n / periods_per_year
    cagr = float(wealth.iloc[-1] ** (1 / years) - 1.0) if years > 0 else 0.0

    vol = float(returns.std(ddof=1) * np.sqrt(periods_per_year))
    mean_ann = float(returns.mean() * periods_per_year)
    sharpe = mean_ann / vol if vol > 0 else 0.0

    return {
        "total_return": total_return,
        "cagr": cagr,
        "vol": vol,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown(returns),
    }


def summarize_backtest(
    returns: pd.Series,
    turnover: pd.Series | None = None,
    periods_per_year: int = 252,
) -> dict[str, float]:
    summary = performance_metrics(returns, periods_per_year=periods_per_year)
    if turnover is None or turnover.empty:
        summary["avg_turnover"] = 0.0
        summary["annualized_turnover"] = 0.0
    else:
        summary["avg_turnover"] = float(turnover.mean())
        summary["annualized_turnover"] = float(turnover.mean() * 12.0)
    return summary
