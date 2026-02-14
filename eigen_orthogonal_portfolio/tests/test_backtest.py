import numpy as np
import pandas as pd

from eop.backtest import run_backtest


def _synthetic_returns(n_days: int = 520, n_assets: int = 6) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    idx = pd.bdate_range("2020-01-01", periods=n_days)
    data = rng.normal(0.0002, 0.01, size=(n_days, n_assets))
    cols = [f"A{i}" for i in range(n_assets)]
    return pd.DataFrame(data, index=idx, columns=cols)


def _synthetic_prices_from_returns(returns: pd.DataFrame) -> pd.DataFrame:
    prices = 100.0 * np.exp(returns.cumsum())
    prices.columns = returns.columns
    return prices


def test_no_lookahead_estimation_window_end_before_rebalance():
    returns = _synthetic_returns()
    result = run_backtest(
        returns=returns,
        strategy="ew",
        window=60,
        rebalance="monthly",
    )

    for reb_date, end_date in result.estimation_window_end.items():
        assert end_date < reb_date


def test_long_only_strategies_respect_constraints():
    returns = _synthetic_returns()
    prices = _synthetic_prices_from_returns(returns)
    meta = pd.DataFrame(
        {
            "float_shares": np.linspace(1e8, 2e8, returns.shape[1]),
            "shares_outstanding": np.linspace(1.2e8, 2.2e8, returns.shape[1]),
            "market_cap": np.linspace(1e10, 2e10, returns.shape[1]),
        },
        index=returns.columns,
    )

    for strategy in ["ew", "mv_lo", "ecb", "ecb_equala", "market_cap"]:
        kwargs = {}
        if strategy == "market_cap":
            kwargs = {"prices": prices, "market_cap_meta": meta}
        result = run_backtest(
            returns=returns,
            strategy=strategy,
            window=60,
            rebalance="monthly",
            k=3,
            cov_estimator="sample",
            tcost_bps=5,
            **kwargs,
        )

        w = result.weights
        assert ((w >= -1e-8).all().all())
        assert np.allclose(w.sum(axis=1).values, 1.0, atol=1e-6)
