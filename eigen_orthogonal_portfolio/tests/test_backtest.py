import numpy as np
import pandas as pd

from eop.backtest import run_backtest


def _synthetic_returns(n_days: int = 520, n_assets: int = 6) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    idx = pd.bdate_range("2020-01-01", periods=n_days)
    data = rng.normal(0.0002, 0.01, size=(n_days, n_assets))
    cols = [f"A{i}" for i in range(n_assets)]
    return pd.DataFrame(data, index=idx, columns=cols)


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

    for strategy in ["ew", "mv_lo", "ecb", "ecb_equala"]:
        result = run_backtest(
            returns=returns,
            strategy=strategy,
            window=60,
            rebalance="monthly",
            k=3,
            cov_estimator="sample",
            tcost_bps=5,
        )

        w = result.weights
        assert ((w >= -1e-8).all().all())
        assert np.allclose(w.sum(axis=1).values, 1.0, atol=1e-6)
