from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .cov import estimate_covariance
from .metrics import mean_abs_offdiag_corr
from .strategies import StrategyOutput, compute_strategy_weights


@dataclass
class BacktestResult:
    strategy: str
    portfolio_returns: pd.Series
    weights: pd.DataFrame
    turnover: pd.Series
    component_returns: pd.DataFrame | None
    insample_offdiag_raw: pd.Series
    insample_offdiag_long_only: pd.Series
    oos_offdiag: pd.Series
    eigen_alignment: pd.Series
    component_weight_drift: pd.Series
    estimation_window_end: pd.Series
    rebalance_dates: pd.DatetimeIndex
    insample_component_corr: pd.DataFrame | None
    oos_component_corr: pd.DataFrame | None


def get_rebalance_indices(index: pd.DatetimeIndex, method: str = "monthly") -> np.ndarray:
    if method != "monthly":
        raise ValueError("Only 'monthly' rebalance is implemented (first trading day of month)")
    if len(index) == 0:
        return np.array([], dtype=int)

    months = index.to_period("M")
    change = np.where(months[1:] != months[:-1])[0] + 1
    return np.concatenate([[0], change])


def _drift_weights(weights: np.ndarray, asset_returns: np.ndarray, gross_return: float) -> np.ndarray:
    denom = 1.0 + gross_return
    if denom <= 0:
        return np.ones_like(weights) / len(weights)
    nxt = weights * (1.0 + asset_returns) / denom
    nxt = np.clip(nxt, 0.0, None)
    s = nxt.sum()
    if s <= 0:
        return np.ones_like(weights) / len(weights)
    return nxt / s


def _one_way_turnover(w_new: np.ndarray, w_old: np.ndarray) -> float:
    return 0.5 * float(np.sum(np.abs(w_new - w_old)))


def run_backtest(
    returns: pd.DataFrame,
    strategy: str = "ecb",
    window: int = 252,
    rebalance: str = "monthly",
    k: int = 5,
    cov_estimator: str = "sample",
    tcost_bps: float = 5.0,
    gamma: float = 1e-3,
    shrink_delta: float = 0.1,
) -> BacktestResult:
    if window < 20:
        raise ValueError("window must be >= 20")
    if returns.shape[0] <= window:
        raise ValueError("Not enough returns rows for requested window")

    returns = returns.sort_index()
    assets = list(returns.columns)

    rebalance_idx_all = get_rebalance_indices(returns.index, method=rebalance)
    rebalance_idx = rebalance_idx_all[rebalance_idx_all >= window]
    if len(rebalance_idx) == 0:
        raise ValueError("No rebalance dates available after applying lookback window")

    traded_dates: list[pd.Timestamp] = []
    daily_weights: list[np.ndarray] = []
    daily_returns: list[float] = []
    daily_component_returns: list[np.ndarray] = []

    turnover_map: dict[pd.Timestamp, float] = {}
    insample_raw_map: dict[pd.Timestamp, float] = {}
    insample_lo_map: dict[pd.Timestamp, float] = {}
    oos_map: dict[pd.Timestamp, float] = {}
    align_map: dict[pd.Timestamp, float] = {}
    component_drift_map: dict[pd.Timestamp, float] = {}
    estimation_end_map: dict[pd.Timestamp, pd.Timestamp] = {}
    insample_corr_mats: list[np.ndarray] = []
    oos_corr_mats: list[np.ndarray] = []

    prev_eigenvectors = None
    prev_component_weights = None
    current_weights = None

    for pos, r_idx in enumerate(rebalance_idx):
        rebalance_date = returns.index[r_idx]
        window_slice = returns.iloc[r_idx - window : r_idx]
        estimation_end = returns.index[r_idx - 1]
        estimation_end_map[rebalance_date] = estimation_end

        cov = estimate_covariance(
            window_slice,
            method=cov_estimator,
            shrink_delta=shrink_delta,
        )

        output: StrategyOutput = compute_strategy_weights(
            strategy=strategy,
            returns_window=window_slice,
            cov=cov,
            k=k,
            gamma=gamma,
            prev_eigenvectors=prev_eigenvectors,
        )

        target_weights = np.clip(output.asset_weights, 0.0, None)
        target_weights = target_weights / target_weights.sum()

        if output.insample_offdiag_raw is not None:
            insample_raw_map[rebalance_date] = output.insample_offdiag_raw
        if output.insample_offdiag_long_only is not None:
            insample_lo_map[rebalance_date] = output.insample_offdiag_long_only
        if output.eigen_alignment_mean is not None:
            align_map[rebalance_date] = output.eigen_alignment_mean
        if output.component_weights is not None and output.component_weights.shape[1] > 1:
            insample_g = window_slice.values @ output.component_weights
            insample_corr_mats.append(np.corrcoef(insample_g, rowvar=False))
            if (
                prev_component_weights is not None
                and prev_component_weights.shape == output.component_weights.shape
            ):
                component_drift_map[rebalance_date] = float(
                    np.mean(np.abs(output.component_weights - prev_component_weights))
                )
            prev_component_weights = output.component_weights.copy()

        prev_eigenvectors = output.eigenvectors

        if current_weights is None:
            turnover = 0.0
        else:
            turnover = _one_way_turnover(target_weights, current_weights)
        turnover_map[rebalance_date] = turnover

        hold_start = r_idx
        hold_end = rebalance_idx[pos + 1] if pos + 1 < len(rebalance_idx) else len(returns)

        hold_component_buffer: list[np.ndarray] = []
        hold_weights = target_weights.copy()

        for day in range(hold_start, hold_end):
            date = returns.index[day]
            r_vec = returns.iloc[day].values.astype(float)

            gross_return = float(np.dot(hold_weights, r_vec))
            net_return = gross_return
            if day == hold_start and turnover > 0:
                net_return -= (tcost_bps / 10000.0) * turnover

            traded_dates.append(date)
            daily_weights.append(hold_weights.copy())
            daily_returns.append(net_return)

            if output.component_weights is not None:
                comp_ret = r_vec @ output.component_weights
                hold_component_buffer.append(comp_ret)
                daily_component_returns.append(comp_ret)

            hold_weights = _drift_weights(hold_weights, r_vec, gross_return)

        current_weights = hold_weights

        if hold_component_buffer:
            hold_arr = np.vstack(hold_component_buffer)
            oos_map[rebalance_date] = mean_abs_offdiag_corr(hold_arr)
            if hold_arr.shape[1] > 1:
                oos_corr_mats.append(np.corrcoef(hold_arr, rowvar=False))

    idx = pd.DatetimeIndex(traded_dates)
    weights_df = pd.DataFrame(daily_weights, index=idx, columns=assets)
    ret_series = pd.Series(daily_returns, index=idx, name=strategy)

    comp_df = None
    if daily_component_returns:
        k_cols = len(daily_component_returns[0])
        cols = [f"comp_{i + 1}" for i in range(k_cols)]
        comp_df = pd.DataFrame(daily_component_returns, index=idx, columns=cols)

    insample_corr_df = None
    if insample_corr_mats:
        avg = np.mean(np.stack(insample_corr_mats), axis=0)
        cols = [f"comp_{i + 1}" for i in range(avg.shape[0])]
        insample_corr_df = pd.DataFrame(avg, index=cols, columns=cols)

    oos_corr_df = None
    if oos_corr_mats:
        avg = np.mean(np.stack(oos_corr_mats), axis=0)
        cols = [f"comp_{i + 1}" for i in range(avg.shape[0])]
        oos_corr_df = pd.DataFrame(avg, index=cols, columns=cols)

    return BacktestResult(
        strategy=strategy,
        portfolio_returns=ret_series,
        weights=weights_df,
        turnover=pd.Series(turnover_map).sort_index(),
        component_returns=comp_df,
        insample_offdiag_raw=pd.Series(insample_raw_map).sort_index(),
        insample_offdiag_long_only=pd.Series(insample_lo_map).sort_index(),
        oos_offdiag=pd.Series(oos_map).sort_index(),
        eigen_alignment=pd.Series(align_map).sort_index(),
        component_weight_drift=pd.Series(component_drift_map).sort_index(),
        estimation_window_end=pd.Series(estimation_end_map).sort_index(),
        rebalance_dates=pd.DatetimeIndex(sorted(turnover_map.keys())),
        insample_component_corr=insample_corr_df,
        oos_component_corr=oos_corr_df,
    )
