from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd

from .components import build_components_with_diagnostics, component_returns
from .eigen import eigen_decompose
from .optimizer import solve_component_risk_parity, solve_long_only_min_variance


@dataclass
class StrategyOutput:
    asset_weights: np.ndarray
    component_weights: np.ndarray | None
    component_blend: np.ndarray | None
    insample_offdiag_raw: float | None
    insample_offdiag_long_only: float | None
    explained_variance_ratio: np.ndarray | None
    eigenvectors: np.ndarray | None
    eigen_alignment_mean: float | None


def _normalize_simplex(x: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(x, dtype=float), 0.0, None)
    s = x.sum()
    if s <= 0:
        return np.ones_like(x) / len(x)
    return x / s


def compute_strategy_weights(
    strategy: str,
    returns_window: pd.DataFrame,
    cov: np.ndarray | None,
    k: int,
    gamma: float,
    prev_eigenvectors: np.ndarray | None = None,
    benchmark_weights: np.ndarray | None = None,
) -> StrategyOutput:
    strategy = strategy.lower()
    n = returns_window.shape[1]

    if strategy == "ew":
        return StrategyOutput(
            asset_weights=np.ones(n) / n,
            component_weights=None,
            component_blend=None,
            insample_offdiag_raw=None,
            insample_offdiag_long_only=None,
            explained_variance_ratio=None,
            eigenvectors=None,
            eigen_alignment_mean=None,
        )

    if strategy == "mv_lo":
        if cov is None:
            raise ValueError("cov is required for strategy 'mv_lo'")
        return StrategyOutput(
            asset_weights=solve_long_only_min_variance(cov),
            component_weights=None,
            component_blend=None,
            insample_offdiag_raw=None,
            insample_offdiag_long_only=None,
            explained_variance_ratio=None,
            eigenvectors=None,
            eigen_alignment_mean=None,
        )

    if strategy in {"market_cap", "float_cap"}:
        if benchmark_weights is None:
            warnings.warn("No benchmark_weights provided for market_cap; falling back to EW")
            benchmark_weights = np.ones(n) / n
        weights = _normalize_simplex(benchmark_weights)
        return StrategyOutput(
            asset_weights=weights,
            component_weights=None,
            component_blend=None,
            insample_offdiag_raw=None,
            insample_offdiag_long_only=None,
            explained_variance_ratio=None,
            eigenvectors=None,
            eigen_alignment_mean=None,
        )

    if strategy not in {"ecb", "ecb_equala"}:
        raise ValueError(
            "Unknown strategy. Use ew, mv_lo, ecb, ecb_equala, or market_cap"
        )
    if cov is None:
        raise ValueError("cov is required for strategy 'ecb' and 'ecb_equala'")

    eig = eigen_decompose(cov, prev_vectors=prev_eigenvectors)
    comp = build_components_with_diagnostics(
        returns_window=returns_window,
        cov=cov,
        eigenvectors=eig.eigenvectors,
        k=k,
        gamma=gamma,
    )

    g = component_returns(returns_window, comp.long_only_weights)
    s = np.cov(g.values, rowvar=False, ddof=1)
    variances = np.diag(s) if np.ndim(s) == 2 else np.array([float(s)])

    if strategy == "ecb":
        blend = solve_component_risk_parity(variances)
    else:
        kk = comp.long_only_weights.shape[1]
        blend = np.ones(kk) / kk

    asset_weights = _normalize_simplex(comp.long_only_weights @ blend)

    align = None
    if eig.alignment is not None:
        align = float(np.mean(eig.alignment))

    return StrategyOutput(
        asset_weights=asset_weights,
        component_weights=comp.long_only_weights,
        component_blend=blend,
        insample_offdiag_raw=comp.raw_insample_offdiag,
        insample_offdiag_long_only=comp.long_only_insample_offdiag,
        explained_variance_ratio=eig.explained_variance_ratio,
        eigenvectors=eig.eigenvectors,
        eigen_alignment_mean=align,
    )
