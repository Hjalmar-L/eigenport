from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .metrics import mean_abs_offdiag_corr
from .optimizer import solve_long_only_component_projection


@dataclass
class ComponentBuildResult:
    raw_weights: np.ndarray
    long_only_weights: np.ndarray
    raw_insample_offdiag: float
    long_only_insample_offdiag: float


def build_raw_eigenportfolios(eigenvectors: np.ndarray, k: int) -> np.ndarray:
    """Build unconstrained eigen-portfolios for diagnostics."""
    v = np.asarray(eigenvectors, dtype=float)
    n = v.shape[0]
    k = min(k, v.shape[1])
    w = np.zeros((n, k))
    for idx in range(k):
        col = v[:, idx]
        denom = np.sum(np.abs(col))
        if denom <= 0:
            w[:, idx] = 0.0
        else:
            w[:, idx] = col / denom
    return w


def component_returns(returns: pd.DataFrame | np.ndarray, component_weights: np.ndarray) -> pd.DataFrame:
    data = np.asarray(returns, dtype=float)
    out = data @ component_weights
    cols = [f"comp_{i + 1}" for i in range(component_weights.shape[1])]
    if isinstance(returns, pd.DataFrame):
        return pd.DataFrame(out, index=returns.index, columns=cols)
    return pd.DataFrame(out, columns=cols)


def build_long_only_components(
    cov: np.ndarray,
    eigenvectors: np.ndarray,
    k: int,
    gamma: float = 1e-3,
) -> np.ndarray:
    """Approximate top-k eigenvectors with long-only simplex-constrained portfolios."""
    n = cov.shape[0]
    k = min(k, eigenvectors.shape[1])
    w = np.zeros((n, k))
    for idx in range(k):
        w[:, idx] = solve_long_only_component_projection(
            cov=cov,
            target_vector=eigenvectors[:, idx],
            gamma=gamma,
        )
    return w


def build_components_with_diagnostics(
    returns_window: pd.DataFrame,
    cov: np.ndarray,
    eigenvectors: np.ndarray,
    k: int,
    gamma: float = 1e-3,
) -> ComponentBuildResult:
    raw = build_raw_eigenportfolios(eigenvectors=eigenvectors, k=k)
    long_only = build_long_only_components(cov=cov, eigenvectors=eigenvectors, k=k, gamma=gamma)

    g_raw = component_returns(returns_window, raw)
    g_long_only = component_returns(returns_window, long_only)

    raw_offdiag = mean_abs_offdiag_corr(g_raw)
    lo_offdiag = mean_abs_offdiag_corr(g_long_only)

    return ComponentBuildResult(
        raw_weights=raw,
        long_only_weights=long_only,
        raw_insample_offdiag=raw_offdiag,
        long_only_insample_offdiag=lo_offdiag,
    )
