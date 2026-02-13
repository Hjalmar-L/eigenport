from __future__ import annotations

import warnings

import numpy as np
from scipy.optimize import minimize


def _simplex_constraints(n: int):
    bounds = [(0.0, 1.0) for _ in range(n)]
    constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - 1.0}]
    return bounds, constraints


def _safe_normalize_long_only(x: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(x, dtype=float), 0.0, None)
    s = clipped.sum()
    if s <= 0:
        return np.ones_like(clipped) / len(clipped)
    return clipped / s


def solve_long_only_min_variance(cov: np.ndarray) -> np.ndarray:
    n = cov.shape[0]
    x0 = np.ones(n) / n
    bounds, constraints = _simplex_constraints(n)

    def obj(x: np.ndarray) -> float:
        return float(x @ cov @ x)

    res = minimize(
        obj,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 300, "ftol": 1e-12},
    )

    if not res.success:
        warnings.warn(f"MV-LO optimization failed ({res.message}); falling back to EW")
        return x0

    return _safe_normalize_long_only(res.x)


def solve_component_risk_parity(var_components: np.ndarray) -> np.ndarray:
    """Approximate risk parity on independent component variances."""
    var_components = np.clip(np.asarray(var_components, dtype=float), 1e-12, None)
    k = len(var_components)
    x0 = np.ones(k) / k
    bounds, constraints = _simplex_constraints(k)

    def obj(a: np.ndarray) -> float:
        contrib = var_components * (a**2)
        target = np.mean(contrib)
        return float(np.sum((contrib - target) ** 2))

    res = minimize(
        obj,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 300, "ftol": 1e-12},
    )

    if not res.success:
        warnings.warn(f"Component risk-parity optimization failed ({res.message}); using equal weights")
        return x0

    return _safe_normalize_long_only(res.x)


def solve_long_only_component_projection(
    cov: np.ndarray,
    target_vector: np.ndarray,
    gamma: float = 1e-3,
) -> np.ndarray:
    """Long-only projection toward an eigenvector with small variance regularization."""
    n = cov.shape[0]
    v = np.asarray(target_vector, dtype=float)
    x0 = _safe_normalize_long_only(np.maximum(v, 0.0))
    if np.allclose(x0, 0):
        x0 = np.ones(n) / n

    bounds, constraints = _simplex_constraints(n)

    def obj(w: np.ndarray) -> float:
        dist = np.sum((w - v) ** 2)
        reg = gamma * (w @ cov @ w)
        return float(dist + reg)

    res = minimize(
        obj,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 400, "ftol": 1e-12},
    )

    if not res.success:
        warnings.warn(
            f"Component long-only projection failed ({res.message}); using EW component fallback"
        )
        return np.ones(n) / n

    return _safe_normalize_long_only(res.x)
