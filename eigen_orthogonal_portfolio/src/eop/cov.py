from __future__ import annotations

import numpy as np
import pandas as pd


def symmetrize_and_make_psd(cov: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Force symmetry and positive semidefiniteness."""
    cov = np.asarray(cov, dtype=float)
    sym = 0.5 * (cov + cov.T)
    vals, vecs = np.linalg.eigh(sym)
    vals = np.clip(vals, eps, None)
    psd = (vecs * vals) @ vecs.T
    return 0.5 * (psd + psd.T)


def sample_covariance(returns: pd.DataFrame | np.ndarray, eps: float = 1e-10) -> np.ndarray:
    data = np.asarray(returns, dtype=float)
    cov = np.cov(data, rowvar=False, ddof=1)
    return symmetrize_and_make_psd(cov, eps=eps)


def shrink_covariance(
    returns: pd.DataFrame | np.ndarray,
    delta: float = 0.1,
    eps: float = 1e-10,
) -> np.ndarray:
    data = np.asarray(returns, dtype=float)
    sample = np.cov(data, rowvar=False, ddof=1)
    n = sample.shape[0]
    mu = float(np.trace(sample)) / n
    shrunk = (1.0 - delta) * sample + delta * mu * np.eye(n)
    return symmetrize_and_make_psd(shrunk, eps=eps)


def ledoit_wolf_covariance(
    returns: pd.DataFrame | np.ndarray,
    fallback_delta: float = 0.1,
    eps: float = 1e-10,
) -> np.ndarray:
    data = np.asarray(returns, dtype=float)
    try:
        from sklearn.covariance import LedoitWolf

        model = LedoitWolf().fit(data)
        cov = model.covariance_
    except Exception:
        cov = shrink_covariance(data, delta=fallback_delta, eps=eps)
        return cov

    return symmetrize_and_make_psd(cov, eps=eps)


def estimate_covariance(
    returns: pd.DataFrame | np.ndarray,
    method: str = "sample",
    shrink_delta: float = 0.1,
    eps: float = 1e-10,
) -> np.ndarray:
    method = method.lower()
    if method == "sample":
        return sample_covariance(returns, eps=eps)
    if method == "ledoitwolf":
        return ledoit_wolf_covariance(returns, fallback_delta=shrink_delta, eps=eps)
    if method == "shrink":
        return shrink_covariance(returns, delta=shrink_delta, eps=eps)
    raise ValueError("Unknown covariance method: expected sample, ledoitwolf, or shrink")
