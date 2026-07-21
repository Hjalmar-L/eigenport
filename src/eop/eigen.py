from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class EigenResult:
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    explained_variance_ratio: np.ndarray
    alignment: np.ndarray | None


def eigen_decompose(
    cov: np.ndarray,
    prev_vectors: np.ndarray | None = None,
) -> EigenResult:
    """Eigendecompose symmetric covariance and align signs to previous vectors."""
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]

    alignment = None
    if prev_vectors is not None and prev_vectors.shape == vecs.shape:
        alignment = np.ones(vecs.shape[1])
        for k in range(vecs.shape[1]):
            dot = float(np.dot(vecs[:, k], prev_vectors[:, k]))
            if dot < 0.0:
                vecs[:, k] *= -1.0
            alignment[k] = abs(dot)

    total = float(np.sum(vals))
    if total <= 0:
        explained = np.zeros_like(vals)
    else:
        explained = vals / total

    return EigenResult(
        eigenvalues=vals,
        eigenvectors=vecs,
        explained_variance_ratio=explained,
        alignment=alignment,
    )


def reconstruct_from_eigen(eigenvalues: np.ndarray, eigenvectors: np.ndarray) -> np.ndarray:
    return (eigenvectors * eigenvalues) @ eigenvectors.T
