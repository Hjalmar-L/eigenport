import numpy as np
import pandas as pd

from eop.components import build_components_with_diagnostics
from eop.eigen import eigen_decompose
from eop.metrics import mean_abs_offdiag_corr


def test_long_only_component_constraints():
    rng = np.random.default_rng(7)
    returns = pd.DataFrame(rng.normal(0, 0.01, size=(300, 10)))
    cov = np.cov(returns.values, rowvar=False, ddof=1)
    eig = eigen_decompose(cov)

    comp = build_components_with_diagnostics(
        returns_window=returns,
        cov=cov,
        eigenvectors=eig.eigenvectors,
        k=4,
        gamma=1e-3,
    )

    w = comp.long_only_weights
    assert w.shape == (10, 4)
    assert np.all(w >= -1e-10)
    assert np.allclose(w.sum(axis=0), 1.0, atol=1e-8)


def test_offdiag_metric():
    perfectly_correlated = pd.DataFrame(
        {
            "c1": [1.0, 2.0, 3.0, 4.0],
            "c2": [2.0, 4.0, 6.0, 8.0],
        }
    )
    metric = mean_abs_offdiag_corr(perfectly_correlated)
    assert np.isclose(metric, 1.0)

    identity_like = np.eye(4)
    metric2 = mean_abs_offdiag_corr(identity_like)
    assert metric2 < 0.5
