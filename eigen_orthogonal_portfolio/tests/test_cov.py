import numpy as np
import pandas as pd

from eop.cov import estimate_covariance


def test_covariance_is_symmetric_and_psd():
    rng = np.random.default_rng(42)
    returns = pd.DataFrame(rng.normal(0, 0.01, size=(300, 8)))

    for method in ["sample", "ledoitwolf", "shrink"]:
        cov = estimate_covariance(returns, method=method, shrink_delta=0.2)
        assert cov.shape == (8, 8)
        assert np.allclose(cov, cov.T, atol=1e-10)
        evals = np.linalg.eigvalsh(cov)
        assert np.all(evals >= -1e-10)
