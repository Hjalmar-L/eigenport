import numpy as np

from eop.eigen import eigen_decompose, reconstruct_from_eigen


def test_eigen_sorting_and_reconstruction():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(6, 6))
    cov = a @ a.T

    res = eigen_decompose(cov)

    assert np.all(np.diff(res.eigenvalues) <= 0)
    recon = reconstruct_from_eigen(res.eigenvalues, res.eigenvectors)
    assert np.allclose(cov, recon, atol=1e-8)


def test_eigen_sign_alignment_against_previous_vectors():
    rng = np.random.default_rng(1)
    a = rng.normal(size=(5, 5))
    cov = a @ a.T

    first = eigen_decompose(cov)
    prev = -first.eigenvectors
    aligned = eigen_decompose(cov, prev_vectors=prev)

    dots = np.sum(aligned.eigenvectors * prev, axis=0)
    assert np.all(dots >= 0)
    assert aligned.alignment is not None
