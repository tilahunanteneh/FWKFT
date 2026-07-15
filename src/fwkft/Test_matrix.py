"""
Matrix construction tests.
"""

import numpy as np

from fwkft.matrix import (
    hadamard_matrix,
    kaczmarz_matrix,
)


def test_hadamard_orthogonal():

    H = hadamard_matrix(16)

    I = np.eye(16)

    assert np.allclose(
        H @ H.T,
        I,
        atol=1e-12,
    )


def test_kaczmarz_orthogonal():

    K = kaczmarz_matrix(16)

    I = np.eye(16)

    assert np.allclose(
        K @ K.T,
        I,
        atol=1e-12,
    )
