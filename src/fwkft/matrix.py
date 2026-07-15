"""
fwkft.matrix
============

Matrix construction for Walsh and Walsh–Kaczmarz transforms.

Provides:
    - Hadamard matrix H_N
    - Kaczmarz permutation matrix P_N
    - Walsh–Kaczmarz matrix K_N = P_N H_N

Author
------
Anteneh Tilahun

License
-------
MIT
"""

from __future__ import annotations

import numpy as np

from .permutation import kaczmarz_permutation
from .utils import is_power_of_two



def hadamard_matrix(N: int) -> np.ndarray:
    """
    Construct normalized Walsh–Hadamard matrix.

    Parameters
    ----------
    N : int
        Matrix size (power of two).

    Returns
    -------
    H : ndarray
        N x N orthogonal Hadamard matrix.

    Examples
    --------
    >>> hadamard_matrix(4)

    Notes
    -----
    The matrix satisfies

        H_N H_N^T = I_N
    """

    if not is_power_of_two(N):
        raise ValueError(
            "N must be a power of two."
        )

    H = np.array([[1.0]])

    while H.shape[0] < N:

        H = np.block(
            [
                [H, H],
                [H, -H]
            ]
        )

    return H / np.sqrt(N)



def permutation_matrix(p: np.ndarray) -> np.ndarray:
    """
    Construct a permutation matrix.

    Parameters
    ----------
    p : ndarray
        Permutation array.

    Returns
    -------
    P : ndarray

    Example
    -------
    >>> permutation_matrix([2,0,1])

    gives

    [[0,0,1],
     [1,0,0],
     [0,1,0]]
    """

    p = np.asarray(p)

    N = len(p)

    P = np.zeros((N, N))

    for i, j in enumerate(p):
        P[i, j] = 1

    return P



def kaczmarz_matrix(N: int) -> np.ndarray:
    """
    Construct Walsh–Kaczmarz transform matrix.

    Parameters
    ----------
    N : int

    Returns
    -------
    K : ndarray

    Formula
    -------
        K_N = P_N H_N

    where

        H_N = Walsh–Paley matrix
        P_N = Kaczmarz permutation matrix
    """

    H = hadamard_matrix(N)

    p = kaczmarz_permutation(N)

    P = permutation_matrix(p)

    return P @ H



def walsh_matrix(N: int) -> np.ndarray:
    """
    Alias for Hadamard matrix.

    Returns the Walsh–Paley matrix.
    """

    return hadamard_matrix(N)



def check_orthogonality(A: np.ndarray,
                        tolerance: float = 1e-12) -> bool:
    """
    Check whether a matrix is orthogonal.

    Tests:

        A A^T = I

    Parameters
    ----------
    A : ndarray

    Returns
    -------
    bool
    """

    N = A.shape[0]

    identity = np.eye(N)

    return np.allclose(
        A @ A.T,
        identity,
        atol=tolerance
    )
