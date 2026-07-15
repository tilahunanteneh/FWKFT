"""
fwkft.transforms
================

Fast Walsh–Hadamard Transform (FWHT)
and Fast Walsh–Kaczmarz Fourier Transform (FWKFT).

Author
------
Anteneh Tilahun Adimasu

License
-------
MIT
"""

from __future__ import annotations

import numpy as np

from .utils import check_input
from .matrix import (
    hadamard_matrix,
    kaczmarz_matrix,
)


# -------------------------------------------------------
# Fast Walsh-Hadamard Transform
# -------------------------------------------------------

def fwht(x):
    """
    Fast Walsh-Hadamard Transform.

    Parameters
    ----------
    x : array_like

    Returns
    -------
    ndarray
    """

    x = check_input(x).copy()

    N = len(x)

    h = 1

    while h < N:

        for i in range(0, N, h * 2):

            for j in range(i, i + h):

                a = x[j]
                b = x[j + h]

                x[j] = a + b
                x[j + h] = a - b

        h *= 2

    return x / np.sqrt(N)


# -------------------------------------------------------
# Inverse FWHT
# -------------------------------------------------------

def ifwht(y):
    """
    Inverse Fast Walsh-Hadamard Transform.
    """

    return fwht(y)


# -------------------------------------------------------
# Reference Walsh-Kaczmarz Transform
# -------------------------------------------------------

def fwkft_matrix(x):
    """
    Reference implementation using the
    Walsh-Kaczmarz matrix.

    Complexity
    ----------
    O(N²)

    Used for verification.
    """

    x = check_input(x)

    K = kaczmarz_matrix(len(x))

    return K @ x


# -------------------------------------------------------
# Reference inverse
# -------------------------------------------------------

def ifwkft_matrix(y):
    """
    Reference inverse transform.

    Since K is orthogonal,

        K^{-1}=K^T
    """

    y = check_input(y)

    K = kaczmarz_matrix(len(y))

    return K.T @ y


# -------------------------------------------------------
# Placeholder for fast algorithm
# -------------------------------------------------------

def fwkft(x):
    """
    Fast Walsh-Kaczmarz Fourier Transform.

    Currently uses the matrix implementation.

    This function will later be replaced by the
    recursive butterfly algorithm from the paper.
    """

    return fwkft_matrix(x)


def ifwkft(y):
    """
    Inverse Fast Walsh-Kaczmarz Fourier Transform.
    """

    return ifwkft_matrix(y)
