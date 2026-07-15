"""
fwkft.permutation
=================

Kaczmarz permutation utilities for Walsh systems.

The Walsh–Kaczmarz system is obtained from the Walsh–Paley
system by reversing the binary digits (or applying the
corresponding permutation of indices).

Author
------
Anteneh Tilahun Adimasu

License
-------
MIT
"""

from __future__ import annotations

import numpy as np


def bit_reverse(n: int, bits: int) -> int:
    """
    Reverse the binary representation of an integer.

    Parameters
    ----------
    n : int
        Integer to reverse.
    bits : int
        Number of binary digits.

    Returns
    -------
    int

    Example
    -------
    >>> bit_reverse(3, 3)
    6

    Explanation:
        3 = 011
        reverse -> 110 = 6
    """

    result = 0

    for i in range(bits):
        if (n >> i) & 1:
            result |= 1 << (bits - 1 - i)

    return result



def kaczmarz_permutation(N: int) -> np.ndarray:
    """
    Construct the Walsh–Kaczmarz permutation.

    Parameters
    ----------
    N : int
        Transform size. Must be a power of two.

    Returns
    -------
    permutation : ndarray

    Notes
    -----
    For N = 2^m, the permutation is defined by

        κ(n) = bit_reverse(n,m)

    Examples
    --------
    >>> kaczmarz_permutation(8)
    array([0,4,2,6,1,5,3,7])
    """

    if N <= 0 or (N & (N - 1)) != 0:
        raise ValueError(
            "N must be a positive power of two."
        )

    m = int(np.log2(N))

    permutation = np.zeros(N, dtype=int)

    for n in range(N):
        permutation[n] = bit_reverse(n, m)

    return permutation



def inverse_permutation(p: np.ndarray) -> np.ndarray:
    """
    Compute inverse of a permutation.

    Parameters
    ----------
    p : ndarray

    Returns
    -------
    ndarray

    Example
    -------
    >>> inverse_permutation([0,2,1])
    array([0,2,1])
    """

    p = np.asarray(p)

    inv = np.zeros_like(p)

    for i, value in enumerate(p):
        inv[value] = i

    return inv



def permute_vector(x: np.ndarray, p: np.ndarray) -> np.ndarray:
    """
    Apply permutation to a vector.

    Parameters
    ----------
    x : ndarray
        Input vector.

    p : ndarray
        Permutation indices.

    Returns
    -------
    ndarray

    Example
    -------
    >>> permute_vector([10,20,30],[2,0,1])
    array([30,10,20])
    """

    x = np.asarray(x)

    return x[p]



def kaczmarz_order(N: int) -> np.ndarray:
    """
    Generate indices in Walsh–Kaczmarz order.

    Parameters
    ----------
    N : int

    Returns
    -------
    ndarray
    """

    return kaczmarz_permutation(N)
