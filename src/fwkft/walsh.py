"""
fwkft.walsh
===========

Walsh–Paley and Walsh–Kaczmarz functions on [0,1).

The implementation is based on the Rademacher functions.

Author
------
Anteneh Tilahun

License
-------
MIT
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------
# Rademacher function
# ---------------------------------------------------------

def rademacher(k: int, x):
    """
    k-th Rademacher function.

    Parameters
    ----------
    k : int
        Index (k >= 0)

    x : float or ndarray
        Point(s) in [0,1)

    Returns
    -------
    ndarray or scalar

    r_k(x) = (-1)^{ floor(2^{k+1} x) }
    """

    x = np.asarray(x)

    return (-1.0) ** np.floor((2 ** (k + 1)) * x)


# ---------------------------------------------------------
# Binary expansion
# ---------------------------------------------------------

def binary_digits(n: int):
    """
    Binary digits of n (least significant first).

    Example
    -------
    13 = 1101

    returns

    [1,0,1,1]
    """

    digits = []

    while n:

        digits.append(n & 1)

        n >>= 1

    if len(digits) == 0:
        digits = [0]

    return digits


# ---------------------------------------------------------
# Walsh–Paley
# ---------------------------------------------------------

def walsh_paley(n: int, x):
    """
    Walsh–Paley function.

    Parameters
    ----------
    n : int

    x : float or ndarray

    Returns
    -------
    ndarray
    """

    bits = binary_digits(n)

    value = np.ones_like(np.asarray(x), dtype=float)

    for k, b in enumerate(bits):

        if b:

            value *= rademacher(k, x)

    return value


# ---------------------------------------------------------
# Kaczmarz permutation
# ---------------------------------------------------------

def kaczmarz_index(n: int) -> int:
    """
    Reverse binary digits.

    Example

    3 = 011

    -> 110 = 6
    """

    bits = binary_digits(n)

    result = 0

    for b in bits:

        result = (result << 1) | b

    return result


# ---------------------------------------------------------
# Walsh–Kaczmarz
# ---------------------------------------------------------

def walsh_kaczmarz(n: int, x):
    """
    Walsh–Kaczmarz function.

    Parameters
    ----------
    n : int

    x : float or ndarray

    Returns
    -------
    ndarray
    """

    return walsh_paley(kaczmarz_index(n), x)


# ---------------------------------------------------------
# Evaluate first N functions
# ---------------------------------------------------------

def walsh_system(N: int, x, ordering="kaczmarz"):
    """
    Evaluate the first N Walsh functions.

    Parameters
    ----------
    N : int

    x : ndarray

    ordering : str
        'paley'
        'kaczmarz'

    Returns
    -------
    ndarray

    shape = (N,len(x))
    """

    x = np.asarray(x)

    W = np.zeros((N, len(x)))

    if ordering.lower() == "paley":

        for n in range(N):
            W[n] = walsh_paley(n, x)

    else:

        for n in range(N):
            W[n] = walsh_kaczmarz(n, x)

    return W
