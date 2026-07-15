"""
FWKFT: Fast Walsh–Kaczmarz Fourier Transform
===========================================

A Python library implementing the Fast Walsh–Kaczmarz Fourier Transform
(FWKFT), Fast Walsh–Hadamard Transform (FWHT), and related utilities.

Author:
    Anteneh Tilahun

License:
    MIT

Example
-------
>>> import numpy as np
>>> from fwkft import fwkft
>>>
>>> x = np.array([1, 2, 3, 4])
>>> y = fwkft(x)
>>> print(y)
"""

from .transforms import (
    fwht,
    ifwht,
    fwkft,
    ifwkft,
)

from .matrix import (
    hadamard_matrix,
    kaczmarz_matrix,
)

from .permutation import (
    kaczmarz_permutation,
)

__version__ = "0.1.0"
__author__ = "Anteneh Tilahun"
__license__ = "MIT"

__all__ = [
    # Transforms
    "fwht",
    "ifwht",
    "fwkft",
    "ifwkft",

    # Matrices
    "hadamard_matrix",
    "kaczmarz_matrix",

    # Permutations
    "kaczmarz_permutation",
]
