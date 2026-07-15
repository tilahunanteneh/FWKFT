"""
Permutation tests.
"""

import numpy as np

from fwkft.permutation import (
    kaczmarz_permutation,
    inverse_permutation,
)


def test_inverse():

    p = kaczmarz_permutation(32)

    inv = inverse_permutation(p)

    for i in range(32):
        assert inv[p[i]] == i


def test_permutation():

    p = kaczmarz_permutation(16)

    assert sorted(p.tolist()) == list(range(16))
