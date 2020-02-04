import numpy as np
from ncon import ncon

# The different calls deliberately use different variation of things like are
# the arguments lists or tuples.


def test_matrixproduct():
    a = np.random.randn(3, 4)
    b = np.random.randn(4, 5)
    ab_ncon = ncon([a, b], ((-1, 1), (1, -2)))
    ab_np = np.dot(a, b)
    assert np.allclose(ab_ncon, ab_np)


def test_disconnected():
    a = np.random.randn(2, 3)
    b = np.random.randn(4)
    ab_ncon = ncon((a, b), ([-3, -2], [-1]))
    ab_np = np.einsum("ij,k -> kji", a, b)
    assert np.allclose(ab_ncon, ab_np)


def test_permutation():
    a = np.random.randn(2, 3, 4, 5)
    aperm_ncon = ncon(a, [-4, -2, -1, -3])
    aperm_np = np.transpose(a, [2, 1, 3, 0])
    assert np.allclose(aperm_ncon, aperm_np)


def test_trace():
    a = np.random.randn(3, 2, 3)
    atr_ncon = ncon((a,), ([1, -1, 1],))
    atr_np = np.einsum("iji->j", a)
    assert np.allclose(atr_ncon, atr_np)


def test_large_contraction():
    a = np.random.randn(3, 4, 5)
    b = np.random.randn(5, 3, 6, 7, 6)
    c = np.random.randn(7, 2)
    d = np.random.randn(8)
    e = np.random.randn(8, 9)
    result_ncon = ncon(
        (a, b, c, d, e), ([3, -2, 2], [2, 3, 1, 4, 1], [4, -1], [5], [5, -3])
    )
    result_np = np.einsum("ijk,kilml,mh,q,qp->hjp", a, b, c, d, e)
    assert np.allclose(result_ncon, result_np)
