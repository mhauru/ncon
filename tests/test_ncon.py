import numpy as np
import pytest
from ncon import ncon

# The different calls deliberately use different variation of things like are the
# arguments lists or tuples.


def test_matrixproduct():
    a = np.random.randn(3, 4)
    b = np.random.randn(4, 5)
    ab_ncon = ncon([a, b], ((-1, 1), (1, -2)))
    ab_np = np.dot(a, b)
    assert np.allclose(ab_ncon, ab_np)
    ab_ncon_str = ncon(
        [a, b], (("a", "b"), ("b", "c")), order=["b"], forder=["a", "c"]
    )
    assert np.allclose(ab_ncon_str, ab_np)


def test_disconnected():
    a = np.random.randn(2, 3)
    b = np.random.randn(4)
    ab_ncon = ncon((a, b), ([-3, -2], [-1]))
    ab_np = np.einsum("ij, k -> kji", a, b)
    assert np.allclose(ab_ncon, ab_np)
    ab_ncon_str = ncon(
        (a, b),
        (["x", 10023], ["blahblah"]),
        order=[],
        forder=["blahblah", 10023, "x"],
    )
    assert np.allclose(ab_ncon_str, ab_np)


def test_permutation():
    a = np.random.randn(2, 3, 4, 5)
    aperm_ncon = ncon(a, [-4, -2, -1, -3])
    aperm_np = np.transpose(a, [2, 1, 3, 0])
    assert np.allclose(aperm_ncon, aperm_np)
    aperm_ncon_str = ncon(
        a, ["4", "2", "1", "3"], order=[], forder=["1", "2", "3", "4"]
    )
    assert np.allclose(aperm_ncon_str, aperm_np)


def test_trace():
    a = np.random.randn(3, 2, 3)
    atr_ncon = ncon((a,), ([1, -1, 1],))
    atr_np = np.einsum("iji -> j", a)
    assert np.allclose(atr_ncon, atr_np)
    atr_ncon_str = ncon(
        (a,),
        [["traced", "not traced", "traced"]],
        order=["traced"],
        forder=["not traced"],
    )
    assert np.allclose(atr_ncon_str, atr_np)


def test_large_contraction():
    a = np.random.randn(3, 4, 5)
    b = np.random.randn(5, 3, 6, 7, 6)
    c = np.random.randn(7, 2)
    d = np.random.randn(8)
    e = np.random.randn(8, 9)
    result_ncon = ncon(
        (a, b, c, d, e), ([3, -2, 2], [2, 3, 1, 4, 1], [4, -1], [5], [5, -3])
    )
    result_np = np.einsum("ijk, kilml, mh, q, qp -> hjp", a, b, c, d, e)
    assert np.allclose(result_ncon, result_np)
    result_ncon_str = ncon(
        (a, b, c, d, e),
        (
            ["3", "-2", "2"],
            ["2", "3", "1", "4", "1"],
            ["4", "-1"],
            ["5"],
            ["5", "-3"],
        ),
        order=("1", "2", "3", "4", "5"),
        forder=("-1", "-2", "-3"),
    )
    assert np.allclose(result_ncon_str, result_np)


def test_missing_order():
    a = np.random.randn(3, 4)
    b = np.random.randn(4, 5)
    with pytest.raises(ValueError):
        ncon([a, b], (("a", "b"), ("b", "c")), forder=["a", "c"])


def test_missing_forder():
    a = np.random.randn(3, 4)
    b = np.random.randn(4, 5)
    with pytest.raises(ValueError):
        ncon([a, b], (("a", "b"), ("b", "c")), order=["b"])


def test_faulty_indices():
    a = np.random.randn(3, 4)
    b = np.random.randn(4, 5)
    c = np.random.randn(5, 6)
    ab_ncon = ncon([a, b, c], ((-1, 1), (1, 2), (2, -2)))
    ab_np = np.einsum("ij, jk, kl->il", a, b, c)
    assert np.allclose(ab_ncon, ab_np)
    with pytest.raises(ValueError):
        ncon([a, b, c], ((-1, 1), (2, 2), (2, -2)))
    with pytest.raises(ValueError):
        ncon([a, b, c], ((-1, 1), (1, 2), (1, -2)))
    with pytest.raises(ValueError):
        ncon([a, b, c], ((-1, 3), (1, 2), (1, -2)))
    with pytest.raises(ValueError):
        ncon([a, b, c], ((-1, 1), (1, 2), (2, 3)))
    with pytest.raises(ValueError):
        ncon([a, b, c], ((-1, 1), (1, 2), (2, -1)))
    with pytest.raises(ValueError):
        ncon([a, b, c], ((-1, 1), (1, 2), (2, -2)), order=[1, 2, 3])
    with pytest.raises(ValueError):
        ncon([a, b, c], ((-1, 1), (1, 2), (2, -2)), order=[2])
    with pytest.raises(ValueError):
        ncon([a, b, c], ((-1, 1), (1, 2), (2, -2)), forder=[-2])
    with pytest.raises(ValueError):
        ncon([a, b, c], ((-1, 1), (1, 2), (2, -2)), forder=[-2, -1, -3])
    with pytest.raises(ValueError):
        ncon([a, b, c], (("-1", 1), (1, 2), (2, -2)))
    with pytest.raises(ValueError):
        ncon([a, b, c], ((-1, 1), (2, -2)))
    with pytest.raises(ValueError):
        ncon([a, b, c], ((-1, 1), (1, 2), (3,), (2, -2)))
    with pytest.raises(ValueError):
        ncon([a, b, c], ((-1, 1), (2, 1), (2, -2)))
    with pytest.raises(ValueError):
        ncon([a, b, c], ("aaaa", (1, 2), (2, -2)))
    with pytest.raises(ValueError):
        ncon([a, b, c], (-1, (1, 2), (2, -2)))
