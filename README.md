# ncon
[![][travis-img]][travis-url] [![][codecov-img]][codecov-url]

ncon is a Python 3 package that implements the NCon function as described here:
https://arxiv.org/abs/1402.0939
This Python implementation lacks some of the fancier features described in
the paper, but the interface is the same.

ncon requires numpy and works with numpy ndarrays. It also works with the
various tensors from [this](https://github.com/mhauru/abeliantensors) package,
but does not require it.

## Installation

`pip install --user ncon`

## Usage

The only thing this package exports is the function `ncon`. It takes a list of
tensors to be contracted, and a list index lists that specify what gets
contracted with that. It returns a single tensor, that is the result of the
contraction. Here's how the syntax works:
```
ncon(L, v, order=None, forder=None, check_indices=True):
```
The first argument `L` is a list of tensors.
The second argument `v` is a list of list, one for each tensor in `L`.
Each `v[i]` consists of integers, each of which labels an index of `L[i]`.
Positive labels mark indices which are to be contracted (summed over).
So if for instance `v[m][i] == 2` and `v[n][j] == 2`, then the `i`th index of
`L[m]` and the `j`th index of `L[n]` are to be identified and summed over.
Negative labels mark indices which are to remain free (uncontracted).

The keyword argument `order` is a list of all the positive labels, which
specifies the order in which the pair-wise tensor contractions are to be done.
By default it is `sorted(all-positive-numbers-in-v)`, so for instance
`[1,2,...]`. Note that whenever an index joining two tensors is about to be
contracted together, `ncon` contracts at the same time all indices connecting
these two tensors, even if some of them only come up later in order.

Correspondingly `forder` specifies the order to which the remaining free
indices are to be permuted. By default it is
`sorted(all-negative-numbers-in-v, reverse=True)`,
meaning for instance `[-1,-2,...]`.

If `check_indices=True` (the default) then checks are performed to make sure
the contraction is well-defined. If not, an `ValueError` with a helpful
description of what went wrong is provided.

If the syntax sounds a lot like Einstein summation, as implemented for example
by `np.einsum`, then that's because it is. The benefits of `ncon` are that many
tensor networkers are used to its syntax, and it is easy to dynamically
generate index lists and contractions.

#### Examples

Here's a few examples, straight from the test file.

A matrix product:
```
from ncon import ncon
a = np.random.randn(3, 4)
b = np.random.randn(4, 5)
ab_ncon = ncon([a, b], ((-1, 1), (1, -2)))
ab_np = np.dot(a, b)
assert np.allclose(ab_ncon, ab_np)
```
Here the last index of `a` and the first index of `b` are contracted.
The result is a tensor with two free indices, labeled by `-1` and `-2`.
The one labeled with `-1` becomes the first index of the result. If we gave the
additional argument `forder=[-2,-1]` the tranpose would be returned instead.

A more complicated example:
```a = np.random.randn(3, 4, 5)
b = np.random.randn(5, 3, 6, 7, 6)
c = np.random.randn(7, 2)
d = np.random.randn(8)
e = np.random.randn(8, 9)
result_ncon = ncon(
    (a, b, c, d, e), ([3, -2, 2], [2, 3, 1, 4, 1], [4, -1], [5], [5, -3])
)
result_np = np.einsum("ijk,kilml,mh,q,qp->hjp", a, b, c, d, e)
assert np.allclose(result_ncon, result_np)
```
Notice that the network here is disconnected, `d` and `e` are not contracted
with any of the others. When contracting disconnected networks, the connected
parts are always contracted first, and their tensor product is taken at the
end. Traces are also okay, like here on two indices of `c`. By default, the
contractions are done in the order [1,2,3,4,5]. This may not be the optimal
choice, in which case we should specify a better contraction order as a keyword
argument.

[travis-img]: https://travis-ci.org/mhauru/ncon.svg?branch=master
[travis-url]: https://travis-ci.org/mhauru/ncon
[codecov-img]: https://codecov.io/gh/mhauru/ncon/branch/master/graph/badge.svg
[codecov-url]: https://codecov.io/gh/mhauru/ncon
