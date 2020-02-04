# ncon
[![][travis-img]][travis-url] [![][codecov-img]][codecov-url]

ncon is a Python 3 package that implements the NCon function as described here:
https://arxiv.org/abs/1402.0939
This Python implementation lacks some of the fancier features described in
the paper, but the interface is the same.

ncon works with either numpy ndarrays, or the various tensor classes of this
package:
https://github.com/mhauru/tensors

For usage instructions, check either the paper, or the Julia version of the
same function:
https://github.com/mhauru/NCon.jl

Also:
```
def ncon(AA, v, order=None, forder=None, check_indices=True):
    """ AA = [A1, A2, ..., Ap] list of tensors.

    v = (v1, v2, ..., vp) tuple of lists of indices e.g. v1 = [3, 4, -1] labels
    the three indices of tensor A1, with -1 indicating an uncontracted index
    (open leg) and 3 and 4 being the contracted indices.

    order, if present, contains a list of all positive indices - if not
    [1, 2, 3, 4, ...] by default. This is the order in which they are
    contracted.

    forder, if present, contains the final ordering of the uncontracted indices
    - if not, [-1, -2, ..i] by default.

    There is some leeway in the way the inputs are given. For example,
    instead of giving a list of tensors as the first argument one can
    give some different iterable of tensors, such as a tuple, or a
    single tensor by itself (anything that has the attribute "shape"
    will be considered a tensor).
    """

```

[travis-img]: https://travis-ci.org/mhauru/ncon.svg?branch=master
[travis-url]: https://travis-ci.org/mhauru/ncon
[codecov-img]: https://codecov.io/gh/mhauru/ncon/branch/master/graph/badge.svg
[codecov-url]: https://codecov.io/gh/mhauru/ncon
