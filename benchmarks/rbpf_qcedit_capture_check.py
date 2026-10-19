"""Exact capture values/layouts and mutable alias relationships.

Pickle's memoization of immutable metadata is not a numerical value check.
No tolerance, NaN normalization or field exclusion is used here.
"""

import struct

import numpy as np


def _aliases(value):
    seen, aliases = {}, []

    def visit(item, path):
        mutable = isinstance(item, (dict, list, np.ndarray)) or (
            hasattr(item, "__dict__") and not isinstance(item, int)
        )
        if mutable:
            if id(item) in seen:
                aliases.append((path, seen[id(item)]))
                return
            seen[id(item)] = path
        if isinstance(item, dict):
            for index, (key, val) in enumerate(item.items()):
                visit(key, path + ((index, "key"),))
                visit(val, path + ((index, "value"),))
        elif isinstance(item, (list, tuple)):
            for index, val in enumerate(item):
                visit(val, path + (index,))
        elif hasattr(item, "__dict__") and not isinstance(item, int):
            visit(vars(item), path + ("vars",))

    visit(value, ())
    return aliases


def exact_tree(left, right):
    def compare(a, b):
        assert type(a) is type(b), "type"
        if isinstance(a, np.ndarray):
            assert a.dtype == b.dtype and a.shape == b.shape, "array schema"
            assert a.strides == b.strides, "strides"
            assert (
                a.flags.c_contiguous,
                a.flags.f_contiguous,
                a.flags.writeable,
                a.flags.aligned,
            ) == (
                b.flags.c_contiguous,
                b.flags.f_contiguous,
                b.flags.writeable,
                b.flags.aligned,
            ), "flags"
            assert a.tobytes() == b.tobytes(), "array bits"
        elif isinstance(a, np.generic):
            assert a.dtype == b.dtype and a.tobytes() == b.tobytes(), "scalar bits"
        elif type(a) is float:
            assert struct.pack("d", a) == struct.pack("d", b), "float bits"
        elif isinstance(a, dict):
            assert len(a) == len(b), "dict length"
            for (ak, av), (bk, bv) in zip(a.items(), b.items(), strict=True):
                compare(ak, bk)
                compare(av, bv)
        elif isinstance(a, (list, tuple)):
            assert len(a) == len(b), "sequence length"
            for av, bv in zip(a, b, strict=True):
                compare(av, bv)
        elif hasattr(a, "__dict__") and not isinstance(a, int):
            compare(vars(a), vars(b))
        else:
            assert a == b, "value"

    compare(left, right)
    assert _aliases(left) == _aliases(right), "mutable aliases"
