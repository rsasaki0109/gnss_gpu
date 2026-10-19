"""Checked scalar domain;reuse the frozen implementation/measurement body."""

from types import FunctionType

import bench_rbpf_ambiguity_initx as base
from rbpf_ambiguity_initx_checked import make_type


def main():
    scope = dict(vars(base), make_type=make_type)
    run = FunctionType(
        base.main.__code__,
        scope,
        base.main.__name__,
        base.main.__defaults__,
        base.main.__closure__,
    )
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
