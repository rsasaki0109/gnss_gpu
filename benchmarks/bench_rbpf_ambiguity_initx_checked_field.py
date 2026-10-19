"""Checked scalar domain;reuse the frozen implementation/measurement body."""

from types import FunctionType

import bench_rbpf_ambiguity_initx_field as base
from rbpf_ambiguity_initx_checked import make_type


def main():
    scope = dict(vars(base), make_type=make_type)
    original = base.CaptureMonitor.instrument_type

    class CheckedCapture(base.CaptureMonitor):
        instrument_type = FunctionType(
            original.__code__,
            scope,
            original.__name__,
            original.__defaults__,
            original.__closure__,
        )

    scope["CaptureMonitor"] = CheckedCapture
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
