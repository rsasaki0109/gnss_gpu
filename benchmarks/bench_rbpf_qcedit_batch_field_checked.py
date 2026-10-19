"""Use the final field-domain QC guard in the frozen actual capture runner."""

import hashlib
from pathlib import Path
from types import FunctionType

import bench_rbpf_qcedit_batch_field as base
from rbpf_qcedit_batch_field_schema import QcEngine, make_type

BASE_SHA256 = "a46cceac676d1f3db3d35accc0feb66c5c073b1a71878c0b69b90ae9337b5d6e"


class CaptureMonitor(base.CaptureMonitor):
    original = base.CaptureMonitor.instrument_type
    instrument_type = FunctionType(
        original.__code__,
        dict(original.__globals__, QcEngine=QcEngine, make_type=make_type),
        original.__name__,
        original.__defaults__,
        original.__closure__,
    )


def main():
    if hashlib.sha256(Path(base.__file__).read_bytes()).hexdigest() != BASE_SHA256:
        raise ValueError("Frozen capture runner changed")
    original = base.main
    run = FunctionType(
        original.__code__,
        dict(original.__globals__, CaptureMonitor=CaptureMonitor),
        original.__name__,
        original.__defaults__,
        original.__closure__,
    )
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
