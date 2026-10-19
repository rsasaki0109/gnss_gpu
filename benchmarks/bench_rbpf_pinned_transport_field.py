"""One pinned-transport full RTK accuracy pair using frozen diagnostic driver."""

from types import FunctionType

import bench_rbpf_native_call_trace_field as original
from rbpf_pinned_transport import TracedPinnedTransportScorer


def main():
    class Selected(TracedPinnedTransportScorer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, transport="pinned", **kwargs)

    return FunctionType(original.main.__code__, dict(original.main.__globals__, NativeCallScorer=Selected))()


if __name__ == "__main__":
    raise SystemExit(main())
