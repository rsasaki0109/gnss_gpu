"""Native host-clock diagnostic in one full independent RTK pair,no profiler."""

import argparse
from pathlib import Path
from types import FunctionType

import bench_rbpf_pair_residuals_field as field
from rbpf_native_call_trace import NativeCallScorer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    plan = field.original.read(args.plan)
    scorers = []

    class Selected(NativeCallScorer):
        def __init__(self, *args_, **kwargs):
            super().__init__(*args_, **kwargs)
            scorers.append(self)

    code = FunctionType(field.main.__code__, dict(field.main.__globals__, HostPhaseScorer=Selected))()
    assert code == 0 and len(scorers) == 1 and scorers[0].handle is None
    scorer = scorers[0]
    assert len(scorer.native_records) == 394
    outer = scorer.observer.records
    result = []
    for record, score in zip(scorer.native_records, outer, strict=True):
        assert record["probe_error"] is None and record["trace"]["full_GPU_path"]
        trace = record["trace"]
        assert trace["totals"]["wall_ticks"] * 10**9 <= score["partition"]["native"] * trace["frequency"]
        result.append(dict(record, label=score["label"], RTK_epoch=score["RTK_epoch"],
            Python_native_span_ns=score["partition"]["native"]))
    field.original.write(field.original.ROOT / plan["output"] / "native_traces.json", dict(
        verified=True, records=result, score_calls=394, no_external_profiler=True,
        new_CUDA_operations=0, numerical_return_objects_unchanged=True,
        scope="One instrumented independent-driver pair. QPC/GetThreadTimes markers only,no CUDA events/profiler/extra synchronization. Normal GC/model/seed/GPU kernel math unchanged. Diagnostic only,no completed calibration/optimization retime or speed acceptance."))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
