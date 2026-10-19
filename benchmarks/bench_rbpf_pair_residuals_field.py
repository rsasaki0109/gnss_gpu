"""One paired accuracy diagnostic of the existing native host score spans.

The completed D284 calibration is never rerun or promoted. Stop after one
instrumented full pair and retain all warm/cold clocks without a speed gate.
"""

import argparse
import json
from pathlib import Path
import sys
from types import FunctionType

import bench_rbpf_inprocess_pair_checked_field as original
from rbpf_epoch_clocks import EpochClocks
from rbpf_inprocess_pair import drive
from rbpf_pair_residuals import ScoreObserver
from rbpf_score_host_phases_v2 import HostPhaseScorer, PHASES


class DiagnosticComplete(Exception):
    pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    plan = original.read(args.plan)
    output = original.ROOT / plan["output"]
    active, scorers = {}, []

    def context():
        found = [(label, len(clock.records)) for label, clock in active.items() if clock.active]
        if len(found) != 1:
            raise RuntimeError("Exactly one owned RTK body must be active")
        return found[0]

    class RecordedScorer(HostPhaseScorer):
        def __init__(self, *args_, **kwargs):
            super().__init__(*args_, **kwargs)
            self.observer = ScoreObserver(context)
            self.score = self.observer.bind(self)
            scorers.append(self)

    def measured_drive(streams, owner, *, pair_index, epochs, after=None, clocks=None):
        assert pair_index == 0 and epochs == 400 and clocks is None and after is not None
        active.update({label: EpochClocks() for label in ("A", "AA")})
        return drive(streams, owner, pair_index=pair_index, epochs=epochs, after=after, clocks=active)

    def finish(path, value):
        original.write(path, value)
        if path == output / "pair0/pair.json":
            raise DiagnosticComplete()

    prior = sys.argv
    sys.argv = [str(Path(original.__file__)), "--plan", str(args.plan)]
    try:
        try:
            FunctionType(original.main.__code__, dict(original.main.__globals__,
                RegisterLUScorer=RecordedScorer, drive=measured_drive, write=finish))()
        except DiagnosticComplete:
            pass
        else:
            raise RuntimeError("Diagnostic must stop after the first complete pair")
    finally:
        sys.argv = prior
    assert len(scorers) == 1 and scorers[0].handle is None
    observer = scorers[0].observer
    rows = observer.records
    assert len(rows) == 394 and len(observer.clocks.records) == 394
    for label in ("A", "AA"):
        selected = [r for r in rows if r["label"] == label]
        assert [r["RTK_epoch"] for r in selected] == list(range(203, 400))
    assert all(r["completed"] is True and r["probe_error"] is None for r in rows)
    assert all(tuple(name for name, _ in r["events"][:-1]) == PHASES for r in rows)
    original.write(output / "score_traces.json", dict(records=rows,
        original_host_profile_AST_restoration_unchanged=True, original_numeric_sources_unchanged=True,
        first_native_call_epoch=203, prefixes=2, RTK_clocks=800, PF_calls=400, score_calls=394,
        both_full_trajectory_and_final_state_exact=True, warm_postprocess_states=400,
        calibration_attempted=False, optimization_speedup_established=False,
        scope="Instrumented diagnostic only:old host span markers plus outer score wall/process/thread clocks. Native span includes ctypes/C API/packing/transfers/kernel/completion;not pure GPU time. Normal GC,one CPU/BLAS1,nonexclusive. Stop after one full pair;no previous failed gate replaced,no epoch exclusion.") )
    print(json.dumps(dict(complete=True, prefixes=2, RTK_clocks=800, score_calls=394,
        accuracy_exact=True, calibration_attempted=False)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
