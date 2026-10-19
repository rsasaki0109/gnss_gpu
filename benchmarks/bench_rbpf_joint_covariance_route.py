"""Frozen complete RTK timing body with explicit covariance-writer choice."""

import hashlib
import json
import os
from pathlib import Path
import sys
from types import FunctionType

import bench_rbpf_epoch_clocks_field as base
from rbpf_epoch_lockstep import LockstepClocks
from rbpf_epoch_clocks import EpochClocks
from rbpf_joint_covariance_field_schema import make_type

FROZEN_FIELD = "b9ea7d7ada64d0f11f8f4546166bfb3174a9aba6c052769cf72785d3f6db5756"


class CovarianceClocks(LockstepClocks):
    def __init__(self):
        super().__init__()
        self.lockstep = os.environ.get("GNSS_GPU_JOINT_LOCKSTEP", "1") == "1"
        self.variant = os.environ["GNSS_GPU_JOINT_VARIANT"]
        if self.variant not in ("reference", "bulk"):
            raise ValueError("Explicit covariance variant required")
        self.engines = []

    def call(self, phase, function, *args, **kwargs):
        if self.lockstep:
            return super().call(phase, function, *args, **kwargs)
        return EpochClocks.call(self, phase, function, *args, **kwargs)

    def instrument_type(self, original):
        chosen = original
        if self.variant == "bulk":
            chosen, engine = make_type(original)
            self.engines.append(engine)
        return super().instrument_type(chosen)


def main():
    if hashlib.sha256(Path(base.__file__).read_bytes()).hexdigest() != FROZEN_FIELD:
        raise ValueError("Frozen complete process body changed")
    monitor = CovarianceClocks()
    function = base.main
    run = FunctionType(
        function.__code__,
        dict(function.__globals__, EpochClocks=lambda: monitor),
        function.__name__,
        function.__defaults__,
        function.__closure__,
    )
    output = Path(sys.argv[sys.argv.index("--output-dir") + 1])
    code = run()
    for engine in monitor.engines:
        assert engine.full_matrix_calls > 0 and engine.fallback_calls == 0
    (output / "lockstep.json").write_text(
        json.dumps(
            dict(
                exit_code=code,
                unchanged_field_code_object=True,
                only_clock_binding_changed=True,
                frozen_field_sha256=FROZEN_FIELD,
                ready_go_done_advance_outside_process_clocks=monitor.lockstep,
                variant=monitor.variant,
                joint_covariance=[
                    dict(
                        counts=e.counts(),
                        small_literal_calls=e.small_literal_calls,
                        AST_proofs=e.proofs,
                    )
                    for e in monitor.engines
                ],
            ),
            indent=2,
        )
        + "\n",
        encoding="utf8",
    )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
