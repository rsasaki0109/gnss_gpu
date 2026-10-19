"""Use the unchanged D270 field body with an external epoch barrier."""

import hashlib
import json
from pathlib import Path
import sys
from types import FunctionType

import bench_rbpf_epoch_clocks_field as base
from rbpf_epoch_lockstep import LockstepClocks

FROZEN_FIELD = "b9ea7d7ada64d0f11f8f4546166bfb3174a9aba6c052769cf72785d3f6db5756"


def main():
    # The digest is checked before running any estimator call.
    path = Path(base.__file__)
    if hashlib.sha256(path.read_bytes()).hexdigest() != FROZEN_FIELD:
        raise ValueError("Frozen three-clock field body changed")
    function = base.main
    scope = dict(function.__globals__, EpochClocks=LockstepClocks)
    run = FunctionType(
        function.__code__,
        scope,
        function.__name__,
        function.__defaults__,
        function.__closure__,
    )
    output = Path(sys.argv[sys.argv.index("--output-dir") + 1])
    code = run()
    (output / "lockstep.json").write_text(
        json.dumps(
            dict(
                exit_code=code,
                unchanged_field_code_object=True,
                only_clock_binding_changed=True,
                frozen_field_sha256=FROZEN_FIELD,
                ready_go_done_advance_outside_process_clocks=True,
            ),
            indent=2,
        )
        + "\n",
        encoding="utf8",
    )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
