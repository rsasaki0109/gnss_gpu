"""Accuracy-only actual graph/PF run, capture every QC and ambiguity update."""

from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import pickle
import sys
from types import FunctionType

import bench_rbpf_qcedit_batch_field as base
from rbpf_ambiguity_initx_checked import capture_input as ambiguity_input
from rbpf_ambiguity_initx_checked import capture_output as ambiguity_output
from rbpf_preprocess_bundle import make_type

BASE_SHA256 = "a46cceac676d1f3db3d35accc0feb66c5c073b1a71878c0b69b90ae9337b5d6e"


class CaptureMonitor(base.CaptureMonitor):
    def __init__(self, stream, variant, ambiguity_stream):
        super().__init__(stream, variant)
        self.ambiguity_stream = ambiguity_stream
        self.ambiguity_captured = 0

    def instrument_type(self, original):
        bundled, engine = make_type(original)
        chosen = bundled if self.variant == "batch" else original
        self.engines.append(engine)
        monitor = self

        class Capture(chosen):
            def qcedit(self, obs, rs, dts, svh, rr=None):
                capture = getattr(self, "phase", 1) == 2
                if capture:
                    incoming = base.capture_input(self, obs, rs, dts, svh, rr)
                result = super().qcedit(obs, rs, dts, svh, rr)
                if capture:
                    pickle.dump(
                        dict(
                            epoch=int(self.epoch2),
                            input=incoming,
                            output=base.capture_output(self, obs, result),
                        ),
                        monitor.stream,
                        protocol=5,
                    )
                    monitor.captured += 1
                return result

            def update_ambiguities(self, obs):
                capture = getattr(self, "phase", 1) == 2
                if capture:
                    incoming = ambiguity_input(self, obs)
                    obs_before = deepcopy(vars(obs))
                result = super().update_ambiguities(obs)
                if capture:
                    pickle.dump(
                        dict(
                            epoch=int(self.epoch2),
                            input=incoming,
                            output=ambiguity_output(self),
                            obs_before=obs_before,
                            obs_after=deepcopy(vars(obs)),
                        ),
                        monitor.ambiguity_stream,
                        protocol=5,
                    )
                    monitor.ambiguity_captured += 1
                return result

        return Capture


def main():
    if hashlib.sha256(Path(base.__file__).read_bytes()).hexdigest() != BASE_SHA256:
        raise ValueError("Frozen actual field runner changed")
    output = Path(sys.argv[sys.argv.index("--output-dir") + 1])
    path = output.with_suffix(".ambiguity.pkl.gz")
    assert not path.exists()
    monitors = []
    with gzip.open(path, "wb") as stream:

        def selected(qc_stream, variant):
            monitor = CaptureMonitor(qc_stream, variant, stream)
            monitors.append(monitor)
            return monitor

        original = base.main
        run = FunctionType(
            original.__code__,
            dict(original.__globals__, CaptureMonitor=selected),
            original.__name__,
            original.__defaults__,
            original.__closure__,
        )
        code = run()
    assert code == 0 and len(monitors) == 1
    monitor = monitors[0]
    assert monitor.captured == 400 and monitor.ambiguity_captured == 200
    output.with_suffix(".bundle.json").write_text(
        json.dumps(
            dict(
                variant_batch_selects="Frozen D247 checked ambiguity bookkeeping plus D268 field QC",
                captured_phase2_QC_calls=400,
                captured_phase2_ambiguity_updates=200,
                ambiguity_capture_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                engines=[e.details() for e in monitor.engines],
                original_numeric_helpers_and_PF_unchanged=True,
                diagnostic_only_no_speed_claim=True,
            ),
            indent=2,
        )
        + "\n",
        encoding="utf8",
    )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
