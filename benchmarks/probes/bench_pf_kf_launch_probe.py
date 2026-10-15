"""Reproduce preliminary KF candidates; run sequentially from the repo root."""

from pathlib import Path
import sys
import json
import numpy as np

sys.path.insert(0, "benchmarks")
from bench_pf_coalesced_compare import _initialize, _load_extension, _scenario
from bench_pf_covariance_compare import _digest, _snapshot
from bench_pf_doppler_compare import _validate, _kf, _observations
from bench_pf_reduction_compare import _measure

native = _load_extension(Path(sys.argv[1]).resolve())
records = _validate(native)
for count in (1, 33, 257):
    rng = np.random.default_rng(count)
    original = rng.normal(size=(count, 16))
    original[:, :3] += _scenario()[0]
    prior = rng.normal(size=count)
    state = native.pf_device_create(count)
    try:
        for variant in (
            "nonsymmetric",
            "zero",
            "nonfinite",
            "signed_zero",
            "large",
            "tiny",
        ):
            initial = original.copy()
            initial[:, 7:] = rng.normal(size=(count, 9))
            if variant == "zero":
                initial[:, 7:] = 0.0
            elif variant == "nonfinite":
                initial[:, 7:] = np.resize([np.nan, np.inf, -np.inf, -0.0], count)[
                    :, None
                ]
            elif variant == "signed_zero":
                initial[:, 7:] = [4.0, 0.0, -0.0, -0.0, 3.0, -0.0, 0.0, 0.0, 2.0]
            elif variant == "large":
                initial[:, 7:] *= 1e150
            elif variant == "tiny":
                initial[:, 7:] *= 1e-300
            for n_sat in (1, 2, 4, 12, 16, 17, 32, 33, 65):
                native.pf_device_set_particle_states(state, initial)
                native.pf_device_set_log_weights(state, prior)
                for _ in range(2):
                    _kf(native, state, _observations(n_sat))
                records[f"N{count}/sat{n_sat}/extra_{variant}"] = _digest(
                    *_snapshot(native, state)
                )
    finally:
        native.pf_device_destroy(state)
timing = {}
_, sat, pr, weights = _scenario()
for count in (2048, 100000, 1000000):
    state = native.pf_device_create(count)
    try:
        for n_sat in (4, 12, 16, 32, 65):
            observations = _observations(n_sat)
            timing[f"N{count}/KF{n_sat}"] = _measure(
                native,
                state,
                lambda: _initialize(native, state),
                lambda: _kf(native, state, observations),
                20,
            )
        observations = _observations(12)

        def step():
            native.pf_device_predict(
                state, 1.0, 2.0, 3.0, 0.1, 1.0, 3.0, 42, 1, 0.0, 0.3, True, 0.1
            )
            native.pf_device_weight(state, sat, pr, weights, 12, 5.0)
            _kf(native, state, observations)
            ess = native.pf_device_ess(state)
            resampled = ess < 0.5 * count
            if resampled:
                native.pf_device_resample_megopolis_coalesced(state, 60, 42)
            return np.array([ess, resampled, *native.pf_device_estimate(state)])

        timing[f"N{count}/PRKF"] = _measure(
            native, state, lambda: _initialize(native, state), step, 20
        )
    finally:
        native.pf_device_destroy(state)
Path(sys.argv[2]).write_text(
    json.dumps(dict(validation=records, measurements=timing)), encoding="utf-8"
)
