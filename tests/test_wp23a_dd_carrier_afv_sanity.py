"""WP23a Gate G1: unit-level sanity test for the DD carrier AFV likelihood.

Verifies that ``ParticleFilterDevice.update_dd_carrier_afv`` -- the
fractional-cycle, no-ambiguity-resolution-needed likelihood WP23a wires into
the non-hybrid RBPF path (``experiments/exp_ppc_ctrbpf_fgo.py``, method
variant ``rbpf+dd+cp+gate``) -- peaks at the true rover position on a
synthetic, noiseless DD-CP epoch, and decays monotonically moving away from
it within less than half a wavelength (the AFV's own periodicity: beyond
that, per the task spec's documented lambda-spaced multimodality hazard, a
displaced hypothesis can alias back onto a high-likelihood ridge one whole
cycle away -- this test intentionally stays inside the first half-cycle so
"peaks at the true position" is unambiguous).
"""

from types import SimpleNamespace

import numpy as np
import pytest

from gnss_gpu.particle_filter_device import ParticleFilterDevice

try:
    from gnss_gpu._gnss_gpu_pf_device import pf_device_create  # noqa: F401

    HAS_GPU = True
except ImportError:
    HAS_GPU = False

_GPS_L1_WAVELENGTH_M = 0.190293673


def _dd_cycles(rover, sat_k, sat_ref, base_range_k, base_range_ref, wavelength):
    range_k = np.linalg.norm(sat_k - rover[np.newaxis, :], axis=1)
    range_ref = np.linalg.norm(sat_ref - rover[np.newaxis, :], axis=1)
    return ((range_k - range_ref) - (base_range_k - base_range_ref)) / wavelength


def _synthetic_dd_carrier_epoch(n_dd=4, seed=23):
    """Build a noiseless synthetic DD-CP epoch with a known true rover pos.

    Geometry is synthetic (not a real orbit/site) but Earth-scale: rover
    near WGS84 surface radius, satellites at ~GPS orbit radius in random
    directions, base station a few meters from the rover (short baseline).
    """
    rng = np.random.default_rng(seed)
    true_rover = np.array([-3.94e6, 3.35e6, 3.75e6], dtype=np.float64)
    base = true_rover + np.array([5.0, -3.0, 2.0], dtype=np.float64)

    directions = rng.normal(size=(n_dd + 1, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    sat_positions = true_rover[np.newaxis, :] + directions * 2.5e7
    sat_ref = np.tile(sat_positions[0], (n_dd, 1))
    sat_k = sat_positions[1:]

    base_range_k = np.linalg.norm(sat_k - base[np.newaxis, :], axis=1)
    base_range_ref = np.linalg.norm(sat_ref - base[np.newaxis, :], axis=1)
    wavelengths = np.full(n_dd, _GPS_L1_WAVELENGTH_M, dtype=np.float64)
    dd_carrier = _dd_cycles(
        true_rover, sat_k, sat_ref, base_range_k, base_range_ref, _GPS_L1_WAVELENGTH_M
    )

    dd_result = SimpleNamespace(
        sat_ecef_k=sat_k,
        sat_ecef_ref=sat_ref,
        dd_carrier_cycles=dd_carrier,
        base_range_k=base_range_k,
        base_range_ref=base_range_ref,
        dd_weights=np.ones(n_dd, dtype=np.float64),
        wavelengths_m=wavelengths,
        n_dd=n_dd,
    )
    # Line-of-sight direction to the (shared) reference satellite: displacing
    # the rover hypothesis along this axis gives strong, well-conditioned,
    # same-sign DD sensitivity across every pair (all pairs share this ref).
    los_ref = sat_ref[0] - true_rover
    los_ref /= np.linalg.norm(los_ref)
    return true_rover, dd_result, los_ref


@pytest.mark.skipif(not HAS_GPU, reason="CUDA module not available")
def test_dd_carrier_afv_likelihood_peaks_at_true_position():
    true_rover, dd_result, los_ref = _synthetic_dd_carrier_epoch()

    # Offsets stay well inside half a wavelength (~0.0951 m for GPS L1) --
    # and, more precisely, inside every individual DD pair's own half-cycle
    # boundary. Sensitivity of pair j to a rover displacement along
    # ``los_ref`` is ``(1 - cos(theta_j)) / wavelength`` cycles/m, which
    # reaches ~2/wavelength (~10.5 cycles/m) for a pair whose non-ref
    # satellite happens to sit nearly opposite the reference satellite as
    # seen from the rover -- an early version of this test used +/-0.09 m
    # offsets and hit exactly that: a near-antipodal random pair wrapped
    # past its own +/-0.5 cycle boundary well before the nominal
    # half-wavelength limit, producing a real (not spurious) local wiggle
    # -- the lambda-spaced multimodality hazard the WP23a spec itself
    # flags. Kept tight here (worst case ~10.5 cycles/m * 0.02 m = 0.21
    # cycles, safely inside +/-0.5 for every pair) so this test isolates
    # "peaks at the true position", not the multimodality hazard itself.
    offsets_m = np.linspace(-0.02, 0.02, 7)
    positions = true_rover[np.newaxis, :] + offsets_m[:, np.newaxis] * los_ref[np.newaxis, :]
    n_particles = offsets_m.size

    pf = ParticleFilterDevice(n_particles=n_particles, sigma_pos=1.0, sigma_cb=1.0, seed=1)
    pf.initialize(true_rover, clock_bias=0.0, spread_pos=1.0e-6, spread_cb=1.0e-6)
    states = pf.get_particle_states()
    states[:, 0:3] = positions
    states[:, 3] = 0.0
    pf.set_particle_states(states)  # resets log-weights to uniform

    # Looser than the production default (0.05 cycles) so the outermost
    # test offsets stay numerically comfortable in log-space; this is a
    # peak-location sanity check, not a sigma-calibration test.
    pf.update_dd_carrier_afv(dd_result, sigma_cycles=0.2, resample=False)
    log_weights = np.asarray(pf.get_log_weights(), dtype=np.float64)

    assert np.all(np.isfinite(log_weights)), f"non-finite log-weights: {log_weights}"

    peak_idx = int(np.argmax(log_weights))
    assert offsets_m[peak_idx] == pytest.approx(0.0, abs=1.0e-9), (
        f"AFV likelihood peaked at offset {offsets_m[peak_idx]:.4f} m, not "
        f"the true position (0.0 m); offsets={offsets_m}, "
        f"log_weights={log_weights}"
    )
    # Monotonic rise into the peak from the left, monotonic fall away from
    # it to the right -- a single, unambiguous mode at the true position.
    left = log_weights[: peak_idx + 1]
    right = log_weights[peak_idx:]
    assert np.all(np.diff(left) >= -1.0e-9), (
        f"expected weight to rise monotonically into the true position "
        f"from the left; log_weights={log_weights}"
    )
    assert np.all(np.diff(right) <= 1.0e-9), (
        f"expected weight to fall monotonically away from the true "
        f"position to the right; log_weights={log_weights}"
    )
    # The true-position particle should be measurably favored over the
    # farthest test offset (sanity on the magnitude, not just the ranking).
    assert log_weights[peak_idx] - min(log_weights[0], log_weights[-1]) > 0.1
