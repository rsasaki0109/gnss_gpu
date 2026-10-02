"""Short-segment regression for the BVH-accelerated 3D particle filter.

A receiver drives 40 epochs down a synthetic street canyon (two rows of box
buildings). Blocked satellites get a positive NLOS bias, computed with the
linear-scan ray tracer. The test pins three properties of the GPU pipeline:

* the BVH filter reproduces the linear-scan ParticleFilter3D trajectory,
* a fixed seed gives a bit-identical trajectory,
* the 2D error stays bounded (10.6 m RMS on the reference GPU).

It deliberately does not assert that PF3D beats the 3D-unaware filter: with
these untuned parameters it does not (see internal_docs/decisions.md D-037).

Requires CUDA and the compiled raytrace / bvh / pf / pf3d / pf3d_bvh modules;
runs in the self-hosted CUDA workflow.
"""

import numpy as np
import pytest

try:
    from gnss_gpu._bvh import bvh_build  # noqa: F401
    from gnss_gpu._gnss_gpu_pf import pf_initialize  # noqa: F401
    from gnss_gpu._gnss_gpu_pf3d import pf_weight_3d  # noqa: F401
    from gnss_gpu._gnss_gpu_pf3d_bvh import pf_weight_3d_bvh  # noqa: F401
    from gnss_gpu._raytrace import raytrace_los_check  # noqa: F401
    from gnss_gpu.bvh import BVHAccelerator
    from gnss_gpu.particle_filter_3d import ParticleFilter3D
    from gnss_gpu.particle_filter_3d_bvh import ParticleFilter3DBVH
    from gnss_gpu.raytrace import BuildingModel
    HAS_GPU = True
except ImportError:
    HAS_GPU = False

pytestmark = pytest.mark.skipif(not HAS_GPU, reason="CUDA module not available")

SEED = 7
N_PARTICLES = 20_000
N_EPOCHS = 40
WARMUP_EPOCHS = 5
CLOCK_BIAS_M = 50.0
VELOCITY = np.array([4.0, 0.0, 0.0])
RMS_2D_LIMIT_M = 15.0


def _street_canyon():
    boxes = [
        BuildingModel.create_box(center=[x + 15.0, y, 20.0], width=30.0, depth=15.0, height=40.0).triangles
        for x in np.arange(0.0, 200.0, 40.0)
        for y in (-22.0, 22.0)
    ]
    return BuildingModel(np.concatenate(boxes, axis=0))


def _scenario(building):
    rng = np.random.default_rng(SEED)
    el = np.radians([75, 60, 45, 35, 30, 25, 20, 15])
    az = np.radians([10, 100, 190, 280, 45, 135, 225, 315])
    dirs = np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], axis=1)
    truth = np.array([[10.0 + 4.0 * k, 0.0, 1.5] for k in range(N_EPOCHS)])
    sats = truth[0] + 2.0e7 * dirs
    pseudoranges = []
    n_blocked = 0
    for k in range(N_EPOCHS):
        los = building.check_los(truth[k], sats)
        n_blocked += int((~los).sum())
        nlos_bias = np.where(los, 0.0, 25.0 + np.abs(rng.normal(0.0, 10.0, len(sats))))
        noise = rng.normal(0.0, 3.0, len(sats))
        pseudoranges.append(np.linalg.norm(sats - truth[k], axis=1) + CLOCK_BIAS_M + noise + nlos_bias)
    return truth, sats, pseudoranges, n_blocked


def _run(pf, truth, sats, pseudoranges):
    pf.initialize(truth[0] + np.array([5.0, -3.0, 0.0]), clock_bias=CLOCK_BIAS_M, spread_pos=10.0, spread_cb=10.0)
    estimates = []
    for k in range(N_EPOCHS):
        if k:
            pf.predict(velocity=VELOCITY, dt=1.0)
        pf.update(sats, pseudoranges[k])
        estimates.append(pf.estimate()[:3])
    return np.asarray(estimates)


def _filter_kwargs():
    return dict(n_particles=N_PARTICLES, sigma_pos=1.0, sigma_cb=5.0,
                sigma_los=3.0, sigma_nlos=30.0, nlos_bias=25.0, seed=SEED)


@pytest.fixture(scope="module")
def segment():
    building = _street_canyon()
    truth, sats, pseudoranges, n_blocked = _scenario(building)
    bvh = BVHAccelerator.from_building_model(building)
    bvh_est = _run(ParticleFilter3DBVH(bvh, **_filter_kwargs()), truth, sats, pseudoranges)
    return building, bvh, truth, sats, pseudoranges, n_blocked, bvh_est


def test_scenario_exercises_nlos(segment):
    *_, n_blocked, _ = segment
    # Roughly two thirds of the 320 observations are blocked by the canyon.
    assert 150 <= n_blocked <= 260


def test_bvh_matches_linear_scan_over_segment(segment):
    building, _, truth, sats, pseudoranges, _, bvh_est = segment
    linear_est = _run(ParticleFilter3D(building, **_filter_kwargs()), truth, sats, pseudoranges)
    np.testing.assert_allclose(bvh_est, linear_est, rtol=0.0, atol=1e-6)


def test_fixed_seed_is_deterministic(segment):
    _, bvh, truth, sats, pseudoranges, _, bvh_est = segment
    rerun = _run(ParticleFilter3DBVH(bvh, **_filter_kwargs()), truth, sats, pseudoranges)
    np.testing.assert_array_equal(rerun, bvh_est)


def test_error_stays_bounded(segment):
    *_, truth, _, _, _, bvh_est = segment
    err_2d = np.linalg.norm(bvh_est[WARMUP_EPOCHS:, :2] - truth[WARMUP_EPOCHS:, :2], axis=1)
    assert np.all(np.isfinite(bvh_est))
    assert float(np.sqrt(np.mean(err_2d**2))) < RMS_2D_LIMIT_M
