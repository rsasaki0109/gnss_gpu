"""Prediction laws, captured guides and asynchronous state ownership."""

import numpy as np
import pytest

native = pytest.importorskip("gnss_gpu._gnss_gpu_pf_device")
pytestmark = [pytest.mark.cuda, pytest.mark.gpu]


def _cloud(count):
    rng = np.random.default_rng(count)
    values = rng.normal(size=(count, 16))
    factors = rng.normal(size=(count, 3, 3))
    values[:, 7:] = (factors @ factors.swapaxes(1, 2) + np.eye(3) * .5).reshape(count, 9)
    return values, rng.normal(size=count)


def _read(state):
    weights = np.empty(state.n_particles)
    native.pf_device_get_log_weights(state, weights)
    return native.pf_device_get_particle_states(state), weights


def _predict(state, guide=(.71, -2.3, 3.14159), *, dt=.17, sigma_pos=1.3, sigma_cb=2.7,
             sigma_vel=0., alpha=.3, kf=False, q=.13, step=13):
    native.pf_device_predict(state, *guide, dt, sigma_pos, sigma_cb, 2**64 - 1, step,
                             sigma_vel, alpha, kf, q)


@pytest.mark.parametrize("count", [1, 33, 257, 8193])
@pytest.mark.parametrize("alpha", [-1., 0., .3, 1., 2.])
@pytest.mark.parametrize("mode", ["default", "negative_sigma", "KF"])
def test_zero_dt_velocity_blend_and_untouched_covariance(count, alpha, mode):
    before, prior = _cloud(count)
    guide = np.array([.71, -2.3, 3.14159])
    coefficient = np.clip(alpha, 0., 1.)
    expected = (1. - coefficient) * before[:, 4:7] + coefficient * guide
    state = native.pf_device_create(count)
    try:
        native.pf_device_set_particle_states(state, before)
        native.pf_device_set_log_weights(state, prior)
        _predict(state, guide, dt=0., sigma_pos=1e-300, sigma_cb=1e-300,
                 sigma_vel=.2 if mode == "KF" else -1. if mode == "negative_sigma" else 0.,
                 alpha=alpha, kf=mode == "KF")
        actual, weights = _read(state)
        np.testing.assert_allclose(actual[:, 4:7], expected, rtol=2e-15, atol=2e-15)
        np.testing.assert_array_equal(actual[:, :4].view(np.uint64), before[:, :4].view(np.uint64))
        np.testing.assert_array_equal(actual[:, 7:].view(np.uint64), before[:, 7:].view(np.uint64))
        np.testing.assert_array_equal(weights.view(np.uint64), prior.view(np.uint64))
    finally:
        native.pf_device_destroy(state)


@pytest.mark.parametrize("mode", ["default", "noisy", "KF"])
def test_prediction_noise_matches_independent_distribution_moments(mode):
    count = 65537
    before = np.zeros((count, 16))
    before[:, 4:7] = [3., -2., 1.5]
    covariance = np.array([[4., 1., .5], [1., 3., .25], [.5, .25, 2.]])
    before[:, 7:] = covariance.ravel()
    guide = np.array([.71, -2.3, 3.14159])
    velocity = .7 * before[0, 4:7] + .3 * guide
    dt, position_sigma, clock_sigma, velocity_sigma = .2, .1, 2., .4
    expected_cov = np.eye(3) * position_sigma**2
    if mode == "noisy":
        expected_cov += np.eye(3) * (dt * velocity_sigma)**2
    elif mode == "KF":
        expected_cov += dt**2 * covariance
    state = native.pf_device_create(count)
    try:
        native.pf_device_set_particle_states(state, before)
        _predict(state, guide, dt=dt, sigma_pos=position_sigma, sigma_cb=clock_sigma,
                 sigma_vel=velocity_sigma if mode != "default" else 0., kf=mode == "KF")
        actual, weights = _read(state)
        residual = actual[:, :3] - dt * velocity
        whitened = np.linalg.solve(np.linalg.cholesky(expected_cov), residual.T).T
        np.testing.assert_allclose(whitened.mean(axis=0), 0., atol=.025)
        np.testing.assert_allclose(np.cov(whitened, rowvar=False), np.eye(3), atol=.025)
        assert abs(actual[:, 3].mean() / clock_sigma) < .025
        assert abs(actual[:, 3].var() / clock_sigma**2 - 1.) < .025
        if mode == "noisy":
            velocity_noise = (actual[:, 4:7] - velocity) / velocity_sigma
            np.testing.assert_allclose(velocity_noise.mean(axis=0), 0., atol=.025)
            np.testing.assert_allclose(np.cov(velocity_noise, rowvar=False), np.eye(3), atol=.025)
            cross = residual.T @ (actual[:, 4:7] - velocity) / count
            np.testing.assert_allclose(cross, np.eye(3) * dt * velocity_sigma**2, atol=.003)
        else:
            np.testing.assert_allclose(actual[:, 4:7], np.broadcast_to(velocity, (count, 3)), atol=2e-15)
        expected_velocity_cov = covariance + (np.eye(3) * .13 * dt if mode == "KF" else 0.)
        np.testing.assert_allclose(actual[:, 7:], np.broadcast_to(expected_velocity_cov.ravel(), (count, 9)),
                                   rtol=2e-15, atol=2e-15)
        np.testing.assert_array_equal(weights, np.zeros(count))
    finally:
        native.pf_device_destroy(state)


@pytest.mark.parametrize("count", [33, 257, 100003])
@pytest.mark.parametrize("public", [False, True])
def test_queued_changing_guides_match_synchronized_reference(count, public):
    from gnss_gpu.particle_filter_device import ParticleFilterDevice

    before, prior = _cloud(count)
    owners = [ParticleFilterDevice(n_particles=count, seed=42) for _ in range(2)] if public else []
    states = [owner._state for owner in owners] if public else [native.pf_device_create(count) for _ in range(2)]
    try:
        for index, state in enumerate(states):
            if public:
                owners[index].initialize([0., 0., 0.])
            native.pf_device_set_particle_states(state, before)
            native.pf_device_set_log_weights(state, prior)
        saved = _read(states[0])
        saved_copy = [array.copy() for array in saved]
        for index, state in enumerate(states):
            for step in range(1, 49):
                guide = [np.sin(step), -.37 * step, 2. / step]
                kf, sigma_vel = step % 3 == 0, .2 if step % 3 == 1 else 0.
                if public:
                    owners[index].predict(guide, dt=.17, sigma_pos=1.3, sigma_vel=sigma_vel,
                                          velocity_guide_alpha=.3, rbpf_velocity_kf=kf,
                                          velocity_process_noise=.13)
                else:
                    _predict(state, guide, step=step, sigma_vel=sigma_vel, kf=kf)
                if index == 1:
                    native.pf_device_sync(state)
                # The launch must own its values after the caller mutates the guide.
                guide[:] = [999., 999., 999.]
        actual, reference = [_read(state) for state in states]
        for array, expected in zip(actual, reference):
            np.testing.assert_array_equal(array.view(np.uint64), expected.view(np.uint64))
        np.testing.assert_array_equal(actual[1].view(np.uint64), prior.view(np.uint64))
        for array, expected in zip(saved, saved_copy):
            np.testing.assert_array_equal(array.view(np.uint64), expected.view(np.uint64))
    finally:
        for state in states:
            native.pf_device_destroy(state)


@pytest.mark.parametrize("mode", ["default", "noisy", "KF"])
@pytest.mark.parametrize("count", [33, 257])
def test_fixed_seed_replay_and_independent_returned_arrays(count, mode):
    before, prior = _cloud(count)
    state = native.pf_device_create(count)
    try:
        outputs = []
        for _ in range(2):
            native.pf_device_set_particle_states(state, before)
            native.pf_device_set_log_weights(state, prior)
            _predict(state, sigma_vel=.2 if mode == "noisy" else 0., kf=mode == "KF")
            outputs.append(_read(state))
        for actual, expected in zip(*outputs):
            np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))
        np.testing.assert_array_equal(outputs[0][1].view(np.uint64), prior.view(np.uint64))
        if mode != "KF":
            np.testing.assert_array_equal(outputs[0][0][:, 7:].view(np.uint64), before[:, 7:].view(np.uint64))
    finally:
        native.pf_device_destroy(state)


@pytest.mark.parametrize("kf", [False, True])
def test_nonfinite_prior_velocity_still_participates_in_alpha_one_blend(kf):
    before, prior = _cloud(33)
    before[:, 4:7] = [np.nan, np.inf, -np.inf]
    state = native.pf_device_create(33)
    try:
        native.pf_device_set_particle_states(state, before)
        native.pf_device_set_log_weights(state, prior)
        _predict(state, alpha=1., kf=kf)
        actual, weights = _read(state)
        assert np.isnan(actual[:, :3]).all()
        assert np.isnan(actual[:, 4:7]).all()
        np.testing.assert_array_equal(weights.view(np.uint64), prior.view(np.uint64))
    finally:
        native.pf_device_destroy(state)


def test_negative_process_noise_is_clamped_without_changing_covariance():
    before, _ = _cloud(257)
    state = native.pf_device_create(257)
    try:
        native.pf_device_set_particle_states(state, before)
        _predict(state, kf=True, q=-1.)
        actual, _ = _read(state)
        np.testing.assert_array_equal(actual[:, 7:].view(np.uint64), before[:, 7:].view(np.uint64))
    finally:
        native.pf_device_destroy(state)
