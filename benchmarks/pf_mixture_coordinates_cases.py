"""Original 3D NumPy oracle, IEEE cases and complete GPU-state comparisons."""

from types import SimpleNamespace

import numpy as np

from bench_pf_covariance_broadcast_compare import _new_pf
from bench_pf_covariance_compare import _digest, _snapshot


COMPONENTS = (1, 2, 3, 4, 7, 8, 13, 33, 65)
GPU_COUNTS = (33, 257, 8193, 32769, 65537)
GPU_COMPONENTS = (2, 4, 8, 13)


def cpu_ids():
    for components in COMPONENTS:
        chunk = max(1, 65536//components)
        for count in (1, 31, 33, 257, chunk-1, chunk, chunk+1, 2*chunk+3):
            for variant in range(5):
                yield components, count, variant, False
    for components in (1, 2, 3, 7, 13, 33):
        for count in (33, 65536//components+17):
            for variant in range(8):
                yield components, count, variant, True


def inputs(components, count, variant, ieee=False):
    rng = np.random.default_rng(count+components*731)
    if ieee:
        bits = rng.integers(0, 2**64, size=(count, 4), dtype=np.uint64)
        if variant % 2:
            patterns = np.array([0x7ff8000000000001, 0xfff8000000000003,
                0x7ff8000000000005, 0x7ff0000000000000, 0xfff0000000000000,
                0x8000000000000000, 0, 1, 0x7fefffffffffffff], dtype=np.uint64)
            bits[:, :3] = np.resize(patterns, (count, 3))
        particles = bits.view(np.float64)
        prior = rng.integers(0, 2**64, size=count, dtype=np.uint64).view(np.float64)
        refs = rng.normal(size=(components, 3))
        if variant == 6:
            with np.errstate(all='ignore'):
                refs *= 1e308
            refs = np.nan_to_num(refs, nan=0., posinf=1e308, neginf=-1e308)
        sigma = (1., 1e-300, 1e154, 1e308, -1., np.inf, np.nan, 0.)[variant]
        mix = None if variant % 2 else np.resize([1., 0., np.nan, np.inf, -1., 1e308], components)
    else:
        particles = rng.normal(size=(count, 4)) + [6378137., 0., 0., 3000.]
        refs = rng.normal(size=(components, 3)) + [6378137., 0., 0.]
        prior = rng.normal(size=count)
        sigma = (1., 1e-300, 1e300, np.inf, np.nan)[variant]
        mix = rng.normal(size=components) if variant % 2 else None
        if variant == 1:
            particles[::17, 0] = np.nan
            particles[::19, 1] = np.inf
            particles[::23, 2] = -np.inf
            prior.view(np.uint64)[::29] = 0x7ff8000000000073
        elif variant == 2:
            particles[:, 0] = np.resize([1e308, -1e308, 0., -0.], count)
        elif variant == 3:
            prior = np.resize([np.nan, np.inf, -np.inf, -0.], count)
            mix = np.resize([0., np.inf, np.nan, -1., 1.], components)
    return particles, prior, refs, sigma, mix


def host(particles, weights, initialized=True):
    state = SimpleNamespace(_initialized=initialized, particles=particles, weights=weights.copy(), calls=[])
    state.get_particles = lambda: state.particles
    state.get_log_weights = lambda: state.weights
    def upload(value):
        state.calls.append('set_log_weights')
        state.weights = value
    state.set_log_weights = upload
    return state


def check_cpu(method, components, count, variant, ieee=False):
    particles, prior, refs, sigma, mix = inputs(components, count, variant, ieee)
    before = _digest(particles, prior, refs, *([] if mix is None else [mix]))
    actual, oracle = host(particles, prior), host(particles, prior)
    with np.errstate(all='ignore'):
        method(actual, refs, sigma, mix)
        original_update(oracle, refs, sigma, mix)
    np.testing.assert_array_equal(actual.weights.view(np.uint64), oracle.weights.view(np.uint64))
    assert actual.calls == oracle.calls == ['set_log_weights']
    assert before == _digest(particles, prior, refs, *([] if mix is None else [mix]))
    return dict(input_sha256=before, output_sha256=_digest(actual.weights), original_3d_numpy_identical=True)


def rejection_arguments():
    for initialized in (False, True):
        for refs, sigma, mix in (
            (np.zeros((2, 3)), 1., np.zeros(1)),
            (np.zeros((2, 3)), 1., np.zeros(3)),
            (np.zeros((2, 3)), 1., ['bad', 'bad']),
            (np.zeros((2, 3)), 'bad', None),
            (np.zeros((2, 3)), [], None),
            (np.zeros(5), 1., None),
            ([['bad']*3], 1., None),
            (np.array([[0., 0., 0.], [np.nan, 0., 0.]]), 1., np.ones(2)),
        ):
            for raising in (False, True):
                yield initialized, refs, sigma, mix, raising


def check_rejection(method, arguments):
    initialized, refs, sigma, mix, raising = arguments
    outcomes = []
    for operation in (method, original_update):
        state = host(np.zeros((33, 4)), np.zeros(33), initialized)
        before = _digest(state.weights)
        try:
            with np.errstate(all='raise' if raising else 'ignore'):
                operation(state, refs, sigma, mix)
        except (TypeError, ValueError, RuntimeError, FloatingPointError) as error:
            outcomes.append(dict(exception_type=type(error).__name__, message=str(error)))
        else:
            raise AssertionError('Expected the original rejection')
        assert not state.calls and before == _digest(state.weights)
    assert outcomes[0] == outcomes[1]
    return outcomes[0]


def check_gpu(native, klass, count, components, variant):
    actual, oracle = _new_pf(klass, count), _new_pf(klass, count)
    try:
        states = actual.get_particle_states()
        particles, prior, refs, sigma, mix = inputs(components, count, variant)
        states[:, :4] = particles
        states.view(np.uint64)[::29, 5] = 0x7ff8000000000023
        for pf in (actual, oracle):
            pf.set_particle_states(states)
            pf.set_log_weights(prior)
            pf.get_ess()
        with np.errstate(all='ignore'):
            actual.position_mixture_update(refs, sigma, mix)
            original_update(oracle, refs, sigma, mix)
        a, b = _snapshot(native, actual._state), _snapshot(native, oracle._state)
        for left, right in zip(a, b, strict=True):
            np.testing.assert_array_equal(left.view(np.uint64), right.view(np.uint64))
        np.testing.assert_array_equal(a[0].view(np.uint64), states.view(np.uint64))
        queries = []
        for pf in (actual, oracle):
            ess = pf.get_ess()
            mean = pf.estimate()
            queries.append(np.r_[ess, mean, pf.get_position_spread()])
        np.testing.assert_array_equal(queries[0].view(np.uint64), queries[1].view(np.uint64))
        return dict(input_sha256=_digest(states, prior, refs), output_sha256=_digest(*a, queries[0]),
            original_3d_numpy_identical=True, all_other_state_bits_preserved=True, ess_estimate_spread_identical=True)
    finally:
        for pf in (actual, oracle):
            native.pf_device_destroy(pf._state)
            pf._state = None


def validate(native, klass):
    records = {}
    method = klass.position_mixture_update
    for components, count, variant, ieee in cpu_ids():
        records[f'CPU/K{components}/N{count}/v{variant}/ieee{ieee}'] = check_cpu(method, components, count, variant, ieee)
    for scenario, mode in policy_ids():
        records[f'error_policy/s{scenario}/{mode}'] = check_error_policy(method, scenario, mode)
    for index, arguments in enumerate(rejection_arguments()):
        records[f'rejection/{index}'] = check_rejection(method, arguments)
    for count in GPU_COUNTS:
        for components in GPU_COMPONENTS:
            for variant in (0, 1):
                records[f'GPU/N{count}/K{components}/v{variant}'] = check_gpu(native, klass, count, components, variant)
    return records


def original_update(self, refs_ecef, sigma_pos=1.0, mixture_weights=None):
    """Apply a CPU-side Gaussian-mixture position likelihood to particles."""
    if not self._initialized:
        raise RuntimeError("ParticleFilterDevice not initialized. Call initialize() first.")
    refs = np.asarray(refs_ecef, dtype=np.float64).reshape(-1, 3)
    if refs.shape[0] == 0:
        return
    refs = refs[np.all(np.isfinite(refs), axis=1)]
    if refs.shape[0] == 0:
        return
    sigma = max(float(sigma_pos), 1.0e-6)
    if mixture_weights is None:
        mix = np.full(refs.shape[0], 1.0 / refs.shape[0], dtype=np.float64)
    else:
        mix = np.asarray(mixture_weights, dtype=np.float64).reshape(-1)
        if mix.size != refs.shape[0]:
            raise ValueError("mixture_weights length must match refs_ecef")
        mix = np.where(np.isfinite(mix) & (mix > 0.0), mix, 0.0)
        total = float(np.sum(mix))
        if total <= 0.0:
            mix = np.full(refs.shape[0], 1.0 / refs.shape[0], dtype=np.float64)
        else:
            mix = mix / total

    particles = np.asarray(self.get_particles(), dtype=np.float64)
    log_weights = np.asarray(self.get_log_weights(), dtype=np.float64)
    diff = particles[:, np.newaxis, :3] - refs[np.newaxis, :, :]
    log_components = (
        np.log(mix[np.newaxis, :])
        - 0.5 * np.sum(diff * diff, axis=2) / (sigma * sigma)
    )
    max_log = np.max(log_components, axis=1)
    finite = np.isfinite(max_log)
    increment = np.full(log_weights.shape, -np.inf, dtype=np.float64)
    increment[finite] = max_log[finite] + np.log(
        np.sum(np.exp(log_components[finite] - max_log[finite, np.newaxis]), axis=1)
    )
    self.set_log_weights(log_weights + increment)


def policy_ids():
    for scenario in (0, 1):
        for mode in ('raise', 'call', 'log', 'warning_error'):
            yield scenario, mode


def check_error_policy(method, scenario, mode):
    import warnings

    count = 32769
    particles = np.zeros((count, 4))
    refs = np.zeros((2, 3))
    if scenario == 0:
        particles[:, :3] = 1e308
        refs[:] = -1e308
        mix = np.array([1., 0.])
    else:
        particles[:, :3] = 1e154
        mix = np.array([.5, .5])
    outcomes = []
    old_handler = np.geterrcall()
    try:
        for operation in (method, original_update):
            state = host(particles, np.zeros(count))
            events = []
            handler = (lambda message, flags: events.append((message, flags))) if mode == 'call' else SimpleNamespace(write=events.append)
            np.seterrcall(handler)
            outcome = None
            try:
                with warnings.catch_warnings():
                    if mode == 'warning_error':
                        warnings.simplefilter('error', RuntimeWarning)
                    with np.errstate(all='warn' if mode == 'warning_error' else mode):
                        operation(state, refs, 1., mix)
            except (FloatingPointError, RuntimeWarning) as error:
                outcome = dict(exception_type=type(error).__name__, message=str(error))
                assert not state.calls
                np.testing.assert_array_equal(state.weights.view(np.uint64), np.zeros(count).view(np.uint64))
            if outcome is None:
                assert state.calls == ['set_log_weights']
                outcome = dict(output_sha256=_digest(state.weights), events=events)
            outcomes.append(outcome)
        assert outcomes[0] == outcomes[1], (scenario, mode, outcomes)
        return outcomes[0]
    finally:
        np.seterrcall(old_handler)
