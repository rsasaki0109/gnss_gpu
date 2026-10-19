"""Compare existing FGO solver modes; this is not a GTSAM RBPF benchmark."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

from bench_pf_private_weights_compare import _pin

ROOT = Path(__file__).resolve().parents[1]
SOURCES = (
    'src/positioning/fgo.cu',
    'include/gnss_gpu/fgo.h',
    'python/gnss_gpu/fgo.py',
    'python/gnss_gpu/_bindings.cpp',
    'experiments/rbpf_fgo/rbpf_fgo.py',
    'experiments/rbpf_fgo/gtsam_rtk_standalone.py',
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(array):
    return hashlib.sha256(str((array.shape, array.dtype.str)).encode() + array.tobytes()).hexdigest()


def problem(family, epochs, seed=20261016):
    """Stationary, noisy GNSS/IMU fixture with Sagnac-corrected pseudoranges."""
    rng = np.random.default_rng(seed + epochs)
    truth = np.array([-3957199., 3310205., 3737911.])
    satellites = rng.normal(size=(12, 3))
    satellites *= 26560000. / np.linalg.norm(satellites, axis=1)[:, None]
    satellite = np.broadcast_to(satellites, (epochs, 12, 3)).copy()
    distance = np.linalg.norm(satellite - truth, axis=2)
    angle = 7.2921151467e-5 * distance / 299792458.
    rotated = satellite.copy()
    rotated[:, :, 0] = satellite[:, :, 0] * np.cos(angle) + satellite[:, :, 1] * np.sin(angle)
    rotated[:, :, 1] = -satellite[:, :, 0] * np.sin(angle) + satellite[:, :, 1] * np.cos(angle)
    pr = np.linalg.norm(rotated - truth, axis=2) + 150. + rng.normal(0., 1., (epochs, 12))
    # A fixed outlier exercises the existing robust/line-search paths.
    pr[epochs // 2, 0] += 20.
    weights = np.ones_like(pr)
    state = np.zeros((epochs, 4 if family == 'basic' else 8))
    state[:, :3] = truth + rng.normal(0., 15., (epochs, 3))
    state[:, 3 if family == 'basic' else 6] = 150. + rng.normal(0., 3., epochs)
    kwargs = dict(motion_sigma_m=2., max_iter=5, tol=1e-7, huber_k=3., line_search=True)
    if family != 'basic':
        kwargs.update(clock_drift_sigma_m=1., dt=np.full(epochs, .2),
            sat_vel=np.zeros_like(satellite), doppler=rng.normal(0., .05, pr.shape),
            doppler_weights=np.full(pr.shape, 100.), doppler_huber_k=3.,
            imu_delta_p=np.zeros((epochs - 1, 3)), imu_delta_v=np.zeros((epochs - 1, 3)),
            imu_position_sigma_m=.1, imu_velocity_sigma_mps=.1)
    return satellite, pr, weights, state, kwargs


def stats(samples):
    return dict(p50_ms=float(np.median(samples)), p95_ms=float(np.percentile(samples, 95)),
                p99_ms=float(np.percentile(samples, 99)), samples=len(samples))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--basic-epochs', type=int, nargs='+', default=[5, 50, 128, 200])
    parser.add_argument('--vd-epochs', type=int, nargs='+', default=[5, 20, 64, 100])
    parser.add_argument('--trace-mode', choices=['off', 'on', 'auto'])
    args = parser.parse_args()
    assert not args.output.exists()
    planpath = args.output.with_suffix('.plan.json')
    checkpoint = args.output.with_suffix('.checkpoint.json')
    assert not planpath.exists() and not checkpoint.exists()
    _pin()
    sys.path.insert(0, str(ROOT / 'python'))
    import gnss_gpu._gnss_gpu as native
    from gnss_gpu.fgo import fgo_gnss_lm, fgo_gnss_lm_vd

    extension = Path(native.__file__)
    sources = {name: sha(ROOT / name) for name in SOURCES}
    extension_sha = sha(extension)
    plan = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(), source_sha256=sources,
        extension_path=extension.relative_to(ROOT).as_posix(), extension_sha256=extension_sha,
        helper_sha256=sha(Path(__file__)), rounds=args.rounds, repeats=args.repeats,
        basic_epochs=args.basic_epochs, vd_epochs=args.vd_epochs,
        estimator='Median within-pair percent reduction in each round, then median of round medians.',
        scope='Existing basic/velocity-Doppler-IMU FGO full calls including allocations, transfers, all requested iterations and host return. Fixed synthetic fixture, no raw GNSS/RBPF/GTSAM accuracy or whole-estimator timing.',
        modes={'off': '0', 'on': '1', 'auto': 'auto', 'AA': '0'},
        interpretation='Feasibility only; no default or implementation change. on/auto request GPU but retain existing silent CPU fallback. Trace needed to establish actual dispatch. A/A condition median abs<5% and>=10%/4of5 improvement screen; bit equality and existing 1e-5m ECEF diagnostic recorded independently. No retry or numeric tolerance relaxation.',
        trace_mode=args.trace_mode)
    planpath.write_text(json.dumps(plan, indent=2) + '\n', encoding='utf8')
    records = {}
    previous = os.environ.get('GNSS_GPU_FGO_GPU_SOLVER')
    try:
        for family, counts in (('basic', args.basic_epochs), ('vd_imu', args.vd_epochs)):
            solver = fgo_gnss_lm if family == 'basic' else fgo_gnss_lm_vd
            for epochs in counts:
                satellite, pr, weights, initial, kwargs = problem(family, epochs)
                before = {name: digest(array) for name, array in (
                    ('satellite', satellite), ('pseudorange', pr), ('weights', weights), ('initial', initial))}

                def call(mode):
                    os.environ['GNSS_GPU_FGO_GPU_SOLVER'] = plan['modes'][mode]
                    state = initial.copy()
                    begin = time.perf_counter_ns()
                    result = solver(satellite, pr, weights, state, **kwargs)
                    elapsed = (time.perf_counter_ns() - begin) / 1e6
                    # Both public native APIs return a host state after synchronous transfers.
                    assert int(result[0]) > 0 and np.all(np.isfinite(state)) and np.isfinite(result[1]), result
                    return state, dict(ms=elapsed, iterations=int(result[0]), mse=float(result[1]), output_sha256=digest(state))

                if args.trace_mode:
                    state, run = call(args.trace_mode)
                    records[f'{family}/epochs{epochs}'] = dict(trace_mode=args.trace_mode, result=run)
                    continue
                reference, baseline = call('off')
                for mode in ('on', 'auto', 'AA'):
                    call(mode)
                pairs = []
                validation = dict(max_position_l2_delta_m=0., max_state_abs_delta=0., max_mse_delta=0.,
                    all_state_bits_identical=True, all_iterations_identical=True, every_result_matches_own_mode=True)
                mode_hashes = {}
                modes = ['off', 'on', 'auto', 'AA']
                for round_index in range(args.rounds):
                    for repeat in range(args.repeats):
                        offset = (round_index + repeat) % 4
                        order = modes[offset:] + modes[:offset]
                        runs = {}
                        for mode in order:
                            state, run = call(mode)
                            if mode in mode_hashes:
                                validation['every_result_matches_own_mode'] &= mode_hashes[mode] == run['output_sha256']
                            mode_hashes[mode] = run['output_sha256']
                            validation['all_state_bits_identical'] &= run['output_sha256'] == baseline['output_sha256']
                            validation['all_iterations_identical'] &= run['iterations'] == baseline['iterations']
                            validation['max_state_abs_delta'] = max(validation['max_state_abs_delta'], float(np.max(np.abs(state - reference))))
                            validation['max_position_l2_delta_m'] = max(validation['max_position_l2_delta_m'], float(np.max(np.linalg.norm(state[:, :3] - reference[:, :3], axis=1))))
                            validation['max_mse_delta'] = max(validation['max_mse_delta'], abs(run['mse'] - baseline['mse']))
                            if mode in ('off', 'AA'):
                                assert run['output_sha256'] == baseline['output_sha256'] and run['iterations'] == baseline['iterations']
                            runs[mode] = run
                        pairs.append(dict(round=round_index, repetition=repeat, order=order, runs=runs))
                summary = {mode: stats([pair['runs'][mode]['ms'] for pair in pairs]) for mode in modes}
                for mode in ('on', 'auto', 'AA'):
                    reductions = [float(np.median([100 * (1 - pair['runs'][mode]['ms'] / pair['runs']['off']['ms'])
                        for pair in pairs if pair['round'] == index])) for index in range(args.rounds)]
                    summary[mode].update(paired_round_reduction_percent=reductions,
                        paired_reduction_percent=float(np.median(reductions)), improving_rounds=sum(value > 0 for value in reductions))
                assert validation['every_result_matches_own_mode']
                assert before == {name: digest(array) for name, array in (
                    ('satellite', satellite), ('pseudorange', pr), ('weights', weights), ('initial', initial))}
                key = f'{family}/epochs{epochs}/states{initial.size}'
                records[key] = dict(inputs=before, kwargs={name: digest(value) if isinstance(value, np.ndarray) else value for name, value in kwargs.items()},
                    state_shape=list(initial.shape), pairs=pairs, summary=summary, validation=validation,
                    baseline_iterations=baseline['iterations'], diagnostic_within_existing_10micrometre_position_limit=validation['max_position_l2_delta_m'] <= 1e-5)
                checkpoint.write_text(json.dumps(records, indent=2) + '\n', encoding='utf8')
                print(key, json.dumps(dict(summary=summary, validation=validation)), flush=True)
    finally:
        if previous is None:
            os.environ.pop('GNSS_GPU_FGO_GPU_SOLVER', None)
        else:
            os.environ['GNSS_GPU_FGO_GPU_SOLVER'] = previous
    assert all(sha(ROOT / name) == value for name, value in sources.items()) and sha(extension) == extension_sha
    args.output.write_text(json.dumps(dict(plan=plan, plan_sha256=sha(planpath), measurements=records,
        source_and_extension_unchanged=True, product_changes=[]), indent=2) + '\n', encoding='utf8')


if __name__ == '__main__':
    main()
