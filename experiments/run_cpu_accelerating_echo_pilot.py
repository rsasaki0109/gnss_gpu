"""Frozen-receiver sensitivity to physically coupled echo acceleration."""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import numpy as np
from scipy.stats import binomtest
from experiments.run_cpu_moving_echo_pilot import synthesize as baseline
from experiments.run_cpu_drift_selection import tasks, digest
from experiments.stress_cpu_carrier_drift import chirp_blocks
from gnss_gpu.bandwidth_drift_paths_cpu import score_drift_paths
from gnss_gpu.accelerating_echo_cpu import accelerating_echo_blocks, minimum_excess_range
from gnss_gpu.moving_echo_cpu import range_rate_from_doppler
from gnss_gpu.multipath_information_cpu import CHIP_RATE, LIGHT_SPEED, ca_passband_spectrum

OUTPUT = Path('output/cpu-accelerating-echo-pilot')
PRIOR = Path('output/cpu-moving-echo-pilot')
RATES = (0., -.5, .5, -2., 2.)
INITIAL_DOPPLER = .7


def tasklist():
    return [(rate, a, f, i) for rate in RATES for _, a, f, i in tasks('echo', 1350000, 16)]


def motion(rate):
    return range_rate_from_doppler(INITIAL_DOPPLER), range_rate_from_doppler(rate)


def synthesize(task):
    rate, drift, offset, replica = task
    b, old, truth = baseline(('moving_code', INITIAL_DOPPLER, drift, offset, replica))
    v, a = motion(rate)
    if minimum_excess_range(.03*LIGHT_SPEED/CHIP_RATE, v, a, .8) <= 0:
        raise ValueError('reflection must remain later than direct path throughout observation')
    if rate == 0:
        return b, old, truth
    rng = np.random.default_rng(np.random.SeedSequence([20261027, 0x42414E44, replica]))
    assert float(rng.uniform(-.3, .3)) == truth
    common = float(rng.uniform(-np.pi, np.pi))
    f, s = ca_passband_spectrum(7.2*CHIP_RATE)
    direct = s*np.exp(-2j*np.pi*f*truth/CHIP_RATE+1j*common)
    echo = accelerating_echo_blocks(f, offset, drift, v, a, 40)
    signal = direct*(chirp_blocks(offset, drift, 40)[:, None]
                     - .8*echo*np.exp(-2j*np.pi*f*.03/CHIP_RATE))
    return signal+(b-old), signal, truth


def evaluate(task):
    b, _, truth = synthesize(task)
    score, w = score_drift_paths(b, 7.2*CHIP_RATE)
    fit = score['two']
    error = (fit.first_delay_chips-truth)*LIGHT_SPEED/CHIP_RATE
    rate, drift, offset, replica = task
    v, a = motion(rate)
    return dict(task=task, relative_drift_hz_s=rate, replica=replica, true_delay_chips=truth,
        excess_range_rate_m_s=v, excess_acceleration_m_s2=a,
        minimum_excess_m=minimum_excess_range(.03*LIGHT_SPEED/CHIP_RATE, v, a, .8),
        final_relative_doppler_hz=INITIAL_DOPPLER+rate*.8,
        fits={k: asdict(score[k]) for k in ('single', 'two', 'preliminary')},
        uncalibrated_score=score['score'], raw_error_m=error, available=fit.locally_usable,
        success=bool(fit.locally_usable and abs(error)<10),
        observation_sha256=hashlib.sha256(b.tobytes()).hexdigest(),
        weights_sha256=hashlib.sha256(w.tobytes()).hexdigest())


def summarize(rows):
    cells = []
    baseline_rows = {r['replica']: r for r in rows if r['relative_drift_hz_s']==0}
    for rate in RATES:
        g = [r for r in rows if r['relative_drift_hz_s']==rate]
        n = len(g); k = sum(r['success'] for r in g)
        cells.append(dict(relative_drift_hz_s=rate, attempts=n, successes=k,
            success_exact_95=list(binomtest(k, n).proportion_ci(method='exact')),
            unavailable=sum(not r['available'] for r in g),
            available_large_errors=sum(r['available'] and abs(r['raw_error_m'])>=10 for r in g),
            median_absolute_raw_error_m=float(np.median([abs(r['raw_error_m']) for r in g])),
            max_absolute_raw_error_m=max(abs(r['raw_error_m']) for r in g),
            gains=sum(r['success'] and not baseline_rows[r['replica']]['success'] for r in g),
            losses=sum(not r['success'] and baseline_rows[r['replica']]['success'] for r in g)))
    return dict(cells=cells, scope='Exploratory 16 shared replicas per cell, known two-path order, pointwise exact intervals. No significance test or pooled independent sample claim.')


def run():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if any(OUTPUT.iterdir()):
        raise ValueError('output must be empty')
    sources = dict(json.loads((PRIOR/'protocol.json').read_text())['source_sha256'])
    for path in (__file__, 'python/gnss_gpu/accelerating_echo_cpu.py'):
        sources[str(Path(path).resolve())] = digest(path)
    assert all(digest(path)==h for path, h in sources.items())
    protocol = dict(source_sha256=sources, tasks=tasklist(),
        design='Five relative carrier drift rates (0,-.5,.5,-2,2) Hz/s x16 fresh shared replicas. Initial relative Doppler .7Hz, 49dBHz before unity filter,800ms,7.3656MHz one-sided,PRN1,echo voltage .8,initial excess .03chip,phase pi.',
        motion='L(t)=L0+v*t+a*t^2/2. v=-c*.7/fL1; a=-c*relative_drift/fL1. Carrier and code share this excess range. Common frequency/drift remain oscillator terms. 24-point block quadrature. Verify minimum positive including interior vertex.',
        evaluation='Unchanged fixed-code/common-drift receiver, known two-path order and unchanged local guards; success=usable and abs direct range error<10m. No calibrated selection gate. Noise and nuisance paired; zero acceleration bit-exact previous moving-code generator.',
        limits='Synthetic harmonic approximation, no inter-bin leakage, full RF, geometry, time-varying amplitude, forest/rain or PVT. Descriptive pilot, not independent 80-replica inference.')
    (OUTPUT/'protocol.json').write_text(json.dumps(protocol, indent=2))
    rows = []
    with ProcessPoolExecutor(max_workers=4) as pool, (OUTPUT/'evaluation.jsonl').open('w') as stream:
        for row in pool.map(evaluate, protocol['tasks']):
            rows.append(row); stream.write(json.dumps(row, allow_nan=False)+'\n'); stream.flush()
            if len(rows)%16==0:
                print(f'Accelerating echo {len(rows)}/80', flush=True)
    (OUTPUT/'report.json').write_text(json.dumps(summarize(rows), indent=2))
    assert all(digest(path)==h for path, h in sources.items())
    (OUTPUT/'completion.json').write_text(json.dumps({name: digest(OUTPUT/name) for name in
        ('protocol.json', 'evaluation.jsonl', 'report.json')}, indent=2))


if __name__=='__main__':
    run()
