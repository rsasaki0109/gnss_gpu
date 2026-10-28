"""Replay observations, paired noise, physical endpoints and full receiver fits."""
from pathlib import Path
import hashlib
import json
import numpy as np
from experiments.run_cpu_accelerating_echo_pilot import (
    OUTPUT, digest, tasklist, synthesize, evaluate, summarize, motion,
    INITIAL_DOPPLER, LIGHT_SPEED, CHIP_RATE)
from experiments.run_cpu_moving_echo_pilot import synthesize as baseline


def run():
    p = json.loads((OUTPUT/'protocol.json').read_text())
    c = json.loads((OUTPUT/'completion.json').read_text())
    assert all(digest(path)==h for path, h in p['source_sha256'].items())
    assert all(digest(OUTPUT/name)==h for name, h in c.items())
    rows = [json.loads(x) for x in (OUTPUT/'evaluation.jsonl').read_text().splitlines()]
    assert len(rows)==80 and [r['task'] for r in rows]==p['tasks']==json.loads(json.dumps(tasklist()))
    assert summarize(rows)==json.loads((OUTPUT/'report.json').read_text())
    ids = {r['replica'] for r in rows}; assert len(ids)==16
    for pattern in ('cpu-drift-*/*.jsonl', 'cpu-wideband-*/*.jsonl', 'cpu-dominant-*/*.jsonl', 'cpu-moving-*/*.jsonl'):
        for path in Path('output').glob(pattern):
            assert ids.isdisjoint(r.get('replica') for r in map(json.loads, path.read_text().splitlines()))
    pairs = zero = 0
    for row in rows:
        rate, drift, offset, replica = row['task']
        b, s, truth = synthesize(row['task'])
        old, signal, oldtruth = baseline(('moving_code', INITIAL_DOPPLER, drift, offset, replica))
        assert truth==oldtruth==row['true_delay_chips']
        np.testing.assert_allclose(b-s, old-signal, rtol=1e-12, atol=1e-15)
        pairs += 1
        if rate==0:
            np.testing.assert_array_equal(b, old); np.testing.assert_array_equal(s, signal)
            zero += 1
        assert hashlib.sha256(b.tobytes()).hexdigest()==row['observation_sha256']
        v, a = motion(rate)
        assert row['excess_range_rate_m_s']==v and row['excess_acceleration_m_s2']==a
        # Independent minimization using endpoints and derivative root.
        times = [0., .8]
        if a != 0 and 0 < -v/a < .8:
            times.append(-v/a)
        ranges = [.03*LIGHT_SPEED/CHIP_RATE+v*t+.5*a*t*t for t in times]
        assert min(ranges)==row['minimum_excess_m'] > 0
        assert row['final_relative_doppler_hz']==INITIAL_DOPPLER+rate*.8
        fit = row['fits']['two']
        error = (fit['first_delay_chips']-truth)*LIGHT_SPEED/CHIP_RATE
        assert row['raw_error_m']==error and row['available']==fit['locally_usable']
        assert row['success']==bool(fit['locally_usable'] and abs(error)<10)
    indices = list(range(0, 80, 16))
    for i in indices:
        assert json.loads(json.dumps(evaluate(p['tasks'][i])))==rows[i]
    (OUTPUT/'audit.json').write_text(json.dumps(dict(evaluation=80, sources_and_artifacts_verified=True,
        prior_ids_disjoint=True, paired_noise_checks=pairs, zero_acceleration_exact=zero,
        physical_endpoints_verified=True, all_endpoints_recomputed=True, full_replays=indices), indent=2))
    print('Audit passed:80 observations/noise pairs,16 zero-acceleration matches,5 full replays')


if __name__=='__main__':
    run()
