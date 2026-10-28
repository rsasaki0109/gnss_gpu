from pathlib import Path
import json,hashlib
import numpy as np
from experiments.run_cpu_moving_echo_pilot import OUTPUT,digest,tasklist,synthesize,evaluate,summarize,LIGHT_SPEED,CHIP_RATE,range_rate_from_doppler


def run():
    p=json.loads((OUTPUT/'protocol.json').read_text());c=json.loads((OUTPUT/'completion.json').read_text())
    assert all(digest(path)==h for path,h in p['source_sha256'].items())
    assert all(digest(OUTPUT/name)==h for name,h in c.items())
    rows=[json.loads(x) for x in (OUTPUT/'evaluation.jsonl').read_text().splitlines()]
    assert len(rows)==128 and [r['task'] for r in rows]==p['tasks']==json.loads(json.dumps(tasklist()))
    assert summarize(rows)==json.loads((OUTPUT/'report.json').read_text())
    ids={r['replica'] for r in rows};assert len(ids)==16
    for pattern in ('cpu-drift-*/*.jsonl','cpu-wideband-*/*.jsonl','cpu-dominant-*/*.jsonl'):
        for path in Path('output').glob(pattern):assert ids.isdisjoint(r.get('replica') for r in map(json.loads,path.read_text().splitlines()))
    for fixed in [r for r in rows if r['mode']=='fixed_code']:
        moving=next(r for r in rows if r['mode']=='moving_code' and r['replica']==fixed['replica'] and r['relative_doppler_hz']==fixed['relative_doppler_hz'])
        b,s,t=synthesize(fixed['task']);mb,ms,mt=synthesize(moving['task']);assert t==mt
        np.testing.assert_allclose(mb-ms,b-s,rtol=1e-12,atol=1e-15)
        if fixed['relative_doppler_hz']==0:
            np.testing.assert_array_equal(mb,b);np.testing.assert_array_equal(ms,s)
            assert fixed['fits']==moving['fits'] and fixed['weights_sha256']==moving['weights_sha256']
        for row,blocks,truth in ((fixed,b,t),(moving,mb,mt)):
            assert hashlib.sha256(blocks.tobytes()).hexdigest()==row['observation_sha256']
            rate=range_rate_from_doppler(row['relative_doppler_hz']) if row['mode']=='moving_code' else 0.
            assert row['excess_range_rate_m_s']==rate
            assert row['excess_midpoint_chips']==.03+rate*.4*CHIP_RATE/LIGHT_SPEED
            assert row['excess_end_chips']==.03+rate*.8*CHIP_RATE/LIGHT_SPEED
            assert row['excess_end_chips']>0
            fit=row['fits']['two'];error=(fit['first_delay_chips']-truth)*LIGHT_SPEED/CHIP_RATE
            assert row['raw_error_m']==error and row['available']==fit['locally_usable']
            assert row['success']==bool(fit['locally_usable'] and abs(error)<10)
            assert row['gap_midpoint_error_m']==(fit['excess_chips']-row['excess_midpoint_chips'])*LIGHT_SPEED/CHIP_RATE
    indices=list(range(0,128,16))
    for i in indices:assert json.loads(json.dumps(evaluate(p['tasks'][i])))==rows[i]
    (OUTPUT/'audit.json').write_text(json.dumps(dict(evaluation=128,sources_and_artifacts_verified=True,
        prior_ids_disjoint=True,paired_noise_checks=64,zero_motion_pipeline_exact=16,all_endpoints_recomputed=True,
        full_replays=indices),indent=2))
    print('Audit passed:128 observations,64 noise pairs,16 zero-motion exact pipelines,8 full replays')


if __name__=='__main__':run()
