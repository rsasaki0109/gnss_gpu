from pathlib import Path
import json,hashlib
import numpy as np
from experiments.stress_cpu_wideband_phase_doppler import OUTPUT,digest,tasklist,synthesize,evaluate,summarize,LIGHT_SPEED,CHIP_RATE


def run():
    p=json.loads((OUTPUT/'protocol.json').read_text());c=json.loads((OUTPUT/'completion.json').read_text())
    assert all(digest(path)==h for path,h in p['source_sha256'].items())
    assert all(digest(OUTPUT/name)==h for name,h in c.items())
    rows=[json.loads(x) for x in (OUTPUT/'evaluation.jsonl').read_text().splitlines()]
    assert len(rows)==192 and [r['task'] for r in rows]==p['tasks']==json.loads(json.dumps(tasklist()))
    assert summarize(rows)==json.loads((OUTPUT/'report.json').read_text())
    ids={r['replica'] for r in rows};assert len(ids)==16
    for pattern in ('cpu-drift-*/*.jsonl','cpu-wideband-*/*.jsonl','cpu-dominant-*/*.jsonl'):
        for path in Path('output').glob(pattern):
            if path.parent==OUTPUT:continue
            assert ids.isdisjoint(r.get('replica') for r in map(json.loads,path.read_text().splitlines()))
    checks=0
    for replica in sorted(ids):
        group=[r for r in rows if r['replica']==replica];assert len(group)==12
        ref=next(r for r in group if r['cn0_dbhz']==37 and r['relative_doppler_hz']==0 and r['phase_rad']==np.pi)
        rb,rs,rt=synthesize(ref['task'])
        for row in group:
            b,s,t=synthesize(row['task']);assert t==rt
            scale=10**((37-row['cn0_dbhz'])/20)
            np.testing.assert_allclose(b-s,(rb-rs)*scale,rtol=1e-12,atol=1e-15)
            assert hashlib.sha256(b.tobytes()).hexdigest()==row['observation_sha256']
            f=row['fits']['two'];error=(f['first_delay_chips']-t)*LIGHT_SPEED/CHIP_RATE
            assert error==row['raw_error_m'] and row['available']==f['locally_usable']
            assert row['success']==bool(f['locally_usable'] and abs(error)<10)
            checks+=1
    indices=list(range(0,192,16))
    for i in indices:assert json.loads(json.dumps(evaluate(p['tasks'][i])))==rows[i]
    (OUTPUT/'audit.json').write_text(json.dumps(dict(evaluation=192,sources_and_artifacts_verified=True,
        prior_ids_disjoint=True,paired_standardized_noise_checks=checks,all_endpoints_recomputed=True,
        full_replays=indices),indent=2))
    print('Audit passed:192 observations/noise checks,12 full replays')


if __name__=='__main__':run()
