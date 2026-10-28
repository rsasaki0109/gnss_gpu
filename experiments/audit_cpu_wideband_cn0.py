from pathlib import Path
import json,hashlib,math
import numpy as np
from experiments.run_cpu_wideband_cn0 import OUTPUT,digest,tasklist,synthesize,evaluate,summarize,LIGHT_SPEED,CHIP_RATE


def run():
    p=json.loads((OUTPUT/'protocol.json').read_text());c=json.loads((OUTPUT/'completion.json').read_text())
    assert all(digest(path)==h for path,h in p['source_sha256'].items())
    assert all(digest(OUTPUT/name)==h for name,h in c.items())
    rows=[json.loads(x) for x in (OUTPUT/'evaluation.jsonl').read_text().splitlines()]
    assert len(rows)==144 and [r['task'] for r in rows]==p['tasks']==json.loads(json.dumps(tasklist()))
    assert summarize(rows)==json.loads((OUTPUT/'report.json').read_text())
    ids={r['replica'] for r in rows};assert len(ids)==48
    for pattern in ('cpu-drift-*/*.jsonl','cpu-wideband-*/*.jsonl','cpu-dominant-*/*.jsonl'):
        for path in Path('output').glob(pattern):
            if path.parent==OUTPUT:continue
            assert ids.isdisjoint(r.get('replica') for r in map(json.loads,path.read_text().splitlines()))
    checks=0
    for base in [r for r in rows if r['cn0_dbhz']==37]:
        bb,bs,bt=synthesize(base['task'])
        for row in [r for r in rows if r['replica']==base['replica']]:
            b,s,t=synthesize(row['task']);assert t==bt
            np.testing.assert_array_equal(s,bs)
            np.testing.assert_allclose(b-s,(bb-bs)*10**((37-row['cn0_dbhz'])/20),rtol=1e-12,atol=1e-15)
            assert hashlib.sha256(b.tobytes()).hexdigest()==row['observation_sha256']
            f=row['fits']['single' if row['kind']=='null' else 'two']
            error=(f['first_delay_chips']-t)*LIGHT_SPEED/CHIP_RATE
            assert error==row['raw_error_m'] and row['available']==f['locally_usable']
            assert row['success']==bool(f['locally_usable'] and abs(error)<10)
            checks+=1
    report=json.loads((OUTPUT/'report.json').read_text());paired=report['primary_37_vs_49']
    gain,loss=paired['gains'],paired['losses'];n=gain+loss
    exact=min(1.,2*sum(math.comb(n,k) for k in range(min(gain,loss)+1))/2**n) if n else 1.
    np.testing.assert_allclose(exact,paired['exact_two_sided_p'],rtol=1e-12,atol=0)
    indices=[0,16,32,48,80,112]
    for i in indices:assert json.loads(json.dumps(evaluate(p['tasks'][i])))==rows[i]
    (OUTPUT/'audit.json').write_text(json.dumps(dict(evaluation=144,sources_and_artifacts_verified=True,
        prior_ids_disjoint=True,signal_noise_scaling_checks=checks,all_endpoints_recomputed=True,
        paired_p_independently_recomputed=True,full_replays=indices),indent=2))
    print('Audit passed:144 records and noise-scaling checks,6 full replays')


if __name__=='__main__':run()
