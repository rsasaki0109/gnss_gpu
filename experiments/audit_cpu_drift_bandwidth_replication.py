from pathlib import Path
import json,hashlib
import numpy as np
from experiments.replicate_cpu_drift_bandwidth import (OUTPUT,digest,tasklist,synthesize,evaluate,summarize,CHIP_RATE,LIGHT_SPEED)
from gnss_gpu.multipath_information_cpu import ca_passband_spectrum


def run():
    p=json.loads((OUTPUT/'protocol.json').read_text());c=json.loads((OUTPUT/'completion.json').read_text())
    assert all(digest(path)==h for path,h in p['source_sha256'].items())
    assert all(digest(OUTPUT/name)==h for name,h in c.items())
    rows=[json.loads(x) for x in (OUTPUT/'evaluation.jsonl').read_text().splitlines()]
    assert len(rows)==192 and [r['task'] for r in rows]==p['tasks']==json.loads(json.dumps(tasklist()))
    assert summarize(rows)==json.loads((OUTPUT/'report.json').read_text())
    ids={r['replica'] for r in rows};assert len(ids)==96
    for pattern in ('cpu-drift-*/*.jsonl','cpu-dominant-*/*.jsonl'):
        for path in Path('output').glob(pattern):
            if path.parent==OUTPUT:continue
            assert ids.isdisjoint(r.get('replica') for r in map(json.loads,path.read_text().splitlines()))
    f,_=ca_passband_spectrum(7.2*CHIP_RATE);mask=abs(f)<=3.6*CHIP_RATE
    pairs=0
    for narrow in [r for r in rows if r['bandwidth_chips']==3.6]:
        wide=next(r for r in rows if r['replica']==narrow['replica'] and r['bandwidth_chips']==7.2)
        nb,ns,nt=synthesize(narrow['task']);wb,ws,wt=synthesize(wide['task'])
        assert nt==wt
        np.testing.assert_array_equal(nb,wb[:,mask]);np.testing.assert_array_equal(ns,ws[:,mask]);pairs+=1
        for row,b,truth in [(narrow,nb,nt),(wide,wb,wt)]:
            assert hashlib.sha256(b.tobytes()).hexdigest()==row['observation_sha256']
            fit=row['fits']['single' if row['kind']=='null' else 'two']
            error=(fit['first_delay_chips']-truth)*LIGHT_SPEED/CHIP_RATE
            assert row['raw_error_m']==error and row['available']==fit['locally_usable']
            assert row['success']==bool(fit['locally_usable'] and abs(error)<10)
    # Independent count calculation for the primary paired contrast.
    echo={b:{r['replica']:r['success'] for r in rows if r['kind']=='echo' and r['bandwidth_chips']==b} for b in (3.6,7.2)}
    gains=sum(not echo[3.6][i] and echo[7.2][i] for i in echo[3.6]);losses=sum(echo[3.6][i] and not echo[7.2][i] for i in echo[3.6])
    import math
    n=gains+losses
    exact=min(1.,2*sum(math.comb(n,k) for k in range(min(gains,losses)+1))/2**n) if n else 1.
    report=json.loads((OUTPUT/'report.json').read_text())
    np.testing.assert_allclose(exact,report['primary_echo_comparison']['exact_two_sided_p'],rtol=1e-12,atol=0)
    indices=[0,32,64,128]
    for i in indices:assert json.loads(json.dumps(evaluate(p['tasks'][i])))==rows[i]
    (OUTPUT/'audit.json').write_text(json.dumps(dict(evaluation=192,sources_and_artifacts_verified=True,
        prior_ids_disjoint=True,exact_nested_pairs=pairs,all_endpoints_recomputed=True,
        paired_test_independently_recomputed=True,full_replays=indices),indent=2))
    print('Audit passed:192 observations,96 exact pairs,independent paired test,4 full replays')


if __name__=='__main__':run()
