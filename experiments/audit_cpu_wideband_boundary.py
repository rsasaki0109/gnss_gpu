import json,hashlib
import numpy as np
from experiments.diagnose_cpu_wideband_boundary import (OUTPUT,PRIOR,KEYS,digest,select_cases,summarize,
    diagnose,diagnostic,dense_problem,synthesize,_problem,estimate_weights,DriftFit,CHIP_RATE)


def run():
    p=json.loads((OUTPUT/'protocol.json').read_text());c=json.loads((OUTPUT/'completion.json').read_text())
    assert all(digest(path)==h for path,h in {**p['source_sha256'],**p['prior_sha256']}.items())
    assert all(digest(OUTPUT/name)==h for name,h in c.items())
    cases=select_cases([json.loads(x) for x in (PRIOR/'evaluation.jsonl').read_text().splitlines()])
    assert p['cases']==cases and len(cases)==29
    rows=[json.loads(x) for x in (OUTPUT/'diagnostics.jsonl').read_text().splitlines()]
    assert len(rows)==29 and summarize(rows)==json.loads((OUTPUT/'report.json').read_text())
    checked=0
    for row,case in zip(rows,cases):
        assert row['replica']==case['replica']
        b,_,truth=synthesize(case['task']);w=estimate_weights(b,DriftFit(**case['fits']['preliminary']),7.2*CHIP_RATE)
        assert hashlib.sha256(b.tobytes()).hexdigest()==case['observation_sha256']
        assert hashlib.sha256(w.tobytes()).hexdigest()==case['weights_sha256']
        solve=_problem(b,w,True,7.2*CHIP_RATE);dense=dense_problem(b,w,7.2*CHIP_RATE)
        assert row['original']['parameters']==[case['fits']['two'][k] for k in KEYS]
        for name in ('original','same_bounds','relaxed_floor'):
            saved=row[name]
            result=diagnostic(solve,dense,saved['parameters'],saved['converged'],truth)
            for k,v in result.items():
                if isinstance(v,float):np.testing.assert_allclose(v,saved[k],rtol=1e-10,atol=1e-10)
                else:assert v==saved[k]
            checked+=1
            if name!='original':
                floor=.01 if name=='same_bounds' else .001
                assert saved['at_gap_floor']==(saved['parameters'][3]-floor<1e-4)
                assert saved['attempted_starts']==25
                assert saved['loss']<=row['original']['loss']+1e-9
                assert saved['relative_loss_improvement']==(row['original']['loss']-saved['loss'])/row['original']['loss']
    for i in (0,28):assert json.loads(json.dumps(diagnose(cases[i])))==rows[i]
    (OUTPUT/'audit.json').write_text(json.dumps(dict(cases=29,all_sources_and_artifacts_verified=True,
        explicit_residual_and_local_information_recomputations=checked,full_search_replays=[0,28]),indent=2))
    print('Audit passed:29 cases,87 dense diagnostics,2 full search replays')


if __name__=='__main__':run()
