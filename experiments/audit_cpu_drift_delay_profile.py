import json
from pathlib import Path
import numpy as np
from gnss_gpu.cached_drift_carrier_paths_cpu import estimate_weights,DriftFit
from gnss_gpu.drift_carrier_paths_cpu import _problem as original_problem
from experiments.run_cpu_drift_delay_profile import OUTPUT,PRIOR,panel,summarize,synthesize,digest


def run():
    p=json.loads((OUTPUT/'protocol.json').read_text());c=json.loads((OUTPUT/'completion.json').read_text())
    assert all(digest(path)==h for path,h in p['source_sha256'].items())
    assert digest(PRIOR/'evaluation.jsonl')==p['prior_evaluation_sha256']
    assert panel()==p['cases']
    for name,ext in [('protocol','.json'),('profiles','.jsonl'),('report','.json')]:assert digest(OUTPUT/(name+ext))==c[name+'_sha256']
    rows=[json.loads(x) for x in (OUTPUT/'profiles.jsonl').read_text().splitlines()]
    assert len(rows)==6
    assert summarize(rows)==json.loads((OUTPUT/'report.json').read_text())
    checked=0
    for row,case in zip(rows,p['cases']):
        assert row['replica']==case['replica']
        task=[case[k] for k in ('kind','echo_voltage','excess_chips','phase_rad','drift_hz_s','initial_carrier_hz','replica')]
        b,_,_=synthesize(task);w=estimate_weights(b,DriftFit(**case['fits']['preliminary']))
        d=case['fits']['single']['first_delay_chips']
        expected=np.unique(np.clip(np.r_[d+np.linspace(-.15,.15,25),d,case['fits']['two']['first_delay_chips']],-1,1))
        for model,points in row['profiles'].items():
            np.testing.assert_array_equal([q['delay_chips'] for q in points],expected)
            solve=original_problem(b,w,model=='two')
            for point in points:
                value=solve(np.r_[point['delay_chips'],point['nuisance']],True)[0]
                np.testing.assert_allclose(value,point['loss'],rtol=0,atol=1e-9)
                np.testing.assert_allclose(2*b.size*np.log(value/row['reference_loss']),point['relative_loss'],atol=1e-6,rtol=0)
                checked+=1
    (OUTPUT/'audit.json').write_text(json.dumps(dict(cases=6,all_sources_and_artifacts_verified=True,
        all_profile_points_recomputed_with_original_dense_residual=checked),indent=2))
    print(f'Audit passed:6 cases,{checked} original dense residual checks')


if __name__=='__main__':run()
