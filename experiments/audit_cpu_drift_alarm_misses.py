import json,hashlib
import numpy as np
from gnss_gpu.cached_drift_carrier_paths_cpu import DriftFit,estimate_weights
from gnss_gpu.drift_carrier_paths_cpu import _problem
from experiments.diagnose_cpu_drift_alarm_misses import OUTPUT,PRIOR,digest,missed_cases,diagnostic_grid,synthesize,summarize


def run():
    p=json.loads((OUTPUT/'protocol.json').read_text());completion=json.loads((OUTPUT/'completion.json').read_text())
    assert all(digest(path)==h for path,h in p['source_sha256'].items())
    assert all(digest(OUTPUT/path)==h for path,h in completion.items())
    assert digest(PRIOR/'evaluation.jsonl')==p['prior_evaluation_sha256']
    assert digest(PRIOR/'frozen_gate.json')==p['prior_gate_sha256']
    assert json.loads((PRIOR/'frozen_gate.json').read_text())['score']==p['old_alarm_gate']
    cases=missed_cases([json.loads(x) for x in (PRIOR/'evaluation.jsonl').read_text().splitlines()])
    assert cases==p['cases'] and len(cases)==11
    rows=[json.loads(x) for x in (OUTPUT/'profiles.jsonl').read_text().splitlines()]
    assert len(rows)==len(cases)
    assert summarize(rows,cases,p['old_alarm_gate'])==json.loads((OUTPUT/'report.json').read_text())
    checked=0
    for row,case in zip(rows,cases):
        assert row['replica']==case['replica']
        task=[case[k] for k in ('kind','echo_voltage','excess_chips','phase_rad','drift_hz_s','initial_carrier_hz','replica')]
        b,_,_=synthesize(task);w=estimate_weights(b,DriftFit(**case['fits']['preliminary']))
        assert hashlib.sha256(b.tobytes()).hexdigest()==case['observation_sha256']
        assert hashlib.sha256(w.tobytes()).hexdigest()==case['weights_sha256']
        assert row['complex_count']==b.size
        for model,points in row['profiles'].items():
            np.testing.assert_array_equal([q['delay_chips'] for q in points],diagnostic_grid(case))
            solve=_problem(b,w,model=='two')
            for q in points:
                dense=solve(np.r_[q['delay_chips'],q['nuisance']],True)[0]
                np.testing.assert_allclose(dense,q['loss'],rtol=0,atol=1e-9)
                np.testing.assert_allclose(2*b.size*np.log(dense/row['reference_loss']),q['relative_loss'],rtol=0,atol=1e-6)
                checked+=1
        oracle=row['oracle_fixed_true_delay'];assert oracle['delay_chips']==case['true_delay_chips']
        dense=_problem(b,w,True)(np.r_[oracle['delay_chips'],oracle['nuisance']],True)[0]
        np.testing.assert_allclose(dense,oracle['loss'],rtol=0,atol=1e-9);checked+=1
    (OUTPUT/'audit.json').write_text(json.dumps(dict(all_cases_and_sources_verified=True,summary_recomputed=True,
        original_dense_residual_checks=checked,oracle_points=11),indent=2))
    print(f'Audit passed:11 cases,{checked} dense residual checks')


if __name__=='__main__':run()
