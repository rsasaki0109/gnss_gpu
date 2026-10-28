import json,hashlib
from pathlib import Path
import numpy as np
from gnss_gpu.cached_drift_carrier_paths_cpu import DriftFit,estimate_weights
from gnss_gpu.dominant_drift_paths_cpu import dominant_problem,geometric_interior,fit_parameters
from gnss_gpu.dominant_path_amplitudes_cpu import dominant_amplitudes
from gnss_gpu.residual_dynamic_paths_cpu import _inverse_std
from experiments.run_cpu_dominant_path_pilot import OUTPUT,digest,tasklist,evaluate,summarize,synthesize,LIGHT_SPEED,CHIP_RATE


def run():
    p=json.loads((OUTPUT/'protocol.json').read_text());c=json.loads((OUTPUT/'completion.json').read_text())
    assert all(digest(path)==h for path,h in p['source_sha256'].items())
    assert all(digest(OUTPUT/name)==h for name,h in c.items())
    rows=[json.loads(x) for x in (OUTPUT/'evaluation.jsonl').read_text().splitlines()]
    assert len(rows)==32 and [r['task'] for r in rows]==p['tasks']==json.loads(json.dumps(tasklist()))
    assert summarize(rows)==json.loads((OUTPUT/'report.json').read_text())
    ids={r['replica'] for r in rows};assert len(ids)==16
    for path in Path('output').glob('cpu-drift-*/*.jsonl'):
        assert ids.isdisjoint(r.get('replica') for r in map(json.loads,path.read_text().splitlines()))
    for i,row in enumerate(rows):
        b,signal,truth=synthesize(row['task']);fit=row['constrained']
        w=estimate_weights(b,DriftFit(**row['baseline_fits']['preliminary']))
        assert hashlib.sha256(b.tobytes()).hexdigest()==row['observation_sha256']
        assert hashlib.sha256(w.tobytes()).hexdigest()==row['weights_sha256']
        dense,a,_=dominant_problem(b,w,True)(fit['parameters'],True)
        np.testing.assert_allclose(dense,fit['loss'],rtol=0,atol=1e-9)
        y=b*_inverse_std(w,b.shape[1]);y=y/np.max(np.abs(y));y=y.ravel()
        amp=dominant_amplitudes(a.conj().T@a,a.conj().T@y)
        residual=y-a@amp
        np.testing.assert_allclose(np.vdot(residual,residual).real,fit['loss'],rtol=0,atol=1e-8)
        ratio=float(abs(amp[1])/abs(amp[0]));assert ratio<=1+1e-12
        assert ratio==fit['echo_to_first_voltage_ratio']
        assert fit['amplitude_constraint_active']==(abs(ratio-1)<1e-7)
        free=DriftFit(**row['baseline_fits']['two'])
        e0=(free.first_delay_chips-truth)*LIGHT_SPEED/CHIP_RATE
        e1=(fit['parameters'][0]-truth)*LIGHT_SPEED/CHIP_RATE
        np.testing.assert_allclose([e0,e1],[row['free_error_m'],row['constrained_error_m']],rtol=0,atol=1e-12)
        assert row['free_eligible']==bool(free.converged and geometric_interior(fit_parameters(free)))
        assert row['constrained_eligible']==bool(fit['converged'] and geometric_interior(fit['parameters']))
        assert row['free_success']==(row['free_eligible'] and abs(e0)<10)
        assert row['constrained_success']==(row['constrained_eligible'] and abs(e1)<10)
        if i<16:
            other,other_signal,other_truth=synthesize(rows[i+16]['task'])
            assert truth==other_truth
            np.testing.assert_allclose(b-signal,other-other_signal,rtol=1e-12,atol=1e-15)
    for i in (0,31):assert json.loads(json.dumps(evaluate(p['tasks'][i])))==rows[i]
    (OUTPUT/'audit.json').write_text(json.dumps(dict(cases=32,paired_noise_checks=16,
        all_amplitude_constraints_and_dense_losses_verified=True,all_endpoints_recomputed=True,
        prior_ids_disjoint=True,full_replays=[0,31]),indent=2))
    print('Audit passed:32 constrained fits,16 paired-noise checks,2 full replays')


if __name__=='__main__':run()
