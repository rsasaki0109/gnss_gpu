from pathlib import Path
import hashlib,json
import numpy as np
from experiments.run_cpu_relative_drift_pilot import OUTPUT,tasklist,digest,summarize,synthesize,evaluate
from gnss_gpu.relative_drift_paths_cpu import _problem
from gnss_gpu.bandwidth_drift_paths_cpu import score_drift_paths
from gnss_gpu.multipath_information_cpu import CHIP_RATE,LIGHT_SPEED


def run():
    p=json.loads((OUTPUT/'protocol.json').read_text());c=json.loads((OUTPUT/'completion.json').read_text())
    assert all(digest(path)==h for path,h in p['source_sha256'].items())
    assert all(digest(OUTPUT/name)==h for name,h in c.items())
    rows=[json.loads(x) for x in (OUTPUT/'evaluation.jsonl').read_text().splitlines()]
    assert len(rows)==96 and [r['task'] for r in rows]==p['tasks']==json.loads(json.dumps(tasklist()))
    assert summarize(rows)==json.loads((OUTPUT/'report.json').read_text())
    ids={r['replica'] for r in rows};assert len(ids)==16
    for pattern in ('cpu-drift-*/*.jsonl','cpu-wideband-*/*.jsonl','cpu-moving-*/*.jsonl',
                    'cpu-accelerating-*/*.jsonl','cpu-echo-*/*.jsonl','cpu-dominant-*/*.jsonl'):
        for path in Path('output').glob(pattern):
            assert ids.isdisjoint(r.get('replica') for r in map(json.loads,path.read_text().splitlines()))
    for index,r in enumerate(rows):
        b,_,truth=synthesize(r['task']);assert truth==r['true_delay_chips']
        assert hashlib.sha256(b.tobytes()).hexdigest()==r['observation_sha256']
        for model in r['models'].values():
            fit=model['fit'];error=(fit['first_delay_chips']-truth)*LIGHT_SPEED/CHIP_RATE
            assert error==model['raw_error_m'] and model['available']==fit['locally_usable']
            assert model['success']==bool(fit['locally_usable'] and abs(error)<10)
        assert r['estimated_relative_drift_error_hz_s']==r['models']['relative_drift']['fit']['relative_drift_hz_s']-r['relative_drift_hz_s']
        if (index+1)%16==0:print(f'Audit observations {index+1}/96',flush=True)
    indices=list(range(0,96,16))
    for index in indices:
        r=rows[index];assert json.loads(json.dumps(evaluate(r['task'])))==r
        b,_,_=synthesize(r['task']);_,w=score_drift_paths(b,7.2*CHIP_RATE)
        fit=r['models']['relative_drift']['fit']
        params=[fit[k] for k in ('first_delay_chips','common_carrier_hz','common_drift_hz_s',
                                'excess_chips','relative_doppler_hz','relative_drift_hz_s')]
        loss,a,j=_problem(b,w,True,7.2*CHIP_RATE)(params,True)
        projected=j-a@np.linalg.lstsq(a,j,rcond=None)[0]
        info=np.real(projected.conj().T@projected);eig=np.linalg.eigvalsh(info)
        std=None if eig[0]<=max(1e-20,1e-12*eig[-1]) else float(np.sqrt(max(0,loss/(2*len(a)-10)*np.linalg.inv(info)[0,0])))
        assert loss==fit['loss'] and std==fit['first_std_chips']
        print(f'Audit full replay {index}',flush=True)
    timing=json.loads((OUTPUT/'timing.json').read_text())
    assert len(timing)==2 and [r['task'] for r in timing]==[p['tasks'][16],p['tasks'][80]]
    for r in timing:
        assert r['baseline_seconds']>0 and r['refinement_seconds']>0
        assert r['total_seconds']==r['baseline_seconds']+r['refinement_seconds']
    (OUTPUT/'audit.json').write_text(json.dumps(dict(evaluation=96,sources_and_artifacts_verified=True,
        prior_ids_disjoint=True,all_endpoints_recomputed=True,full_replays=indices,
        covariance_checks=6,serial_timing_records=2),indent=2))
    print('Audit passed:96 observations,6 full replays/covariance checks,2 serial timings')


if __name__=='__main__':run()
