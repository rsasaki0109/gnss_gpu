import json,hashlib
import numpy as np
from experiments.diagnose_cpu_relative_drift_search import OUTPUT,PRIOR,digest,selected_rows,synthesize,evaluate,summarize
from gnss_gpu.bandwidth_drift_paths_cpu import score_drift_paths
from gnss_gpu.relative_drift_paths_cpu import _problem
from gnss_gpu.broad_relative_drift_cpu import KEYS,BOUNDS
from gnss_gpu.multipath_information_cpu import CHIP_RATE,LIGHT_SPEED


def run():
    p=json.loads((OUTPUT/'protocol.json').read_text());c=json.loads((OUTPUT/'completion.json').read_text())
    assert all(digest(path)==h for path,h in p['source_sha256'].items())
    assert all(digest(OUTPUT/name)==h for name,h in c.items())
    assert all(digest(PRIOR/name)==h for name,h in p['prior_completion'].items())
    rows=[json.loads(x) for x in (OUTPUT/'evaluation.jsonl').read_text().splitlines()]
    originals=selected_rows()
    assert len(rows)==40 and [r['task'] for r in rows]==p['tasks']==[r['task'] for r in originals]
    assert summarize(rows)==json.loads((OUTPUT/'report.json').read_text())
    indices=[];seen=set()
    for index,(r,original) in enumerate(zip(rows,originals)):
        assert r['original']==original['models']['relative_drift']
        assert r['observation_sha256']==original['observation_sha256']
        assert r['weights_sha256']==original['weights_sha256']
        b,_,truth=synthesize(r['task'])
        assert hashlib.sha256(b.tobytes()).hexdigest()==r['observation_sha256']
        assert truth==r['true_delay_chips']
        model=r['broad'];fit=model['fit'];oldloss=r['original']['fit']['loss']
        assert model['raw_error_m']==(fit['first_delay_chips']-truth)*LIGHT_SPEED/CHIP_RATE
        assert model['available']==fit['locally_usable']
        assert model['success']==bool(model['available'] and abs(model['raw_error_m'])<10)
        assert r['relative_loss_improvement']==(oldloss-fit['loss'])/oldloss
        assert fit['loss']<=oldloss+1e-8*oldloss
        key=(r['initial_doppler_hz'],r['relative_drift_hz_s'])
        if key not in seen:indices.append(index);seen.add(key)
    for index in indices:
        r=rows[index];assert json.loads(json.dumps(evaluate(originals[index])))==r
        b,_,_=synthesize(r['task']);_,w=score_drift_paths(b,7.2*CHIP_RATE)
        fit=r['broad']['fit'];params=[fit[k] for k in KEYS]
        solve=_problem(b,w,True,7.2*CHIP_RATE);loss,a,j=solve(params,True)
        np.testing.assert_allclose(solve(params)[0],loss,rtol=1e-12,atol=1e-9)
        projected=j-a@np.linalg.lstsq(a,j,rcond=None)[0]
        info=np.real(projected.conj().T@projected);eig=np.linalg.eigvalsh(info)
        std=None if eig[0]<=max(1e-20,1e-12*eig[-1]) else float(np.sqrt(max(0,loss/(2*len(a)-10)*np.linalg.inv(info)[0,0])))
        assert loss==fit['loss'] and std==fit['first_std_chips']
        reason=('optimizer did not converge' if not fit['converged'] else
                'search boundary' if min(min(x-lo,hi-x) for x,(lo,hi) in zip(params,BOUNDS))<1e-4 else
                'weak local delay information' if std is None or std>.1 else
                'interior locally conditioned fit')
        assert fit['reason']==reason
        print(f'Audit full search {index}',flush=True)
    (OUTPUT/'audit.json').write_text(json.dumps(dict(observations=40,prior_and_sources_verified=True,
        all_endpoints_recomputed=True,full_search_replays=indices,dense_covariance_checks=6),indent=2))
    print('Audit passed:40 observations,6 full searches and dense covariance checks')


if __name__=='__main__':run()
