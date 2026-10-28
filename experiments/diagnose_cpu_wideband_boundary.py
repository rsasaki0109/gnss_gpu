"""All wideband boundary failures: expanded search and relaxed gap floor."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import json,hashlib
import numpy as np
from scipy.optimize import minimize
from gnss_gpu.bandwidth_drift_paths_cpu import _problem,estimate_weights,DriftFit,temporal_chirp
from gnss_gpu.multipath_information_cpu import ca_passband_spectrum,CHIP_RATE,LIGHT_SPEED
from experiments.replicate_cpu_drift_bandwidth import synthesize,digest

PRIOR=Path('output/cpu-drift-bandwidth-replication')
OUTPUT=Path('output/cpu-wideband-boundary')
KEYS=('first_delay_chips','common_carrier_hz','common_drift_hz_s','excess_chips','relative_doppler_hz')


def select_cases(rows):
    return [r for r in rows if r['kind']=='echo' and r['bandwidth_chips']==7.2 and not r['available'] and r['fits']['two']['reason']=='search boundary']


def dense_problem(blocks,weights,bandwidth_hz):
    f,s=ca_passband_spectrum(bandwidth_hz)
    inv=1/np.sqrt(weights/weights.mean());y=blocks*inv;y=(y/np.max(np.abs(y))).ravel();s=s*inv
    def solve(p):
        d,c,a,g,r=p
        design=np.column_stack([(temporal_chirp(c,a,blocks.shape[0])[0][:,None]*s*np.exp(-2j*np.pi*f*d/CHIP_RATE)).ravel(),
             (temporal_chirp(c+r,a,blocks.shape[0])[0][:,None]*s*np.exp(-2j*np.pi*f*(d+g)/CHIP_RATE)).ravel()])
        amp=np.linalg.lstsq(design,y,rcond=None)[0];prediction=design@amp;residual=y-prediction
        return dict(loss=float(np.vdot(residual,residual).real),template_condition=float(np.linalg.cond(design)),
            cancellation_ratio=float(sum(np.linalg.norm(design[:,i]*amp[i]) for i in range(2))/np.linalg.norm(prediction)))
    return solve


def diagnostic(solve,dense,p,converged,truth):
    detail=dense(p);loss,a,jac=solve(p,True)
    projected=jac-a@np.linalg.lstsq(a,jac,rcond=None)[0]
    info=np.real(projected.conj().T@projected);eig=np.linalg.eigvalsh(info);std=None
    if eig[0]>max(1e-20,1e-12*eig[-1]):std=float(np.sqrt(max(0,detail['loss']/(2*len(a)-9)*np.linalg.inv(info)[0,0])))
    return dict(detail,parameters=list(map(float,p)),converged=bool(converged),first_std_chips=std,
        error_m=float((p[0]-truth)*LIGHT_SPEED/CHIP_RATE),compact_dense_difference=float(loss-detail['loss']))


def diagnose(row):
    blocks,_,truth=synthesize(row['task']);bandwidth=7.2*CHIP_RATE
    assert hashlib.sha256(blocks.tobytes()).hexdigest()==row['observation_sha256']
    w=estimate_weights(blocks,DriftFit(**row['fits']['preliminary']),bandwidth)
    assert hashlib.sha256(w.tobytes()).hexdigest()==row['weights_sha256']
    solve=_problem(blocks,w,True,bandwidth);dense=dense_problem(blocks,w,bandwidth)
    old=row['fits']['two'];parameters=[old[k] for k in KEYS]
    original=diagnostic(solve,dense,parameters,old['converged'],truth)
    np.testing.assert_allclose(original['loss'],old['loss'],rtol=1e-12)
    # Observation-derived carrier/drift seeds plus broad fixed seeds; no truth.
    carriers=sorted(set((-1.5,0.,1.5,old['common_carrier_hz'])))
    drifts=sorted(set((-2.,0.,2.,old['common_drift_hz_s'])))
    seeds=[(d,c,a,g,r) for d in np.linspace(-1,1,17) for c in carriers for a in drifts
           for g in (.01,.05,.2,.5,1.) for r in (-10.,-1.,0.,1.,10.)]
    seeds=sorted(seeds,key=lambda p:solve(p)[0])[:24]+[parameters]
    results={}
    for name,floor in [('same_bounds',.01),('relaxed_floor',.001)]:
        fits=[minimize(solve,seed,jac=True,method='L-BFGS-B',bounds=((-1,1),(-2,2),(-3,3),(floor,1.2),(-12,12)),
              options=dict(maxiter=400,ftol=1e-13,gtol=1e-9)) for seed in seeds]
        candidates=[dict(parameters=parameters,loss=original['loss'],converged=old['converged'])]+[
            dict(parameters=f.x.tolist(),loss=dense(f.x)['loss'],converged=bool(f.success)) for f in fits]
        best=min(candidates,key=lambda r:r['loss'])
        results[name]=dict(diagnostic(solve,dense,best['parameters'],best['converged'],truth),
            relative_loss_improvement=(original['loss']-best['loss'])/original['loss'],
            at_gap_floor=bool(best['parameters'][3]-floor<1e-4),
            successful_starts=sum(bool(f.success) for f in fits),attempted_starts=len(fits))
    return dict(replica=row['replica'],original=original,**results)


def summarize(rows):
    methods={}
    for name in ('same_bounds','relaxed_floor'):
        g=[r[name] for r in rows]
        methods[name]=dict(materially_better=sum(r['relative_loss_improvement']>1e-9 for r in g),
            at_gap_floor=sum(r['at_gap_floor'] for r in g),best_nonconverged=sum(not r['converged'] for r in g),
            weak_local_information=sum(r['first_std_chips'] is None or r['first_std_chips']>.1 for r in g),
            raw_error_under_10m=sum(abs(r['error_m'])<10 for r in g),
            max_relative_loss_improvement=max(r['relative_loss_improvement'] for r in g),
            cancellation_range=[min(r['cancellation_ratio'] for r in g),max(r['cancellation_ratio'] for r in g)],
            condition_range=[min(r['template_condition'] for r in g),max(r['template_condition'] for r in g)],
            finite_std_range=[min((r['first_std_chips'] for r in g if r['first_std_chips'] is not None),default=None),
                              max((r['first_std_chips'] for r in g if r['first_std_chips'] is not None),default=None)])
    return dict(cases=len(rows),methods=methods,
        scope='Retrospective all29 selected failures,not new population performance. Material improvement cutoff1e-9 predeclared. Raw error ignores boundary/uncertainty guards;not a receiver success endpoint. No calibrated gate or production receiver changed.')


def run():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    if any(OUTPUT.iterdir()):raise ValueError('output must be empty')
    sources=dict(json.loads((PRIOR/'protocol.json').read_text())['source_sha256']);sources[str(Path(__file__).resolve())]=digest(__file__)
    prior={str(PRIOR/name):digest(PRIOR/name) for name in ('protocol.json','evaluation.jsonl','completion.json')}
    assert all(digest(p)==h for p,h in {**sources,**prior}.items())
    cases=select_cases([json.loads(x) for x in (PRIOR/'evaluation.jsonl').read_text().splitlines()]);assert len(cases)==29
    p=dict(source_sha256=sources,prior_sha256=prior,cases=cases,
        selection='All29 wideband echo search-boundary failures in independent bandwidth replication. Fixed saved residual weights and exact IQ hashes.',
        search='17 delays x4 carrier seeds x4 drift seeds x5 gaps x5 relative frequencies;top24 plus original.400 iterations. Original candidate retained. Minimum explicit lstsq residual,even if nonconverged(marked). Same bounds vs gap floor.001 instead of.01;all other bounds unchanged.',
        endpoints='Material relative loss improvement>1e-9,gap-floor occupancy,convergence,local first-delay std,cancellation,conditioning,raw range error. No guard relaxation or truth-informed optimization.')
    (OUTPUT/'protocol.json').write_text(json.dumps(p,indent=2))
    rows=[]
    with ProcessPoolExecutor(max_workers=4) as pool,(OUTPUT/'diagnostics.jsonl').open('w') as stream:
        for r in pool.map(diagnose,cases):
            rows.append(r);stream.write(json.dumps(r,allow_nan=False)+'\n');stream.flush();print(f'Boundary diagnosis {len(rows)}/29',flush=True)
    (OUTPUT/'report.json').write_text(json.dumps(summarize(rows),indent=2))
    assert all(digest(path)==h for path,h in {**sources,**prior}.items())
    (OUTPUT/'completion.json').write_text(json.dumps({name:digest(OUTPUT/name) for name in ('protocol.json','diagnostics.jsonl','report.json')},indent=2))


if __name__=='__main__':run()
