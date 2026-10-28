"""Retrospective fixed-weight boundary diagnosis; never a detector update."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import hashlib,json
import numpy as np
from scipy.optimize import minimize
from gnss_gpu.common_carrier_paths_cpu import _problem,estimate_weights,CarrierFit
from gnss_gpu.dynamic_multipath_cpu import temporal_echo
from gnss_gpu.multipath_information_cpu import ca_passband_spectrum,CHIP_RATE,LIGHT_SPEED
from experiments.stress_cpu_common_carrier_phase import synthesize,digest

PRIOR=Path('output/cpu-common-carrier-phase-stress')
OUTPUT=Path('output/cpu-common-carrier-boundary')
KEYS=('first_delay_chips','common_carrier_hz','excess_chips','relative_doppler_hz')


def select_cases(rows):
    return [r for r in rows if r['selected']['model']=='ambiguous' and r['selected']['reason']=='search boundary']


def dense_problem(blocks,weights):
    f,s=ca_passband_spectrum(3.6*CHIP_RATE)
    inv=1/np.sqrt(weights/weights.mean());y=blocks*inv;y=(y/np.max(np.abs(y))).ravel();s=s*inv
    def solve(p):
        d,c,g,r=p
        a=np.column_stack([(temporal_echo(c)[0][:,None]*s*np.exp(-2j*np.pi*f*d/CHIP_RATE)).ravel(),
                           (temporal_echo(c+r)[0][:,None]*s*np.exp(-2j*np.pi*f*(d+g)/CHIP_RATE)).ravel()])
        amp=np.linalg.lstsq(a,y,rcond=None)[0];prediction=a@amp;residual=y-prediction
        cancellation=sum(np.linalg.norm(a[:,i]*amp[i]) for i in range(2))/np.linalg.norm(prediction)
        return dict(loss=float(np.vdot(residual,residual).real),template_condition=float(np.linalg.cond(a)),cancellation_ratio=float(cancellation))
    return solve


def diagnostic(solve,dense,p,converged,truth):
    detail=dense(p);loss,a,jac=solve(p,True)
    projected=jac-a@np.linalg.lstsq(a,jac,rcond=None)[0]
    info=np.real(projected.conj().T@projected);eig=np.linalg.eigvalsh(info);std=None
    if eig[0]>max(1e-20,1e-12*eig[-1]):std=float(np.sqrt(max(0,detail['loss']/(2*len(a)-8)*np.linalg.inv(info)[0,0])))
    return dict(detail,parameters=list(map(float,p)),converged=bool(converged),first_std_chips=std,
        error_m=float((p[0]-truth)*LIGHT_SPEED/CHIP_RATE),separable_explicit_difference=float(loss-detail['loss']))


def diagnose(row):
    task=tuple(row[k] for k in ('echo_voltage','residual_carrier_hz','relative_doppler_hz','phase_rad','replica'))
    blocks,_,truth=synthesize(task)
    assert hashlib.sha256(blocks.tobytes()).hexdigest()==row['observation_sha256']
    weights=estimate_weights(blocks,CarrierFit(**row['fits']['preliminary']))
    assert hashlib.sha256(weights.tobytes()).hexdigest()==row['weights_sha256']
    solve=_problem(blocks,weights,True);dense=dense_problem(blocks,weights)
    old=row['fits']['two'];parameters=[old[k] for k in KEYS]
    original=diagnostic(solve,dense,parameters,old['converged'],truth)
    np.testing.assert_allclose(original['loss'],old['loss'],rtol=1e-12)
    seeds=[(d,c,g,r) for d in np.linspace(-1,1,25) for c in (-1.5,0.,1.5)
           for g in (.01,.05,.2,.5,1.2) for r in (-12.,-5.,-1.,-.1,0.,.1,1.,5.,12.)]
    seeds=sorted(seeds,key=lambda p:solve(p)[0])[:32]+[parameters]
    results={}
    for name,floor in [('same_bounds',.01),('relaxed_floor',.001)]:
        fits=[minimize(solve,seed,jac=True,method='L-BFGS-B',bounds=((-1,1),(-2,2),(floor,1.2),(-12,12)),
            options={'maxiter':400,'ftol':1e-13,'gtol':1e-9}) for seed in seeds]
        candidates=[dict(parameters=parameters,loss=original['loss'],converged=old['converged'])]+[
            dict(parameters=f.x.tolist(),loss=dense(f.x)['loss'],converged=bool(f.success)) for f in fits]
        best=min(candidates,key=lambda r:r['loss'])
        results[name]=dict(diagnostic(solve,dense,best['parameters'],best['converged'],truth),
            relative_loss_improvement=(original['loss']-best['loss'])/original['loss'],
            at_gap_floor=best['parameters'][2]-floor<1e-4,
            successful_starts=sum(f.success for f in fits),attempted_starts=len(fits))
    return dict(replica=row['replica'],relative_doppler_hz=row['relative_doppler_hz'],phase_rad=row['phase_rad'],
        observation_sha256=row['observation_sha256'],weights_sha256=row['weights_sha256'],original=original,**results)


def summarize(rows):
    return dict(cases=len(rows),max_original_dense_difference=max(abs(r['original']['separable_explicit_difference']) for r in rows),
        methods={name:dict(materially_better=sum(r[name]['relative_loss_improvement']>1e-9 for r in rows),
            at_gap_floor=sum(r[name]['at_gap_floor'] for r in rows),best_nonconverged=sum(not r[name]['converged'] for r in rows),
            weak_local_information=sum(r[name]['first_std_chips'] is None or r[name]['first_std_chips']>.1 for r in rows),
            max_relative_loss_improvement=max(r[name]['relative_loss_improvement'] for r in rows),
            cancellation_range=[min(r[name]['cancellation_ratio'] for r in rows),max(r[name]['cancellation_ratio'] for r in rows)]) for name in ('same_bounds','relaxed_floor')})


def run():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    if any(OUTPUT.iterdir()):raise ValueError('output must be empty')
    sources=dict(json.loads((PRIOR/'protocol.json').read_text())['source_sha256']);sources[str(Path(__file__).resolve())]=digest(__file__)
    prior={str(PRIOR/name):digest(PRIOR/name) for name in ('protocol.json','evaluation.jsonl','completion.json')}
    assert all(digest(p)==h for p,h in {**sources,**prior}.items())
    cases=select_cases([json.loads(x) for x in (PRIOR/'evaluation.jsonl').read_text().splitlines()])
    assert len(cases)==13
    protocol=dict(source_sha256=sources,prior_sha256=prior,cases=[[r['replica'],r['relative_doppler_hz'],r['phase_rad']] for r in cases],
        selection='ALL13 previously unavailable search-boundary observations; retrospective diagnostic, not heldout performance.',
        weights='Regenerate original fitted residual weights from saved preliminary fit, require exact SHA256. Freeze throughout both searches.',
        search='25 delays x3 common carriers x5 gaps x9 relative Dopplers, top32+original, max400 iterations. Same bounds vs gap floor .001 instead of .01; all other bounds unchanged. Original candidate always retained; select minimum explicit full-design lstsq SSE.',
        material='Relative loss decrease >1e-9, fixed before diagnostics. Record convergence, local std, boundary, cancellation and template conditioning.',
        limits='Finite multistart cannot prove global optimality. Relaxed-floor minimum is not a valid accepted range or recalibrated detector. No receiver/gate changes.')
    (OUTPUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    rows=[]
    with ProcessPoolExecutor(max_workers=4) as pool,(OUTPUT/'diagnostics.jsonl').open('w') as stream:
        for row in pool.map(diagnose,cases):
            rows.append(row);stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush();print(f'Diagnosis {len(rows)}/13',flush=True)
    (OUTPUT/'report.json').write_text(json.dumps(summarize(rows),indent=2))
    assert all(digest(p)==h for p,h in {**sources,**prior}.items())
    (OUTPUT/'completion.json').write_text(json.dumps(dict(cases=len(rows),diagnostics_sha256=digest(OUTPUT/'diagnostics.jsonl'),report_sha256=digest(OUTPUT/'report.json')),indent=2))


if __name__=='__main__':run()
