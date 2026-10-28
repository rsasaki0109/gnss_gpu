"""Post-hoc loss-surface diagnosis; never changes receiver availability."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import json
import numpy as np
from scipy.optimize import minimize
from gnss_gpu.multipath_information_cpu import ca_passband_spectrum, CHIP_RATE
from gnss_gpu.noise_aware_multipath_cpu import estimate_noise_weights
from experiments.run_cpu_matched_dynamic_reflection import synthesize, digest

PRIOR=Path('output/cpu-matched-dynamic-reflection')
OUTPUT=Path('output/cpu-path-boundary-diagnosis')


def objective(observation, weights):
    f,s=ca_passband_spectrum(3.6*CHIP_RATE)
    inv=1/np.sqrt(weights/weights.mean());s=s*inv;y=observation*inv
    y=y/np.max(np.abs(y));slope=-2j*np.pi*f/CHIP_RATE
    def solve(parameters):
        first,gap=parameters
        a=np.column_stack([s*np.exp(slope*first),s*np.exp(slope*(first+gap))])
        amp=np.linalg.lstsq(a,y,rcond=None)[0];residual=y-a@amp
        derivatives=slope[:,None]*a*amp
        jacobian=np.column_stack([derivatives.sum(axis=1),derivatives[:,1]])
        gradient=-2*np.real(jacobian.conj().T@residual)
        loss=float(np.vdot(residual,residual).real)
        return loss,gradient,a,amp,derivatives
    return solve


def describe(solve,parameters):
    loss,gradient,a,amp,d=solve(parameters)
    projected=d-a@np.linalg.lstsq(a,d,rcond=None)[0]
    info=np.real(projected.conj().T@projected);eig=np.linalg.eigvalsh(info)
    std=None
    if eig[0]>max(1e-20,1e-12*eig[-1]):std=float(np.sqrt(max(0,loss/(2*len(a)-6)*np.linalg.inv(info)[0,0])))
    return dict(parameters=list(map(float,parameters)),loss=loss,gradient=gradient.tolist(),first_std_chips=std,
        template_condition=float(np.linalg.cond(a)),
        cancellation_ratio=float(sum(np.linalg.norm(a[:,i]*amp[i]) for i in range(2))/np.linalg.norm(a@amp)))


def search(solve,minimum,original):
    seeds=[(d,g) for d in np.linspace(-1,1,41) for g in (minimum,.003,.01,.03,.1,.3,.6,1.2) if g>=minimum]
    seeds=sorted(seeds,key=lambda p:solve(p)[0])[:32]+[original]
    fits=[minimize(lambda p:solve(p)[:2],seed,jac=True,method='L-BFGS-B',bounds=((-1,1),(minimum,1.2)),
        options={'maxiter':300,'ftol':1e-13,'gtol':1e-9}) for seed in seeds]
    best=min(fits,key=lambda fit:fit.fun)
    return dict(describe(solve,best.x),converged=bool(best.success),successful_starts=sum(bool(f.success) for f in fits),starts=len(fits))


def evaluate(row):
    blocks,_,_,_=synthesize(row['doppler_hz'],row['phase_rad'],row['replica'])
    import hashlib
    assert hashlib.sha256(blocks.tobytes()).hexdigest()==row['observation_sha256']['dynamic']
    solve=objective(blocks.mean(axis=0),estimate_noise_weights(blocks))
    old=row['arms']['dynamic']['fit'];parameters=(old['first_delay_chips'],old['excess_chips'])
    original=describe(solve,parameters)
    assert np.isclose(original['first_std_chips'],old['first_std_chips'],rtol=1e-10)
    same=search(solve,.01,parameters);relaxed=search(solve,.001,parameters)
    # Noise-free truth is used only to inspect the observed loss, never to seed search.
    truth=describe(solve,(row['arms']['dynamic']['true_delay_chips'],.1))
    return dict(doppler_hz=row['doppler_hz'],phase_rad=row['phase_rad'],replica=row['replica'],
        prior_reason=row['arms']['dynamic']['selected']['reason'],original=original,
        dense_same_bounds=same,relaxed_gap=relaxed,truth_parameters_loss=truth['loss'])


def summarize(rows):
    return dict(cases=len(rows),boundary_cases=sum(r['prior_reason']=='two-path search boundary' for r in rows),
        same_bounds_materially_better=sum(r['original']['loss']-r['dense_same_bounds']['loss']>1e-9*max(1,r['original']['loss']) for r in rows),
        same_bounds_at_gap_floor=sum(r['dense_same_bounds']['parameters'][1] < .0101 for r in rows),
        relaxed_at_gap_floor=sum(r['relaxed_gap']['parameters'][1]<.0011 for r in rows),
        relaxed_materially_better=sum(r['original']['loss']-r['relaxed_gap']['loss']>1e-9*max(1,r['original']['loss']) for r in rows),
        relaxed_std_above_guard=sum(r['relaxed_gap']['first_std_chips'] is None or r['relaxed_gap']['first_std_chips']>.1 for r in rows),
        same_bounds_nonconverged=sum(not r['dense_same_bounds']['converged'] for r in rows),
        relaxed_nonconverged=sum(not r['relaxed_gap']['converged'] for r in rows),
        original_cancellation_range=[min(r['original']['cancellation_ratio'] for r in rows),max(r['original']['cancellation_ratio'] for r in rows)],
        relaxed_cancellation_range=[min(r['relaxed_gap']['cancellation_ratio'] for r in rows),max(r['relaxed_gap']['cancellation_ratio'] for r in rows)])


def run():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    if any(OUTPUT.iterdir()):raise ValueError('output must be empty')
    prior=json.loads((PRIOR/'protocol.json').read_text());sources=dict(prior['source_sha256'])
    sources[str(Path(__file__).resolve())]=digest(__file__)
    hashes={str(PRIOR/name):digest(PRIOR/name) for name in ('protocol.json','evaluation.jsonl','report.json','completion.json')}
    assert all(digest(p)==h for p,h in sources.items())
    rows=[json.loads(line) for line in (PRIOR/'evaluation.jsonl').read_text().splitlines()]
    selected=[r for r in rows if r['arms']['dynamic']['error_m'] is None]
    protocol=dict(source_sha256=sources,prior_sha256=hashes,
        cases=[[r['doppler_hz'],r['phase_rad'],r['replica']] for r in selected],
        selection='All 29 unavailable dynamic-arm records from prior study; retrospective diagnostic, no fresh heldout performance claim.',
        search='41 first-delay grid points x gap seeds; best 32 plus original fit; same bounds and relaxed gap floor .001; maxiter300.',
        material_loss_improvement='original loss - candidate loss > 1e-9*max(1,original loss)',
        limits='No guarantee of global optimum. Smaller loss is not better physical identification. Guard/receiver/gate unchanged; relaxed model has no calibrated selector.')
    (OUTPUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    results=[]
    with ProcessPoolExecutor(max_workers=4) as pool,(OUTPUT/'diagnostics.jsonl').open('w') as stream:
        for row in pool.map(evaluate,selected):
            results.append(row);stream.write(json.dumps(row,allow_nan=False)+'\n')
            if len(results)%5==0:stream.flush();print(f'Diagnosed {len(results)}/29',flush=True)
    (OUTPUT/'report.json').write_text(json.dumps(summarize(results),indent=2))
    assert all(digest(p)==h for p,h in {**sources,**hashes}.items())
    (OUTPUT/'completion.json').write_text(json.dumps(dict(cases=len(results),diagnostics_sha256=digest(OUTPUT/'diagnostics.jsonl'),report_sha256=digest(OUTPUT/'report.json')),indent=2))

if __name__=='__main__':run()
