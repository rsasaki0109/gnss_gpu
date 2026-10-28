"""Retrospective numerical and oracle-noise diagnosis of null-score tails."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import hashlib,json
import numpy as np
from scipy.optimize import minimize,minimize_scalar
from gnss_gpu.multipath_information_cpu import ca_passband_spectrum,CHIP_RATE
from gnss_gpu.dynamic_multipath_cpu import temporal_echo,DynamicFit
from gnss_gpu.residual_dynamic_paths_cpu import _problem,estimate_residual_weights,fit_weighted_dynamic_paths,fit_weighted_single_path
from gnss_gpu.spectral_noise_cpu import spectral_noise_shape
from experiments.stress_cpu_dynamic_noise import synthesize,PROFILES
from experiments.run_cpu_residual_dynamic import digest

PRIOR=Path('output/cpu-residual-ranked')
OUTPUT=Path('output/cpu-residual-tail-diagnosis')


def select_cases(rows):
    chosen=[]
    for profile in PROFILES:
        ranked=sorted([r for r in rows if r['profile']==profile],key=lambda r:(-r['score']['bic_improvement'],r['replica']))
        for rank in (1,2,3,320):chosen.append(dict(ranked[rank-1],selected_rank=rank))
    return chosen


def dense_problem(blocks,weights):
    """Explicit full design matrix and lstsq, independent of separable Gram solve."""
    f,s=ca_passband_spectrum(3.6*CHIP_RATE);inv=1/np.sqrt(weights/weights.mean())
    y=blocks*inv;y=y/np.max(np.abs(y));s=s*inv;slope=-2j*np.pi*f/CHIP_RATE
    def solve(parameters):
        first,gap,fd=parameters
        a0=np.broadcast_to(s*np.exp(slope*first),y.shape)
        a1=temporal_echo(fd)[0][:,None]*s*np.exp(slope*(first+gap))
        a=np.column_stack([a0.ravel(),a1.ravel()])
        amp=np.linalg.lstsq(a,y.ravel(),rcond=None)[0];residual=y.ravel()-a@amp
        return float(np.vdot(residual,residual).real),float(np.linalg.cond(a))
    def single(delay):
        a=s*np.exp(slope*delay);amp=np.vdot(a,y.sum(axis=0))/(10*np.vdot(a,a))
        residual=y-amp*a
        return float(np.vdot(residual,residual).real)
    return solve,single


def bic(one,two):return float(147300*np.log(one/two)-4*np.log(147300))


def diagnose(row):
    task=tuple(row[k] for k in ('profile','echo_voltage','doppler_hz','phase_rad','replica'))
    blocks,_,_=synthesize(task)
    assert hashlib.sha256(blocks.tobytes()).hexdigest()==row['observation_sha256']
    weights=estimate_residual_weights(blocks,DynamicFit(**row['old_score']['two_path']))
    assert hashlib.sha256(weights.tobytes()).hexdigest()==row['weights_sha256']
    original=row['score'];fit=original['two_path']
    parameters=[fit[k] for k in ('first_delay_chips','excess_chips','echo_doppler_hz')]
    dense,single=dense_problem(blocks,weights)
    original_two,condition=dense(parameters);original_one=single(original['single_delay_chips'])
    np.testing.assert_allclose(original_two,fit['loss'],rtol=1e-12)
    np.testing.assert_allclose(original_one,original['single_loss'],rtol=1e-12)
    grid=np.linspace(-1,2.2,641)
    indices=np.argsort([single(d) for d in grid])[:16]
    onefits=[minimize_scalar(single,bounds=(grid[max(0,int(i)-1)],grid[min(640,int(i)+1)]),method='bounded',options={'xatol':1e-12}) for i in indices]
    one=min([dict(delay=original['single_delay_chips'],loss=original_one,converged=original['single_converged'])]+[dict(delay=float(f.x),loss=float(f.fun),converged=bool(f.success)) for f in onefits],key=lambda r:r['loss'])
    solve=_problem(blocks,weights)
    seeds=[(d,g,fd) for d in np.linspace(-1,1,25) for g in (.03,.1,.3,.7,1.2) for fd in (-12.,-8.,-4.,-1.,0.,1.,4.,8.,12.)]
    seeds=sorted(seeds,key=lambda p:solve(p)[0])[:32]+[parameters]
    fits=[minimize(solve,seed,jac=True,method='L-BFGS-B',bounds=((-1,1),(.01,1.2),(-12,12)),options={'maxiter':400,'ftol':1e-13,'gtol':1e-9}) for seed in seeds]
    candidates=[dict(parameters=parameters,loss=original_two,converged=fit['converged'])]
    candidates += [dict(parameters=f.x.tolist(),loss=dense(f.x)[0],converged=bool(f.success)) for f in fits]
    two=min(candidates,key=lambda r:r['loss'])
    f,_=ca_passband_spectrum(3.6*CHIP_RATE);truth=spectral_noise_shape(f,row['profile'])
    oracle_one=fit_weighted_single_path(blocks,truth);oracle_two=fit_weighted_dynamic_paths(blocks,truth)
    return dict(profile=row['profile'],replica=row['replica'],selected_rank=row['selected_rank'],original_score=original['bic_improvement'],
        explicit_score=bic(original_one,original_two),original_template_condition=condition,
        original_locally_usable=fit['locally_usable'],original_reason=fit['reason'],
        expanded_single=one,expanded_two=two,expanded_score=bic(one['loss'],two['loss']),
        relative_single_improvement=(original_one-one['loss'])/original_one,
        relative_two_improvement=(original_two-two['loss'])/original_two,
        oracle_score=bic(oracle_one[1],oracle_two.loss),oracle_converged=oracle_one[2] and oracle_two.converged,
        weight_relative_rms=float(np.sqrt(np.mean(((weights-truth)/truth)**2))))


def summarize(rows):
    return dict(cases=len(rows),max_explicit_score_difference=max(abs(r['explicit_score']-r['original_score']) for r in rows),
        materially_better_single=sum(r['relative_single_improvement']>1e-9 for r in rows),
        materially_better_two=sum(r['relative_two_improvement']>1e-9 for r in rows),
        max_expanded_score_change=max(abs(r['expanded_score']-r['original_score']) for r in rows),
        expanded_nonconverged=sum(not(r['expanded_single']['converged'] and r['expanded_two']['converged']) for r in rows),
        oracle_nonconverged=sum(not r['oracle_converged'] for r in rows))


def run():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    if any(OUTPUT.iterdir()):raise ValueError('output must be empty')
    sources=dict(json.loads((PRIOR/'protocol.json').read_text())['source_sha256']);sources[str(Path(__file__).resolve())]=digest(__file__)
    assert all(digest(p)==h for p,h in sources.items())
    prior={str(PRIOR/name):digest(PRIOR/name) for name in ('protocol.json','calibration.jsonl','frozen_gates.json')}
    cases=select_cases([json.loads(x) for x in (PRIOR/'calibration.jsonl').read_text().splitlines()])
    protocol=dict(source_sha256=sources,prior_sha256=prior,cases=[[r['profile'],r['replica'],r['selected_rank']] for r in cases],
        selection='Top1/2/3 and rank320 from640 per profile; all12 fixed before diagnostics. Retrospective tail sample, not fresh performance evidence.',
        diagnostics='Explicit full-matrix lstsq residual; single grid641 top16 bounded refinements; two-path grid25x5x9 top32+original,400 iterations. Weights frozen from original preliminary fit.',
        oracle='Known simulated PSD used for diagnostic refit only; no deployment or calibrated inference claim. Same optimizer may still have local minima.',
        limits='No guarantee of global optimum. Finite selected cases cannot establish full tail distribution. No gate/receiver changes or deletion of calibration cases.')
    (OUTPUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    rows=[]
    with ProcessPoolExecutor(max_workers=4) as pool,(OUTPUT/'diagnostics.jsonl').open('w') as stream:
        for row in pool.map(diagnose,cases):
            rows.append(row);stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush();print(f'Diagnosis {len(rows)}/12',flush=True)
    (OUTPUT/'report.json').write_text(json.dumps(summarize(rows),indent=2))
    assert all(digest(p)==h for p,h in {**sources,**prior}.items())
    (OUTPUT/'completion.json').write_text(json.dumps(dict(cases=len(rows),diagnostics_sha256=digest(OUTPUT/'diagnostics.jsonl'),report_sha256=digest(OUTPUT/'report.json')),indent=2))

if __name__=='__main__':run()
