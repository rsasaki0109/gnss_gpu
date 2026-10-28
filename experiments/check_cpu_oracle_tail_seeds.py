"""Supplement: check oracle local minima using estimated-weight fit as a seed."""
from pathlib import Path
import json
from scipy.optimize import minimize
from gnss_gpu.residual_dynamic_paths_cpu import _problem,fit_weighted_single_path,fit_weighted_dynamic_paths
from gnss_gpu.multipath_information_cpu import ca_passband_spectrum,CHIP_RATE
from gnss_gpu.spectral_noise_cpu import spectral_noise_shape
from experiments.diagnose_cpu_residual_tail import OUTPUT,PRIOR,select_cases,digest,bic,dense_problem
from experiments.stress_cpu_dynamic_noise import synthesize


def run():
    target=OUTPUT/'oracle_seed_check.json'
    if target.exists():raise ValueError('supplement output already exists')
    protocol=json.loads((OUTPUT/'protocol.json').read_text())
    assert all(digest(p)==h for p,h in {**protocol['source_sha256'],**protocol['prior_sha256']}.items())
    cases=select_cases([json.loads(x) for x in (PRIOR/'calibration.jsonl').read_text().splitlines()]);rows=[]
    for r in cases:
        blocks,_,_=synthesize(tuple(r[k] for k in ('profile','echo_voltage','doppler_hz','phase_rad','replica')))
        f,_=ca_passband_spectrum(3.6*CHIP_RATE);weights=spectral_noise_shape(f,r['profile'])
        solve=_problem(blocks,weights);dense,_=dense_problem(blocks,weights)
        fit=r['score']['two_path'];seed=[fit[k] for k in ('first_delay_chips','excess_chips','echo_doppler_hz')]
        improved=minimize(solve,seed,jac=True,method='L-BFGS-B',bounds=((-1,1),(.01,1.2),(-12,12)),options={'maxiter':400,'ftol':1e-13,'gtol':1e-9})
        one=fit_weighted_single_path(blocks,weights);original=fit_weighted_dynamic_paths(blocks,weights)
        loss=dense(improved.x)[0]
        rows.append(dict(profile=r['profile'],replica=r['replica'],rank=r['selected_rank'],default_oracle_score=bic(one[1],original.loss),
            additional_seed_score=bic(one[1],loss),additional_seed_converged=bool(improved.success),
            best_oracle_score=bic(one[1],min(loss,original.loss))))
    target.write_text(json.dumps(dict(source=str(Path(__file__).resolve()),source_sha256=digest(__file__),
        diagnostic_sha256=digest(OUTPUT/'diagnostics.jsonl'),scope='Retrospective extra seed diagnostic, no global-optimum claim or gate change',rows=rows),indent=2))
    for row in rows:
        if abs(row['best_oracle_score']-row['default_oracle_score'])>1e-4:print(row)

if __name__=='__main__':run()
