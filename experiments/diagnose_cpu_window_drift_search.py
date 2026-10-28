"""Retrospective half-window initialization diagnosis with a frozen full model."""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
import hashlib,json
import numpy as np
from experiments.run_cpu_relative_drift_pilot import synthesize,digest
from gnss_gpu.bandwidth_drift_paths_cpu import score_drift_paths
from gnss_gpu.relative_drift_paths_cpu import RelativeDriftFit
from gnss_gpu.window_seed_relative_drift_cpu import fit_window_seeded_drift
from gnss_gpu.multipath_information_cpu import CHIP_RATE,LIGHT_SPEED

OUTPUT=Path('output/cpu-window-drift-search')
PRIOR=Path('output/cpu-relative-drift-search')


def selected_rows():
    return [json.loads(x) for x in (PRIOR/'evaluation.jsonl').read_text().splitlines()]


def evaluate(row):
    b,_,truth=synthesize(row['task'])
    assert hashlib.sha256(b.tobytes()).hexdigest()==row['observation_sha256']
    old,w=score_drift_paths(b,7.2*CHIP_RATE)
    assert hashlib.sha256(w.tobytes()).hexdigest()==row['weights_sha256']
    seed=RelativeDriftFit(**row['broad']['fit'])
    fit,diagnostics=fit_window_seeded_drift(b,w,seed,7.2*CHIP_RATE)
    error=(fit.first_delay_chips-truth)*LIGHT_SPEED/CHIP_RATE
    return dict(task=row['task'],replica=row['replica'],initial_doppler_hz=row['initial_doppler_hz'],
        relative_drift_hz_s=row['relative_drift_hz_s'],true_delay_chips=truth,
        original=row['broad'],window_diagnostics=diagnostics,
        previously_unimproved=bool(row['relative_loss_improvement']<=1e-9),
        broad=dict(fit=asdict(fit),raw_error_m=error,available=fit.locally_usable,
                   success=bool(fit.locally_usable and abs(error)<10)),
        relative_loss_improvement=(seed.loss-fit.loss)/seed.loss,
        observation_sha256=row['observation_sha256'],weights_sha256=row['weights_sha256'])


def summarize(rows):
    cells=[]
    for fd,rate in dict.fromkeys((r['initial_doppler_hz'],r['relative_drift_hz_s']) for r in rows):
        g=[r for r in rows if r['initial_doppler_hz']==fd and r['relative_drift_hz_s']==rate]
        cells.append(dict(initial_doppler_hz=fd,relative_drift_hz_s=rate,count=len(g),
            materially_lower_loss=sum(r['relative_loss_improvement']>1e-9 for r in g),
            median_relative_loss_improvement=float(np.median([r['relative_loss_improvement'] for r in g])),
            original_median_absolute_error_m=float(np.median([abs(r['original']['raw_error_m']) for r in g])),
            broad_median_absolute_error_m=float(np.median([abs(r['broad']['raw_error_m']) for r in g])),
            broad_max_absolute_error_m=max(abs(r['broad']['raw_error_m']) for r in g),
            broad_unavailable=sum(not r['broad']['available'] for r in g),
            broad_nonconverged=sum(not r['broad']['fit']['converged'] for r in g),
            median_absolute_drift_error_hz_s=float(np.median([abs(r['broad']['fit']['relative_drift_hz_s']-rate) for r in g]))))
    remaining=[r for r in rows if r['previously_unimproved'] and r['initial_doppler_hz']*(r['initial_doppler_hz']+.8*r['relative_drift_hz_s'])<0]
    return dict(cells=cells,previously_unimproved_count=len(remaining),
        previously_unimproved_lower_loss=sum(r['relative_loss_improvement']>1e-9 for r in remaining),scope='Retrospective diagnosis: all32 zero-crossing cases plus first2 replicas of each other cell. No held-out claim, significance test or global optimum proof.')


def run():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    if any(OUTPUT.iterdir()):raise ValueError('output must be empty')
    sources=dict(json.loads((PRIOR/'protocol.json').read_text())['source_sha256'])
    for path in (__file__,'python/gnss_gpu/window_seed_relative_drift_cpu.py'):
        sources[str(Path(path).resolve())]=digest(path)
    assert all(digest(path)==h for path,h in sources.items())
    prior_completion=json.loads((PRIOR/'completion.json').read_text())
    assert all(digest(PRIOR/name)==h for name,h in prior_completion.items())
    rows=selected_rows();assert len(rows)==40
    protocol=dict(source_sha256=sources,prior_completion=prior_completion,tasks=[r['task'] for r in rows],
        design='Retrospective all32 zero-crossing cases and first2 replicas in each of4 control cells;same observations/model/weights/bounds/guards. Existing result retained as candidate.',
        search='Split40blocks into two20block windows;constant two-carrier fits with independent complex amplitudes. Eight starts/window,keep3 converged fits;pair midpoint frequencies at.2/.6s to estimate initial frequency and drift. Clip to existing full bounds. Up to18 paired starts using averaged or prior spatial parameters plus prior fit. Prior fit remains candidate;400 full iterations.',
        endpoint='Residual-based selection only;no generating parameters passed to fitter. Material loss improvement threshold relative1e-9. Record direct range/drift errors after fit. No confidence intervals or significance claim.',
        limits='Finite multistart diagnosis,not global optimum proof or independent validation. No model-order gate recalibration. Synthetic fixed-code receiver versus moving-code signal; no PVT/forest/rain.')
    (OUTPUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    results=[]
    with ProcessPoolExecutor(max_workers=4) as pool,(OUTPUT/'evaluation.jsonl').open('w') as stream:
        for r in pool.map(evaluate,rows):
            results.append(r);stream.write(json.dumps(r,allow_nan=False)+'\n');stream.flush()
            if len(results)%8==0:print(f'Window search {len(results)}/40',flush=True)
    (OUTPUT/'report.json').write_text(json.dumps(summarize(results),indent=2))
    assert all(digest(path)==h for path,h in sources.items())
    (OUTPUT/'completion.json').write_text(json.dumps({name:digest(OUTPUT/name) for name in
        ('protocol.json','evaluation.jsonl','report.json')},indent=2))


if __name__=='__main__':run()
