"""Second independent residual-rank calibration, shared fresh evaluation."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import json
from experiments.run_cpu_residual_ranked import calibrate,evaluate,digest,PROFILES,apply_gates,summarize as summarize_draw

OUTPUT=Path('output/cpu-residual-repeat')
PRIOR=Path('output/cpu-residual-ranked')


def apply_draws(row,draws):
    return dict(row,outcomes={name:apply_gates(row,draw['gates'])['outcomes'] for name,draw in draws.items()})


def summarize(rows,draws):
    cells=[];contrasts=[]
    for name,draw in draws.items():
        report=summarize_draw([dict(r,outcomes=r['outcomes'][name]) for r in rows],draw['gates'])
        cells.extend(dict(c,draw=name) for c in report['cells'])
        contrasts.extend(dict(c,draw=name,adjusted_primary_p=min(1.,4*c['paired_exact_p']) if c['primary'] else None) for c in report['contrasts'])
    variability={}
    for method in ('maximum','ranked'):
        thresholds=[d['gates'][method]['score'] for d in draws.values()]
        wins={p:[next(c['successes'] for c in cells if c['draw']==name and c['method']==method and c['profile']==p and c['echo_voltage']) for name in draws] for p in PROFILES}
        variability[method]=dict(thresholds=thresholds,absolute_threshold_difference=abs(thresholds[1]-thresholds[0]),echo_successes_on_same_data=wins)
    return dict(cells=cells,contrasts=contrasts,descriptive_variability=variability)


def run():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    if any(OUTPUT.iterdir()):raise ValueError('output must be empty')
    sources=dict(json.loads((PRIOR/'protocol.json').read_text())['source_sha256']);sources[str(Path(__file__).resolve())]=digest(__file__)
    assert all(digest(p)==h for p,h in sources.items())
    prior={str(PRIOR/name):digest(PRIOR/name) for name in ('protocol.json','calibration.jsonl','frozen_gates.json','completion.json')}
    original=[json.loads(x) for x in (PRIOR/'calibration.jsonl').read_text().splitlines()]
    draws={'original':calibrate(original)}
    assert draws['original']==json.loads((PRIOR/'frozen_gates.json').read_text())
    cal=[(p,0.,0.,0.,720000+j*1000+r) for j,p in enumerate(PROFILES) for r in range(640)]
    tasks=[(p,0.,0.,0.,730000+r) for p in PROFILES for r in range(128)]
    tasks += [(p,.8,0.,0.,740000+r) for p in PROFILES for r in range(64)]
    protocol=dict(new_calibration_tasks=cal,evaluation_tasks=tasks,source_sha256=sources,prior_sha256=prior,
        design='Preserve original640/profile calibration; one independent new640/profile draw, fixed second-largest rank, same frozen estimator.',
        primary='Ranked vs same-draw maximum on white/edge static same-phase echo, both draws, four paired exact two-sided tests Bonferroni4.',
        guarantee='Per draw: 3*BinomialCDF(1;640,.01)=.036029235 failure upper bound. Across both draws union .07205847, NOT simultaneous95%. Fixed profiles only.',
        context='PRN1,37 reference dB-Hz,200ms,3.6828MHz; echo .8 voltage,.1chip excess,zero Doppler and phase.',
        stability='Report two thresholds and success counts on identical fresh observations; only two draws, descriptive, no population variance or repeatability proof.',
        independence='Original calibration inspected development reference; new calibration independent and all four gates frozen before fresh evaluation. No best draw selection. Shared profile/draw observations not independent samples.')
    (OUTPUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    with ProcessPoolExecutor(max_workers=4) as pool:
        calibration=[]
        with (OUTPUT/'new_calibration.jsonl').open('w') as stream:
            for row in pool.map(evaluate,cal):
                calibration.append(row);stream.write(json.dumps(row,allow_nan=False)+'\n')
                if len(calibration)%160==0:stream.flush();print(f'New calibration {len(calibration)}/1920',flush=True)
        draws['new']=calibrate(calibration)
        gatepath=OUTPUT/'frozen_draws.json';gatepath.write_text(json.dumps(draws,indent=2));gatehash=digest(gatepath)
        print(f'All gates frozen: { {name:d["gates"] for name,d in draws.items()} }',flush=True)
        rows=[]
        with (OUTPUT/'evaluation.jsonl').open('w') as stream:
            for raw in pool.map(evaluate,tasks):
                row=apply_draws(raw,draws);rows.append(row);stream.write(json.dumps(row,allow_nan=False)+'\n')
                if len(rows)%96==0:stream.flush();print(f'Evaluation {len(rows)}/576',flush=True)
    (OUTPUT/'report.json').write_text(json.dumps(summarize(rows,draws),indent=2))
    assert all(digest(p)==h for p,h in {**sources,**prior}.items()) and digest(gatepath)==gatehash
    (OUTPUT/'completion.json').write_text(json.dumps(dict(new_calibration=1920,reused_calibration=1920,evaluation=576,gate_sha256=gatehash,
        new_calibration_sha256=digest(OUTPUT/'new_calibration.jsonl'),evaluation_sha256=digest(OUTPUT/'evaluation.jsonl'),report_sha256=digest(OUTPUT/'report.json')),indent=2))

if __name__=='__main__':run()
