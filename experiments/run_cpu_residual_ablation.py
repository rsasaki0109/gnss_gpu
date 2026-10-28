"""2x2 ablation with preserved paired calibration and fresh evaluation."""
from dataclasses import asdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import json
from gnss_gpu.calibrated_path_selection_cpu import calibrate_path_gate,PathSelectionGate
from experiments.run_cpu_residual_dynamic import evaluate,digest,PROFILES
from experiments.run_cpu_dynamic_selection import select,summarize as summarize_single
from experiments.validate_cpu_bandlimited_multipath import paired_summary

OUTPUT=Path('output/cpu-residual-ablation')
PRIOR=Path('output/cpu-residual-dynamic')
ARMS={'unweighted_white':('old_score','white'),'unweighted_multi':('old_score','multi'),
      'weighted_white':('score','white'),'weighted_multi':('score','multi')}


def calibrate(rows):
    strata={}
    for model,key in [('unweighted','old_score'),('weighted','score')]:
        strata[model]={p:asdict(calibrate_path_gate([r[key]['bic_improvement'] for r in rows if r['profile']==p],confidence_failure=.05/3)) for p in PROFILES}
    arms={}
    for arm,(key,coverage) in ARMS.items():
        model='weighted' if key=='score' else 'unweighted'
        group=strata[model]
        arms[arm]=group['white'] if coverage=='white' else asdict(PathSelectionGate(max(g['score'] for g in group.values()),1224,.01,.05))
    return dict(strata=strata,arms=arms)


def apply_gates(row,gates):
    outcomes={}
    for arm,(key,_) in ARMS.items():
        chosen=select(dict(row,score=row[key]),PathSelectionGate(**gates['arms'][arm]))
        outcomes[arm]={k:chosen[k] for k in ('selected','error_m','success','forced_error_m','forced_success','score_exceeds')}
    return dict(row,outcomes=outcomes)


def summarize(rows):
    cells=[];contrasts=[]
    for profile in PROFILES:
        group=[r for r in rows if r['profile']==profile]
        for arm in ARMS:
            cells.extend(dict(c,profile=profile,arm=arm) for c in summarize_single([dict(r,**r['outcomes'][arm]) for r in group])['cells'])
        echo=sorted([r for r in group if r['echo_voltage']],key=lambda r:r['replica'])
        for name,a,b in [('weighting','unweighted_white','weighted_white'),('coverage','weighted_white','weighted_multi')]:
            paired=paired_summary([r['outcomes'][a]['success'] for r in echo],[r['outcomes'][b]['success'] for r in echo])
            primary=profile in ('white','edge')
            contrasts.append(dict(profile=profile,contrast=name,**paired,primary=primary,adjusted_primary_p=min(1.,4*paired['paired_exact_p']) if primary else None))
    return dict(cells=cells,contrasts=contrasts)


def run():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    if any(OUTPUT.iterdir()):raise ValueError('output must be empty')
    sources=dict(json.loads((PRIOR/'protocol.json').read_text())['source_sha256'])
    sources[str(Path(__file__).resolve())]=digest(__file__)
    hashes={str(PRIOR/name):digest(PRIOR/name) for name in ('protocol.json','calibration.jsonl','frozen_gate.json','completion.json')}
    assert all(digest(p)==h for p,h in sources.items())
    calibration=[json.loads(x) for x in (PRIOR/'calibration.jsonl').read_text().splitlines()]
    gates=calibrate(calibration)
    assert gates['arms']['weighted_multi']==json.loads((PRIOR/'frozen_gate.json').read_text())['combined']
    tasks=[(p,0.,0.,0.,660000+r) for p in PROFILES for r in range(128)]
    tasks += [(p,.8,0.,0.,670000+r) for p in PROFILES for r in range(64)]
    protocol=dict(evaluation_tasks=tasks,source_sha256=sources,prior_sha256=hashes,
        calibration='Reuse all preserved 408 paired scores/profile for both models; all four arms use MAXIMUM rule. No recalibration fitting or rank-vs-max confound.',
        guarantee='Each model separately: joint failure across its three calibrated profiles <=3*.99**408 <.05. Across both models only union <.10, NOT joint95%. White-only arms cover white only.',
        primary='White and edge static same-phase echoes: weighting (weighted_white vs unweighted_white), coverage (weighted_multi vs weighted_white). Four paired exact two-sided tests Bonferroni4.',
        scope='Frozen estimators,37 reference dB-Hz,200ms,3.6828MHz; echo .8 voltage,.1chip gap,zero Doppler/phase. No generalization across powers/dynamics.',
        independence='Prior calibration already inspected development reference; all fresh null128/profile and static echo64/profile; seeds paired across profiles, no pooling.',
        limitations='Weighting contrast includes necessary per-estimator recalibration on same samples. Prior rank-to-max effect not estimated. One calibration draw, conditional evaluation only.')
    (OUTPUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    gatepath=OUTPUT/'frozen_gates.json';gatepath.write_text(json.dumps(gates,indent=2));gatehash=digest(gatepath)
    print(f'Frozen gates: {gates["arms"]}',flush=True)
    rows=[]
    with ProcessPoolExecutor(max_workers=4) as pool,(OUTPUT/'evaluation.jsonl').open('w') as stream:
        for raw in pool.map(evaluate,tasks):
            row=apply_gates(raw,gates);rows.append(row);stream.write(json.dumps(row,allow_nan=False)+'\n')
            if len(rows)%96==0:stream.flush();print(f'Evaluation {len(rows)}/576',flush=True)
    (OUTPUT/'report.json').write_text(json.dumps(summarize(rows),indent=2))
    assert all(digest(p)==h for p,h in {**sources,**hashes}.items()) and digest(gatepath)==gatehash
    (OUTPUT/'completion.json').write_text(json.dumps(dict(reused_calibration=1224,evaluation=576,gate_sha256=gatehash,
        evaluation_sha256=digest(OUTPUT/'evaluation.jsonl'),report_sha256=digest(OUTPUT/'report.json')),indent=2))

if __name__=='__main__':run()
