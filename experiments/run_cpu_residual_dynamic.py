"""Independent multi-profile calibration of residual-weighted dynamic fitting."""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
import hashlib,json
import numpy as np
from gnss_gpu.calibrated_dynamic_paths_cpu import DynamicScore,fit_single_path
from gnss_gpu.residual_dynamic_paths_cpu import score_residual_dynamic_paths
from gnss_gpu.calibrated_path_selection_cpu import PathSelectionGate,calibrate_path_gate
from experiments.stress_cpu_dynamic_noise import synthesize,PROFILES
from experiments.run_cpu_dynamic_selection import select,summarize as summarize_single,digest
from experiments.validate_cpu_bandlimited_multipath import paired_summary

OUTPUT=Path('output/cpu-residual-dynamic')
PRIOR=Path('output/cpu-dynamic-noise-stress')
OLD=Path('output/cpu-dynamic-ranked-gate/frozen_gates.json')


def evaluate(task):
    blocks,_,delay=synthesize(task)
    score,preliminary,weights=score_residual_dynamic_paths(blocks)
    first,loss,ok=fit_single_path(blocks);n=2*blocks.size;tiny=np.finfo(float).tiny
    old=DynamicScore(first,loss,ok,float(n*np.log(max(loss,tiny)/max(preliminary.loss,tiny))-4*np.log(n)),preliminary)
    profile,ratio,fd,phase,replica=task
    return dict(profile=profile,echo_voltage=ratio,doppler_hz=fd,phase_rad=phase,replica=replica,true_delay_chips=delay,
        score=asdict(score),old_score=asdict(old),weights_sha256=hashlib.sha256(weights.tobytes()).hexdigest(),
        weight_min=float(weights.min()),weight_max=float(weights.max()),
        observation_sha256=hashlib.sha256(blocks.tobytes()).hexdigest())


def calibrate(rows):
    strata={p:calibrate_path_gate([r['score']['bic_improvement'] for r in rows if r['profile']==p],confidence_failure=.05/3) for p in PROFILES}
    return dict(strata={p:asdict(g) for p,g in strata.items()},combined=asdict(PathSelectionGate(max(g.score for g in strata.values()),sum(g.calibration_count for g in strata.values()),.01,.05)))


def apply_gates(row,new,old):
    outcomes={}
    for name,score,gate in [('weighted',row['score'],new),('old',row['old_score'],old)]:
        r=select(dict(row,score=score),PathSelectionGate(**gate))
        outcomes[name]={k:r[k] for k in ('selected','error_m','success','forced_error_m','forced_success','score_exceeds')}
    return dict(row,outcomes=outcomes)


def summarize(rows):
    cells=[]
    for p in PROFILES:
        for method in ('old','weighted'):
            group=[dict(r,**r['outcomes'][method]) for r in rows if r['profile']==p]
            cells.extend(dict(c,profile=p,method=method) for c in summarize_single(group)['cells'])
    center=[r for r in rows if r['profile']=='center' and not r['echo_voltage']]
    primary=paired_summary([r['outcomes']['old']['selected']['model']!='two_path' for r in center],
        [r['outcomes']['weighted']['selected']['model']!='two_path' for r in center])
    return dict(cells=cells,primary_center_no_false_selection=primary)


def run():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    if any(OUTPUT.iterdir()):raise ValueError('output must be empty')
    sources=dict(json.loads((PRIOR/'protocol.json').read_text())['source_sha256'])
    for p in (Path(__file__).resolve(),Path('python/gnss_gpu/residual_dynamic_paths_cpu.py')):sources[str(p)]=digest(p)
    assert all(digest(p)==h for p,h in sources.items())
    old=json.loads(OLD.read_text())['gates']['ranked'];oldhash=digest(OLD)
    cal=[(p,0.,0.,0.,630000+1000*j+r) for j,p in enumerate(PROFILES) for r in range(408)]
    null=[(p,0.,0.,0.,640000+r) for p in PROFILES for r in range(128)]
    echo=[(p,.8,fd,0.,650000+r) for p in PROFILES for fd in (0.,.1,.7) for r in range(16)]
    protocol=dict(calibration_tasks=cal,evaluation_tasks=null+echo,source_sha256=sources,old_gate_path=str(OLD),old_gate_sha256=oldhash,old_gate=old,
        estimator='Unweighted dynamic fit -> fitted two-path residual -> mean residual energy per harmonic -> 101-bin smoothed weights/floor1e-6 -> one weighted refit. Same weights for one/two path score.',
        data_reuse='Weights depend on fitted observations; no independence or chi-square claim. Entire pipeline calibrated, including invalid preliminary fits used as nuisance predictions.',
        calibration='408 iid null draws per white/center/edge profile, max per profile, max of three gates. Union failure bound 3*.99**408 <= .05 for <=1% null tail in each fixed profile.',
        primary='Center null false-two-path selection reduction vs old white ranked gate; paired exact two-sided test alpha .05. Unavailable and range failures reported separately.',
        comparison='Joint weighting and broader recalibration, not isolation of weighting alone. Echo cells descriptive, no best-threshold selection.',
        context='PRN1,37 reference dB-Hz,200ms,3.6828MHz; static wiped direct path; colored harmonic noise stationary; echo .8 voltage,.1chip excess,phase0.',
        independence='Calibration IDs disjoint per profile and from evaluation. Null128 shared across profiles; echo16 shared across profiles/Dopplers. No pooling as independent.',
        limits='Max gate chosen to first establish multi-profile null control; not rank-gate sensitivity optimization. No real forest/rain or arbitrary PSD guarantee.')
    (OUTPUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    with ProcessPoolExecutor(max_workers=4) as pool:
        calibration=[]
        with (OUTPUT/'calibration.jsonl').open('w') as stream:
            for row in pool.map(evaluate,cal):
                calibration.append(row);stream.write(json.dumps(row,allow_nan=False)+'\n')
                if len(calibration)%102==0:stream.flush();print(f'Calibration {len(calibration)}/1224',flush=True)
        gates=calibrate(calibration);gatepath=OUTPUT/'frozen_gate.json';gatepath.write_text(json.dumps(gates,indent=2));gatehash=digest(gatepath)
        print(f'Frozen weighted gate: {gates["combined"]["score"]}',flush=True)
        rows=[]
        with (OUTPUT/'evaluation.jsonl').open('w') as stream:
            for raw in pool.map(evaluate,null+echo):
                row=apply_gates(raw,gates['combined'],old);rows.append(row);stream.write(json.dumps(row,allow_nan=False)+'\n')
                if len(rows)%96==0:stream.flush();print(f'Evaluation {len(rows)}/528',flush=True)
    (OUTPUT/'report.json').write_text(json.dumps(summarize(rows),indent=2))
    assert all(digest(p)==h for p,h in sources.items()) and digest(OLD)==oldhash and digest(gatepath)==gatehash
    (OUTPUT/'completion.json').write_text(json.dumps(dict(calibration=1224,evaluation=528,gate_sha256=gatehash,
        calibration_sha256=digest(OUTPUT/'calibration.jsonl'),evaluation_sha256=digest(OUTPUT/'evaluation.jsonl'),report_sha256=digest(OUTPUT/'report.json')),indent=2))

if __name__=='__main__':run()
