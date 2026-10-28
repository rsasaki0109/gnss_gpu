"""Fresh multi-profile rank calibration for the frozen residual-weighted fit."""
from dataclasses import asdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import json
from gnss_gpu.ranked_path_gate_cpu import calibrate_ranked_gate,permissible_rank
from gnss_gpu.calibrated_path_selection_cpu import calibrate_path_gate,PathSelectionGate
from experiments.run_cpu_residual_dynamic import evaluate,digest,PROFILES
from experiments.run_cpu_dynamic_ranked_gate import apply_gates
from experiments.run_cpu_dynamic_selection import summarize as summarize_single
from experiments.validate_cpu_bandlimited_multipath import paired_summary

OUTPUT=Path('output/cpu-residual-ranked')
N_CAL=640


def calibrate(rows):
    strata={}
    for p in PROFILES:
        scores=[r['score']['bic_improvement'] for r in rows if r['profile']==p]
        strata[p]=dict(maximum=asdict(calibrate_path_gate(scores,confidence_failure=.05/3)),
            ranked=asdict(calibrate_ranked_gate(scores,confidence_failure=.05/3)))
    gates={method:asdict(PathSelectionGate(max(s[method]['score'] for s in strata.values()),sum(s[method]['calibration_count'] for s in strata.values()),.01,.05)) for method in ('maximum','ranked')}
    return dict(strata=strata,gates=gates)


def summarize(rows,gates):
    cells=[];contrasts=[]
    for profile in PROFILES:
        group=[r for r in rows if r['profile']==profile]
        for method in gates:
            cells.extend(dict(c,profile=profile,method=method) for c in summarize_single([dict(r,**r['outcomes'][method]) for r in group])['cells'])
        echo=[r for r in group if r['echo_voltage']]
        paired=paired_summary([r['outcomes']['maximum']['success'] for r in echo],[r['outcomes']['ranked']['success'] for r in echo])
        primary=profile in ('white','edge')
        contrasts.append(dict(profile=profile,**paired,primary=primary,adjusted_primary_p=min(1.,2*paired['paired_exact_p']) if primary else None))
    return dict(cells=cells,contrasts=contrasts)


def run():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    if any(OUTPUT.iterdir()):raise ValueError('output must be empty')
    sources=dict(json.loads(Path('output/cpu-residual-ablation/protocol.json').read_text())['source_sha256'])
    for p in (Path(__file__).resolve(),Path('python/gnss_gpu/ranked_path_gate_cpu.py')):sources[str(p)]=digest(p)
    assert all(digest(p)==h for p,h in sources.items())
    cal=[(p,0.,0.,0.,680000+j*1000+r) for j,p in enumerate(PROFILES) for r in range(N_CAL)]
    null=[(p,0.,0.,0.,690000+r) for p in PROFILES for r in range(128)]
    echo=[(p,.8,0.,0.,700000+r) for p in PROFILES for r in range(64)]
    protocol=dict(calibration_tasks=cal,evaluation_tasks=null+echo,source_sha256=sources,
        rank_from_largest=permissible_rank(N_CAL,confidence_failure=.05/3),n_per_profile=N_CAL,
        calibration='All fresh 640 null scores/profile; rank from n/alpha/delta only. Common threshold=max of three profile thresholds. Same 1920 observations for maximum comparator.',
        guarantee='Each profile null tail <=1% except calibration failure <=BinomialCDF(k-1;640,.01); sum over3 <=.05. Maximum stricter, covered by same event. Fixed distributions only.',
        primary='Static same-phase .1-chip .8-voltage echoes in white and edge: ranked vs maximum success abs(error)<10m; two paired exact two-sided tests Bonferroni2. Center descriptive.',
        context='Frozen residual noise estimation and dynamic fit, PRN1,37 reference dB-Hz,200ms,3.6828MHz; no score/estimator/guard changes.',
        independence='Calibration IDs disjoint across profiles and evaluation; null128 and echo64 shared across profiles but null/echo disjoint. No pooling dependent profiles.',
        limits='One calibration draw; no arbitrary PSD/real data guarantee; choice of second rank is fixed by sample size, never by scores or echo outcomes.')
    (OUTPUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    with ProcessPoolExecutor(max_workers=4) as pool:
        calibration=[]
        with (OUTPUT/'calibration.jsonl').open('w') as stream:
            for row in pool.map(evaluate,cal):
                calibration.append(row);stream.write(json.dumps(row,allow_nan=False)+'\n')
                if len(calibration)%160==0:stream.flush();print(f'Calibration {len(calibration)}/1920',flush=True)
        result=calibrate(calibration);gates=result['gates'];gatepath=OUTPUT/'frozen_gates.json';gatepath.write_text(json.dumps(result,indent=2));gatehash=digest(gatepath)
        print(f'Frozen gates: {gates}',flush=True)
        rows=[]
        with (OUTPUT/'evaluation.jsonl').open('w') as stream:
            for raw in pool.map(evaluate,null+echo):
                row=apply_gates(raw,gates);rows.append(row);stream.write(json.dumps(row,allow_nan=False)+'\n')
                if len(rows)%96==0:stream.flush();print(f'Evaluation {len(rows)}/576',flush=True)
    (OUTPUT/'report.json').write_text(json.dumps(summarize(rows,gates),indent=2))
    assert all(digest(p)==h for p,h in sources.items()) and digest(gatepath)==gatehash
    (OUTPUT/'completion.json').write_text(json.dumps(dict(calibration=1920,evaluation=576,gate_sha256=gatehash,
        calibration_sha256=digest(OUTPUT/'calibration.jsonl'),evaluation_sha256=digest(OUTPUT/'evaluation.jsonl'),report_sha256=digest(OUTPUT/'report.json')),indent=2))

if __name__=='__main__':run()
