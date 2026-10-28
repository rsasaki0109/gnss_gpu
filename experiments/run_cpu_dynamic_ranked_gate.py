"""Fresh rank-tolerance calibration of the frozen dynamic path score."""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
import json
from gnss_gpu.ranked_path_gate_cpu import calibrate_ranked_gate,permissible_rank
from gnss_gpu.calibrated_path_selection_cpu import calibrate_path_gate,PathSelectionGate
from experiments.run_cpu_dynamic_selection import evaluate,select,summarize as summarize_single,digest
from experiments.validate_cpu_bandlimited_multipath import paired_summary

OUTPUT=Path('output/cpu-dynamic-ranked-gate')
PRIOR=Path('output/cpu-dynamic-selection')


def calibrate(rows):
    scores=[r['score']['bic_improvement'] for r in rows]
    rank=calibrate_ranked_gate(scores,confidence_failure=.05)
    return dict(maximum=asdict(calibrate_path_gate(scores)),ranked=asdict(PathSelectionGate(rank.score,rank.calibration_count,rank.tail_budget,rank.confidence_failure))),asdict(rank)


def apply_gates(row,gates):
    outcomes={}
    for method,gate in gates.items():
        chosen=select(row,PathSelectionGate(**gate))
        outcomes[method]={k:chosen[k] for k in ('selected','error_m','success','forced_error_m','forced_success','score_exceeds')}
    return dict(row,outcomes=outcomes)


def summarize(rows,gates):
    cells=[];contrasts=[]
    for method in gates:
        report=summarize_single([dict(r,**r['outcomes'][method]) for r in rows])
        cells.extend(dict(c,method=method) for c in report['cells'])
    for fd in (0.,.1,.7):
        group=sorted([r for r in rows if r['echo_voltage'] and r['doppler_hz']==fd],key=lambda r:r['replica'])
        p=paired_summary([r['outcomes']['maximum']['success'] for r in group],[r['outcomes']['ranked']['success'] for r in group])
        contrasts.append(dict(doppler_hz=fd,**p,primary=fd in (0.,.1),adjusted_primary_p=min(1.,2*p['paired_exact_p']) if fd in (0.,.1) else None))
    return dict(cells=cells,contrasts=contrasts)


def run():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    if any(OUTPUT.iterdir()):raise ValueError('output must be empty')
    sources=dict(json.loads((PRIOR/'protocol.json').read_text())['source_sha256'])
    for p in (Path(__file__).resolve(),Path('python/gnss_gpu/ranked_path_gate_cpu.py')):sources[str(p)]=digest(p)
    assert all(digest(p)==h for p,h in sources.items())
    old=PRIOR/'frozen_gate.json';oldhash=digest(old)
    cal=[(0.,0.,0.,570000+r) for r in range(1000)]
    null=[(0.,0.,0.,580000+r) for r in range(299)]
    echo=[(.8,fd,0.,590000+r) for fd in (0.,.1,.7) for r in range(128)]
    protocol=dict(calibration_tasks=cal,evaluation_tasks=null+echo,source_sha256=sources,prior_gate=str(old),prior_gate_sha256=oldhash,
        rank_from_largest=permissible_rank(1000,confidence_failure=.05),tail_budget=.01,confidence_failure=.05,
        calibration='1000 new null scores; kth-largest rank chosen only from n/alpha/delta. Compare maximum of SAME 1000 and preserved prior299 gate.',
        guarantee='P(true null score exceedance>.01)<=BinomialCDF(k-1;1000,.01) under fixed iid null model. Maximum comparator stricter so covered by same event.',
        primary='Ranked vs same-calibration maximum, success abs(error)<10m, two paired exact two-sided tests for 0/.1Hz same-phase echo, Bonferroni2. .7Hz descriptive.',
        context='Frozen white-noise dynamic model: PRN1,37dB-Hz,200ms,3.6828MHz,default delay assistance,ratio .8 gap .1 for echoes.',
        independence='Calibration/null/echo IDs disjoint; echo replicas shared across Dopplers. Original299 gate already inspected, developmental comparator only.',
        limitations='One calibration draw; fixed null distribution only; not general colored-noise/forest/rain validation. Prior299 comparator excluded from confirmatory family.')
    (OUTPUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    with ProcessPoolExecutor(max_workers=4) as pool:
        calibration=[]
        with (OUTPUT/'calibration.jsonl').open('w') as stream:
            for row in pool.map(evaluate,cal):
                calibration.append(row);stream.write(json.dumps(row,allow_nan=False)+'\n')
                if len(calibration)%100==0:stream.flush();print(f'Calibration {len(calibration)}/1000',flush=True)
        gates,rank=calibrate(calibration);gates['prior299']=json.loads(old.read_text())
        gatepath=OUTPUT/'frozen_gates.json';gatepath.write_text(json.dumps(dict(gates=gates,rank=rank),indent=2));gatehash=digest(gatepath)
        print(f'Frozen gates: {gates}',flush=True)
        rows=[]
        with (OUTPUT/'evaluation.jsonl').open('w') as stream:
            for raw in pool.map(evaluate,null+echo):
                row=apply_gates(raw,gates);rows.append(row);stream.write(json.dumps(row,allow_nan=False)+'\n')
                if len(rows)%100==0:stream.flush();print(f'Evaluation {len(rows)}/683',flush=True)
    (OUTPUT/'report.json').write_text(json.dumps(summarize(rows,gates),indent=2))
    assert all(digest(p)==h for p,h in sources.items()) and digest(old)==oldhash and digest(gatepath)==gatehash
    (OUTPUT/'completion.json').write_text(json.dumps(dict(calibration=1000,evaluation=683,gate_sha256=gatehash,
        calibration_sha256=digest(OUTPUT/'calibration.jsonl'),evaluation_sha256=digest(OUTPUT/'evaluation.jsonl'),report_sha256=digest(OUTPUT/'report.json')),indent=2))

if __name__=='__main__':run()
