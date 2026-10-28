"""Paired full-pipeline timings plus replay of every frozen calibration/eval row."""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
import hashlib,json,statistics
import numpy as np
from gnss_gpu import drift_carrier_paths_cpu as baseline
from gnss_gpu import cached_drift_carrier_paths_cpu as cached
from experiments.run_cpu_drift_selection import null_signal,echo_signal,digest,apply_gate,PathSelectionGate,summarize

OUTPUT=Path('output/cpu-drift-cache-benchmark')
PRIOR=Path('output/cpu-drift-selection')


def observation(row):
    b,_,_=(null_signal if row['kind']=='null' else echo_signal)((40,row['drift_hz_s'],row['initial_carrier_hz'],row['replica']))
    assert hashlib.sha256(b.tobytes()).hexdigest()==row['observation_sha256']
    return b


def verify_result(row,result,weights):
    assert result['score']==row['uncalibrated_score']
    for key in ('single','two','preliminary'):assert asdict(result[key])==row['fits'][key]
    assert hashlib.sha256(weights.tobytes()).hexdigest()==row['weights_sha256']


def replay(item):
    stage,index,row=item
    b=observation(row);start=perf_counter();result,weights=cached.score_drift_paths(b);elapsed=perf_counter()-start
    verify_result(row,result,weights)
    return dict(stage=stage,index=index,replica=row['replica'],exact=True,cached_elapsed_seconds=elapsed,
        score=result['score'],weights_sha256=hashlib.sha256(weights.tobytes()).hexdigest())


def run():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    if any(OUTPUT.iterdir()):raise ValueError('output must be empty')
    sources=dict(json.loads((PRIOR/'protocol.json').read_text())['source_sha256'])
    for path in (Path(__file__).resolve(),Path('python/gnss_gpu/cached_drift_carrier_paths_cpu.py')):sources[str(path)]=digest(path)
    prior={str(PRIOR/name):digest(PRIOR/name) for name in ('protocol.json','calibration.jsonl','evaluation.jsonl','report.json','frozen_gate.json')}
    assert all(digest(path)==h for path,h in {**sources,**prior}.items())
    cal=[json.loads(x) for x in (PRIOR/'calibration.jsonl').read_text().splitlines()]
    evaluation=[json.loads(x) for x in (PRIOR/'evaluation.jsonl').read_text().splitlines()]
    cases=[('calibration',0,cal[0]),('evaluation',0,evaluation[0]),('evaluation',64,evaluation[64]),('evaluation',91,evaluation[91])]
    protocol=dict(source_sha256=sources,prior_sha256=prior,benchmark_cases=[[s,i,r['replica']] for s,i,r in cases],
        change='Only bounded64-entry per-problem spatial cache: exact delay float -> template, delay derivative, observation projections. Same vector arithmetic, no rounding, no global observation cache. Search and selection unchanged.',
        memory='At800ms,64entries x2 complex7365 vectors +2 complex40 vectors ~15.2MB array payload per active problem, plus Python/cache/observation/fit allocations. Not measured peak RSS.',
        timing='Four preselected existing cases including remaining boundary failure,2 repeats. Warm both modules once outside timing; alternate baseline/cached execution order by repeat and case. Sequential full pipeline only, exclude synthesis/checks/imports; BLAS1 thread. Same process/machine; no timing claim from concurrent replay.',
        replay='All299 calibration+96 heldout observations. Compare full fits,score and weights bytes EXACTLY, not tolerance. Preserve original gate; recompute all96 choices/report.',
        limits='Empirical equivalence on395 observations and targeted tests, not proof for every input/platform. Timing small-case local benchmark; no universal speedup or real-time claim.')
    (OUTPUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    warm=observation(cases[0][2])
    for module in (baseline,cached):
        score,w=module.score_drift_paths(warm);verify_result(cases[0][2],score,w)
    timings=[]
    with (OUTPUT/'timings.jsonl').open('w') as stream:
        for repeat in range(2):
            for j,(stage,index,row) in enumerate(cases):
                b=observation(row);entry=dict(repeat=repeat,stage=stage,index=index,replica=row['replica'])
                order=('baseline','cached') if (repeat+j)%2==0 else ('cached','baseline')
                for method in order:
                    module=baseline if method=='baseline' else cached
                    start=perf_counter();result,w=module.score_drift_paths(b);entry[method+'_seconds']=perf_counter()-start
                    verify_result(row,result,w)
                entry['speedup']=entry['baseline_seconds']/entry['cached_seconds'];timings.append(entry)
                stream.write(json.dumps(entry)+'\n');stream.flush();print(f'Timing {len(timings)}/8: {entry["speedup"]:.2f}x',flush=True)
    items=[('calibration',i,r) for i,r in enumerate(cal)]+[('evaluation',i,r) for i,r in enumerate(evaluation)]
    rows=[]
    with ProcessPoolExecutor(max_workers=4) as pool,(OUTPUT/'replay.jsonl').open('w') as stream:
        for row in pool.map(replay,items):
            rows.append(row);stream.write(json.dumps(row)+'\n')
            if len(rows)%50==0:stream.flush();print(f'Exact replay {len(rows)}/395',flush=True)
    gate=PathSelectionGate(**json.loads((PRIOR/'frozen_gate.json').read_text()))
    for row in evaluation:
        raw={k:v for k,v in row.items() if k not in ('selected','error_m','success','score_exceeds')}
        assert apply_gate(raw,gate)==row
    assert summarize(evaluation)==json.loads((PRIOR/'report.json').read_text())
    report=dict(exact_calibration=299,exact_evaluation=96,all_original_choices_and_report_preserved=True,
        timing_pairs=len(timings),median_baseline_seconds=statistics.median(r['baseline_seconds'] for r in timings),
        median_cached_seconds=statistics.median(r['cached_seconds'] for r in timings),
        median_paired_speedup=statistics.median(r['speedup'] for r in timings),
        speedup_range=[min(r['speedup'] for r in timings),max(r['speedup'] for r in timings)])
    (OUTPUT/'report.json').write_text(json.dumps(report,indent=2))
    assert all(digest(path)==h for path,h in {**sources,**prior}.items())
    (OUTPUT/'completion.json').write_text(json.dumps(dict(replays=len(rows),timing_pairs=len(timings),
        replay_sha256=digest(OUTPUT/'replay.jsonl'),timings_sha256=digest(OUTPUT/'timings.jsonl'),report_sha256=digest(OUTPUT/'report.json')),indent=2))
    print(json.dumps(report),flush=True)


if __name__=='__main__':run()
