"""Development replay of drift-aware single/two-path models; no selector."""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
import hashlib,json
import numpy as np
from gnss_gpu.drift_carrier_paths_cpu import score_drift_paths
from gnss_gpu.multipath_information_cpu import CHIP_RATE,LIGHT_SPEED
from experiments.run_cpu_drift_null_selection import synthesize as null_signal,digest
from experiments.stress_cpu_carrier_drift import synthesize as echo_signal

OUTPUT=Path('output/cpu-drift-model-pilot')
PRIORS={'null':Path('output/cpu-drift-null-selection'),'echo':Path('output/cpu-carrier-drift-stress')}


def select_cases():
    cases=[]
    for kind,path in PRIORS.items():
        rows=[json.loads(x) for x in (path/'evaluation.jsonl').read_text().splitlines()]
        ids=sorted({r['replica'] for r in rows})[:8]
        cases.extend(dict(r,kind=kind) for r in rows if r['block_count']==40 and r['replica'] in ids)
    return cases


def evaluate(old):
    task=tuple(old[k] for k in ('block_count','drift_hz_s','initial_carrier_hz','replica'))
    blocks,_,delay=(null_signal if old['kind']=='null' else echo_signal)(task)
    assert hashlib.sha256(blocks.tobytes()).hexdigest()==old['observation_sha256']
    score,weights=score_drift_paths(blocks)
    oracle=score['single' if old['kind']=='null' else 'two']
    error=(oracle.first_delay_chips-delay)*LIGHT_SPEED/CHIP_RATE if oracle.locally_usable else None
    return dict(kind=old['kind'],replica=old['replica'],drift_hz_s=old['drift_hz_s'],initial_carrier_hz=old['initial_carrier_hz'],
        fits={k:asdict(score[k]) for k in ('single','two','preliminary')},uncalibrated_score=score['score'],
        old_score=old['uncalibrated_score'],old_oracle_success=old['oracle_success'],old_oracle_error_m=old['oracle_error_m'],
        oracle_error_m=error,oracle_success=error is not None and abs(error)<10,
        oracle_drift_error_hz_s=oracle.common_drift_hz_s-old['drift_hz_s'],
        observation_sha256=old['observation_sha256'],weights_sha256=hashlib.sha256(weights.tobytes()).hexdigest())


def summarize(rows):
    cells=[]
    for kind in ('null','echo'):
        for rate in (0.,.5,2.):
            g=[r for r in rows if r['kind']==kind and r['drift_hz_s']==rate]
            cells.append(dict(kind=kind,drift_hz_s=rate,attempts=len(g),old_oracle_successes=sum(r['old_oracle_success'] for r in g),
                new_oracle_successes=sum(r['oracle_success'] for r in g),new_oracle_unavailable=sum(r['oracle_error_m'] is None for r in g),
                gains=sum(r['oracle_success'] and not r['old_oracle_success'] for r in g),losses=sum(r['old_oracle_success'] and not r['oracle_success'] for r in g),
                old_score_range=[min(r['old_score'] for r in g),max(r['old_score'] for r in g)],
                new_score_range=[min(r['uncalibrated_score'] for r in g),max(r['uncalibrated_score'] for r in g)],
                drift_mae_hz_s=float(np.mean([abs(r['oracle_drift_error_hz_s']) for r in g]))))
    return dict(cells=cells,scope='Retrospective model-fit pilot, known order only; new scores are NOT calibrated and old gate is not applied.')


def run():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    if any(OUTPUT.iterdir()):raise ValueError('output must be empty')
    sources={};prior={}
    for path in PRIORS.values():
        sources.update(json.loads((path/'protocol.json').read_text())['source_sha256'])
        for name in ('protocol.json','evaluation.jsonl','completion.json'):prior[str(path/name)]=digest(path/name)
    for path in (Path(__file__).resolve(),Path('python/gnss_gpu/drift_carrier_paths_cpu.py')):sources[str(path)]=digest(path)
    assert all(digest(path)==h for path,h in {**sources,**prior}.items())
    cases=select_cases();assert len(cases)==48
    protocol=dict(source_sha256=sources,prior_sha256=prior,cases=[[r['kind'],r['replica'],r['drift_hz_s']] for r in cases],
        selection='800ms only, first8 replica IDs in each prior dataset, all3rates. Deterministic subset regardless of outcomes; development replay, not fresh heldout.',
        model='Common linear frequency drift unknown in BOTH single and two paths.5/9 real parameters including amplitudes; drift bounds[-3,3]Hz/s. Common initial carrier[-2,2],relative[-12,12]Hz.24-point block quadrature with analytic derivatives.',
        pipeline='Unit-weight single then two -> residual smoothed PSD -> weighted single/two. Same weights for scores. Two coarse drift seeds use0 and observation-derived single drift; default unassisted uses-2/0/2.12starts,max300iterations.',
        scoring='n log(SSE1/SSE2)-4log(n), uncalibrated ranking only. No old gate or model-order selector; known order used only for range diagnostics.',
        limits='8paired replicas per cell, retrospective; no error-rate or superiority claims. Fixed code delay, synthetic oscillator-like drift, no realRF/loop/nav/PVT.')
    (OUTPUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    rows=[]
    with ProcessPoolExecutor(max_workers=4) as pool,(OUTPUT/'evaluation.jsonl').open('w') as stream:
        for row in pool.map(evaluate,cases):
            rows.append(row);stream.write(json.dumps(row,allow_nan=False)+'\n')
            if len(rows)%8==0:stream.flush();print(f'Evaluation {len(rows)}/48',flush=True)
    (OUTPUT/'report.json').write_text(json.dumps(summarize(rows),indent=2))
    assert all(digest(path)==h for path,h in {**sources,**prior}.items())
    (OUTPUT/'completion.json').write_text(json.dumps(dict(evaluation=len(rows),evaluation_sha256=digest(OUTPUT/'evaluation.jsonl'),report_sha256=digest(OUTPUT/'report.json')),indent=2))


if __name__=='__main__':run()
