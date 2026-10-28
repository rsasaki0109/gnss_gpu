"""Exploratory block fit; known failures plus a small fresh all-case panel."""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
import hashlib,json
from gnss_gpu.dynamic_multipath_cpu import fit_dynamic_paths
from gnss_gpu.multipath_information_cpu import LIGHT_SPEED,CHIP_RATE
from experiments.run_cpu_matched_dynamic_reflection import synthesize,evaluate as old_evaluate,apply_gate,CONDITIONS,digest,GATE

PRIOR=Path('output/cpu-matched-dynamic-reflection')
OUTPUT=Path('output/cpu-dynamic-path-fit')


def evaluate(task):
    panel,fd,phase,replica=task
    blocks,_,delay,_=synthesize(fd,phase,replica)
    fit=fit_dynamic_paths(blocks)
    old=apply_gate(old_evaluate((fd,phase,replica)),json.loads(GATE.read_text())['combined']['ranked'])['arms']['dynamic']
    error=(fit.first_delay_chips-delay)*LIGHT_SPEED/CHIP_RATE
    return dict(panel=panel,doppler_hz=fd,phase_rad=phase,replica=replica,
        observation_sha256=hashlib.sha256(blocks.tobytes()).hexdigest(),fit=asdict(fit),error_m=error,
        locally_usable_success=fit.locally_usable and abs(error)<10,
        raw_success=abs(error)<10,old=old)


def summarize(rows):
    cells=[]
    for panel in ('prior_failures','fresh'):
        for fd,phase in CONDITIONS:
            group=[r for r in rows if (r['panel'],r['doppler_hz'],r['phase_rad'])==(panel,fd,phase)]
            if not group:continue
            cells.append(dict(panel=panel,doppler_hz=fd,phase_rad=phase,attempts=len(group),
                old_successes=sum(r['old']['success'] for r in group),
                new_local_successes=sum(r['locally_usable_success'] for r in group),raw_successes=sum(r['raw_success'] for r in group),
                locally_unusable=sum(not r['fit']['locally_usable'] for r in group),
                nonconverged=sum(not r['fit']['converged'] for r in group),
                gains=sum(r['locally_usable_success'] and not r['old']['success'] for r in group),
                losses=sum(r['old']['success'] and not r['locally_usable_success'] for r in group)))
    return dict(cells=cells)


def run():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    if any(OUTPUT.iterdir()):raise ValueError('output must be empty')
    prior=json.loads((PRIOR/'protocol.json').read_text());sources=dict(prior['source_sha256'])
    for p in (Path(__file__).resolve(),Path('python/gnss_gpu/dynamic_multipath_cpu.py')):sources[str(p)]=digest(p)
    hashes={str(PRIOR/name):digest(PRIOR/name) for name in ('protocol.json','evaluation.jsonl')}
    hashes[str(GATE)]=digest(GATE)
    assert all(digest(p)==h for p,h in sources.items())
    previous=[json.loads(line) for line in (PRIOR/'evaluation.jsonl').read_text().splitlines()]
    tasks=[('prior_failures',r['doppler_hz'],r['phase_rad'],r['replica']) for r in previous if r['arms']['dynamic']['error_m'] is None]
    tasks += [('fresh',fd,phase,530000+r) for fd,phase in CONDITIONS for r in range(8)]
    protocol=dict(tasks=tasks,source_sha256=sources,prior_sha256=hashes,
        context='Same narrowband 10x20ms synthetic model,37dB-Hz white noise,gap .1 chip,echo voltage .8; unknown phase/delay/Doppler.',
        fit='Unweighted full block two-path fit, unknown complex amplitudes and echo Doppler [-12,12]; direct carrier perfectly wiped. Separable Gram objective preserves all samples.',
        selection='Forced two-path diagnostic with convergence/boundary/local-standard-deviation guard; NO path-count calibration. Existing gate used only for old receiver.',
        panels='All 29 previously unavailable records (retrospective); fresh nine conditions x8 shared replicas =72 observations.',
        limits='Exploratory pilot, no confirmatory significance testing. Fresh panel tiny; no null false-path test; cannot claim receiver superiority or general forest/rain robustness.')
    (OUTPUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    rows=[]
    with ProcessPoolExecutor(max_workers=4) as pool,(OUTPUT/'evaluation.jsonl').open('w') as stream:
        for row in pool.map(evaluate,tasks):
            rows.append(row);stream.write(json.dumps(row,allow_nan=False)+'\n')
            if len(rows)%10==0:stream.flush();print(f'Evaluation {len(rows)}/101',flush=True)
    (OUTPUT/'report.json').write_text(json.dumps(summarize(rows),indent=2))
    assert all(digest(p)==h for p,h in {**sources,**hashes}.items())
    (OUTPUT/'completion.json').write_text(json.dumps(dict(records=len(rows),evaluation_sha256=digest(OUTPUT/'evaluation.jsonl'),report_sha256=digest(OUTPUT/'report.json')),indent=2))

if __name__=='__main__':run()
