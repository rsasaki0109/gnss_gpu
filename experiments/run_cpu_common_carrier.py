"""Development replay with oracle model order; NOT calibrated path detection."""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import numpy as np
from gnss_gpu.common_carrier_paths_cpu import score_common_carrier_paths
from gnss_gpu.multipath_information_cpu import LIGHT_SPEED, CHIP_RATE
from experiments.stress_cpu_residual_carrier import synthesize, digest

OUTPUT = Path('output/cpu-common-carrier-pilot')
PRIOR = Path('output/cpu-residual-carrier-stress')


def evaluate(task):
    blocks,_,delay = synthesize(task)
    score,weights = score_common_carrier_paths(blocks)
    ratio,offset,replica = task
    # Both fits computed without truth; oracle order used ONLY to report diagnostics.
    fits = {name:asdict(score[name]) for name in ('single','two','preliminary')}
    oracle = score['two' if ratio else 'single']
    error = (oracle.first_delay_chips-delay)*LIGHT_SPEED/CHIP_RATE if oracle.locally_usable else None
    return dict(echo_voltage=ratio,residual_carrier_hz=offset,replica=replica,true_delay_chips=delay,
        fits=fits,uncalibrated_score=score['score'],oracle_error_m=error,
        oracle_success=error is not None and abs(error)<10,
        oracle_carrier_error_hz=oracle.common_carrier_hz-offset,
        observation_sha256=hashlib.sha256(blocks.tobytes()).hexdigest(),
        weights_sha256=hashlib.sha256(weights.tobytes()).hexdigest())


def summarize(rows,old):
    cells=[]
    for ratio in (0.,.8):
        for offset in (0.,.1,.5,1.):
            group=[(r,o) for r,o in zip(rows,old) if r['echo_voltage']==ratio and r['residual_carrier_hz']==offset]
            old_errors=[]
            for _,o in group:
                if ratio:
                    error=o['forced_error_m']
                else:
                    s=o['score'];d=s['single_delay_chips']
                    usable=s['single_converged'] and min(d+1,2.2-d)>=1e-4
                    error=(d-o['true_delay_chips'])*LIGHT_SPEED/CHIP_RATE if usable else None
                old_errors.append(error)
            old_success=[e is not None and abs(e)<10 for e in old_errors]
            new_success=[r['oracle_success'] for r,_ in group]
            cells.append(dict(echo_voltage=ratio,residual_carrier_hz=offset,attempts=len(group),
                old_oracle_successes=sum(old_success),new_oracle_successes=sum(new_success),
                old_oracle_unavailable=sum(e is None for e in old_errors),
                new_oracle_unavailable=sum(r['oracle_error_m'] is None for r,_ in group),
                old_selected_successes=sum(o['success'] for _,o in group),
                gains=sum(b and not a for a,b in zip(old_success,new_success)),
                losses=sum(a and not b for a,b in zip(old_success,new_success)),
                carrier_mae_hz=float(np.mean([abs(r['oracle_carrier_error_hz']) for r,_ in group]))))
    return dict(cells=cells,scope='Oracle-order development replay, not detection or heldout generalization.')


def run():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    if any(OUTPUT.iterdir()):raise ValueError('output must be empty')
    prior_protocol=json.loads((PRIOR/'protocol.json').read_text())
    sources=dict(prior_protocol['source_sha256'])
    for p in (Path(__file__).resolve(),Path('python/gnss_gpu/common_carrier_paths_cpu.py')):
        sources[str(p)]=digest(p)
    assert all(digest(p)==h for p,h in sources.items())
    old=[json.loads(x) for x in (PRIOR/'evaluation.jsonl').read_text().splitlines()]
    tasks=prior_protocol['evaluation_tasks']
    protocol=dict(evaluation_tasks=tasks,source_sha256=sources,
        prior_sha256={str(PRIOR/name):digest(PRIOR/name) for name in ('protocol.json','evaluation.jsonl','completion.json')},
        model='Single: delay/common carrier/complex amplitude (4 real parameters). Two: first delay/common carrier/gap/relative Doppler/two complex amplitudes (8). Common carrier [-2,2]Hz.',
        pipeline='Unit-weight single and two fits; two-fit residual PSD smoothed101bins; weighted single and two fits, same estimated weights; analytic gradients, 12 multistarts.',
        scoring='n log(SSE1/SSE2)-4 log(n) only uncalibrated ranking. Old gate MUST NOT be reused. No detector implemented.',
        evaluation='Replay ALL 192 previously inspected observations, no subset or threshold tuning. Both fits run blind; known generating order used ONLY for diagnostic range summary.',
        baseline='Old same-order fits plus local guards; old single uses convergence/bounds, new single also local std guard. Old calibrated selections reported separately, not a fair detector comparison.',
        limits='Development data, white37dBHz, fixed relative Doppler .7Hz/gap.1chip/phase0, narrowband model; paired replicas across offsets. No statistical superiority or detection guarantee.')
    (OUTPUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    rows=[]
    with ProcessPoolExecutor(max_workers=4) as pool,(OUTPUT/'evaluation.jsonl').open('w') as stream:
        for row,o in zip(pool.map(evaluate,tasks),old):
            assert row['observation_sha256']==o['observation_sha256']
            rows.append(row);stream.write(json.dumps(row,allow_nan=False)+'\n')
            if len(rows)%24==0:stream.flush();print(f'Evaluation {len(rows)}/192',flush=True)
    (OUTPUT/'report.json').write_text(json.dumps(summarize(rows,old),indent=2))
    assert all(digest(p)==h for p,h in {**sources,**protocol['prior_sha256']}.items())
    (OUTPUT/'completion.json').write_text(json.dumps(dict(evaluation=len(rows),
        evaluation_sha256=digest(OUTPUT/'evaluation.jsonl'),report_sha256=digest(OUTPUT/'report.json')),indent=2))


if __name__=='__main__':run()
