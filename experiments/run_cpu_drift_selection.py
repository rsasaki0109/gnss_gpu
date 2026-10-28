"""Independent full-pipeline calibration of the drift-aware receiver."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from dataclasses import asdict
import hashlib,json
import numpy as np
from scipy.stats import binomtest
from gnss_gpu.drift_carrier_paths_cpu import DriftFit,score_drift_paths
from gnss_gpu.calibrated_common_carrier_cpu import select_common_carrier_paths
from gnss_gpu.calibrated_path_selection_cpu import PathSelectionGate,calibrate_path_gate
from gnss_gpu.multipath_information_cpu import CHIP_RATE,LIGHT_SPEED
from experiments.run_cpu_drift_null_selection import synthesize as null_signal,digest
from experiments.stress_cpu_carrier_drift import synthesize as echo_signal

OUTPUT=Path('output/cpu-drift-selection')
PRIOR=Path('output/cpu-drift-model-pilot')


def tasks(kind,start,count):
    result=[]
    for replica in range(start,start+count):
        rng=np.random.default_rng(np.random.SeedSequence([20261027,0x44524341,replica]))
        result.append((kind,float(rng.uniform(-2,2)),float(rng.uniform(-1,1)),replica))
    return result


def evaluate(task):
    kind,drift,offset,replica=task
    blocks,_,delay=(null_signal if kind=='null' else echo_signal)((40,drift,offset,replica))
    score,weights=score_drift_paths(blocks)
    oracle=score['single' if kind=='null' else 'two']
    error=(oracle.first_delay_chips-delay)*LIGHT_SPEED/CHIP_RATE if oracle.locally_usable else None
    return dict(kind=kind,drift_hz_s=drift,initial_carrier_hz=offset,replica=replica,true_delay_chips=delay,
        fits={k:asdict(score[k]) for k in ('single','two','preliminary')},uncalibrated_score=score['score'],
        oracle_error_m=error,oracle_success=error is not None and abs(error)<10,
        oracle_drift_error_hz_s=oracle.common_drift_hz_s-drift,
        observation_sha256=hashlib.sha256(blocks.tobytes()).hexdigest(),weights_sha256=hashlib.sha256(weights.tobytes()).hexdigest())


def apply_gate(row,gate):
    score=dict(score=row['uncalibrated_score'],**{k:DriftFit(**row['fits'][k]) for k in ('single','two')})
    selected=select_common_carrier_paths(score,gate)
    error=None if selected['delay_chips'] is None else (selected['delay_chips']-row['true_delay_chips'])*LIGHT_SPEED/CHIP_RATE
    return dict(row,selected=selected,error_m=error,success=error is not None and abs(error)<10,
        score_exceeds=score['score']>gate.score)


def summarize(rows):
    cells=[]
    for kind in ('null','echo'):
        g=[r for r in rows if r['kind']==kind];n=len(g)
        two=sum(r['selected']['model']=='two_path' for r in g);success=sum(r['success'] for r in g)
        cells.append(dict(kind=kind,attempts=n,two_paths=two,two_path_exact_95=list(binomtest(two,n).proportion_ci(method='exact')),
            successes=success,success_exact_95=list(binomtest(success,n).proportion_ci(method='exact')),
            unavailable=sum(r['error_m'] is None for r in g),score_exceedances=sum(r['score_exceeds'] for r in g),
            oracle_successes=sum(r['oracle_success'] for r in g),oracle_unavailable=sum(r['oracle_error_m'] is None for r in g),
            gains_vs_oracle=sum(r['success'] and not r['oracle_success'] for r in g),losses_vs_oracle=sum(r['oracle_success'] and not r['success'] for r in g),
            oracle_drift_mae_hz_s=float(np.mean([abs(r['oracle_drift_error_hz_s']) for r in g]))))
    return dict(cells=cells,scope='Marginal signed drift/carrier mixture, not uniform-in-drift control; pointwise intervals.')


def run():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    if any(OUTPUT.iterdir()):raise ValueError('output must be empty')
    sources=dict(json.loads((PRIOR/'protocol.json').read_text())['source_sha256']);sources[str(Path(__file__).resolve())]=digest(__file__)
    assert all(digest(p)==h for p,h in sources.items())
    caltasks=tasks('null',980000,299);evaluation=tasks('null',990000,64)+tasks('echo',1000000,32)
    protocol=dict(source_sha256=sources,calibration_tasks=caltasks,evaluation_tasks=evaluation,
        context='Frozen drift-aware receiver,800ms,PRN1,white37dBHz,3.6828MHz; independent drift uniform[-2,2]Hz/s and initial carrier uniform[-1,1]Hz per observation. Echo voltage.8,gap.1chip,relative Doppler+.1Hz,initial phasepi.',
        calibration='299 independent nulls, maximum-score strict gate; failure probability for marginal null tail>.01 bounded by .99**299<.05. Full pipeline includes fitted residual PSD and guards.',
        selection='Existing guarded selector accepts DriftFit: score<=gate guarded single, >gate guarded two; unusable requested model remains ambiguous. No generating kind or truth used.',
        evaluation='Fresh64 null and32 echo, disjoint IDs from calibration and prior development. Gate frozen before heldout evaluation. No old gate reuse or stress-driven retuning.',
        scope='Conditional on full mixture distribution; NOT each drift/carrier value, arbitrary PSD/CN0 or worst case. Pointwise exact95 intervals; no detector-superiority test.',
        limits='Synthetic linear common drift and fixed code delays, no realRF/loop/nav/stochastic phase noise/forest/rain/PVT. Changed receiver plus changed calibration, not isolated model effect.')
    (OUTPUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    with ProcessPoolExecutor(max_workers=4) as pool:
        cal=[]
        with (OUTPUT/'calibration.jsonl').open('w') as stream:
            for row in pool.map(evaluate,caltasks):
                cal.append(row);stream.write(json.dumps(row,allow_nan=False)+'\n')
                if len(cal)%25==0:stream.flush();print(f'Calibration {len(cal)}/299',flush=True)
        gate=calibrate_path_gate([r['uncalibrated_score'] for r in cal]);gatepath=OUTPUT/'frozen_gate.json'
        gatepath.write_text(json.dumps(asdict(gate),indent=2));gatehash=digest(gatepath)
        print(f'Frozen gate: {gate.score}',flush=True)
        rows=[]
        with (OUTPUT/'evaluation.jsonl').open('w') as stream:
            for raw in pool.map(evaluate,evaluation):
                row=apply_gate(raw,gate);rows.append(row);stream.write(json.dumps(row,allow_nan=False)+'\n')
                if len(rows)%16==0:stream.flush();print(f'Evaluation {len(rows)}/96',flush=True)
    (OUTPUT/'report.json').write_text(json.dumps(summarize(rows),indent=2))
    assert all(digest(p)==h for p,h in sources.items()) and digest(gatepath)==gatehash
    (OUTPUT/'completion.json').write_text(json.dumps(dict(calibration=299,evaluation=96,gate_sha256=gatehash,
        calibration_sha256=digest(OUTPUT/'calibration.jsonl'),evaluation_sha256=digest(OUTPUT/'evaluation.jsonl'),report_sha256=digest(OUTPUT/'report.json')),indent=2))


if __name__=='__main__':run()
