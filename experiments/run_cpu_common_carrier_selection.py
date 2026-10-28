"""Independent calibration/evaluation for a signed residual-carrier mixture."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from dataclasses import asdict
import json
import numpy as np
from scipy.stats import binomtest
from gnss_gpu.common_carrier_paths_cpu import CarrierFit
from gnss_gpu.calibrated_common_carrier_cpu import select_common_carrier_paths
from gnss_gpu.calibrated_path_selection_cpu import PathSelectionGate, calibrate_path_gate
from gnss_gpu.multipath_information_cpu import LIGHT_SPEED, CHIP_RATE
from experiments.run_cpu_common_carrier import evaluate, digest

OUTPUT=Path('output/cpu-common-carrier-selection')
PRIOR=Path('output/cpu-common-carrier-pilot')


def tasks(ratio,start,count):
    # Independent nuisance stream; observation generator has a different seed key.
    return [(ratio,float(np.random.default_rng(np.random.SeedSequence([20261027,0x43415252,start+r])).uniform(-1,1)),start+r)
            for r in range(count)]


def apply_gate(row,gate):
    score=dict(score=row['uncalibrated_score'],**{k:CarrierFit(**row['fits'][k]) for k in ('single','two')})
    selected=select_common_carrier_paths(score,gate)
    error=None if selected['delay_chips'] is None else (selected['delay_chips']-row['true_delay_chips'])*LIGHT_SPEED/CHIP_RATE
    return dict(row,selected=selected,error_m=error,success=error is not None and abs(error)<10,
                score_exceeds=score['score']>gate.score)


def summarize(rows):
    cells=[]
    for ratio in (0.,.8):
        group=[r for r in rows if r['echo_voltage']==ratio]
        n=len(group);two=sum(r['selected']['model']=='two_path' for r in group)
        success=sum(r['success'] for r in group)
        cells.append(dict(echo_voltage=ratio,attempts=n,two_paths=two,
            two_path_exact_95=list(binomtest(two,n).proportion_ci(method='exact')),
            successes=success,success_exact_95=list(binomtest(success,n).proportion_ci(method='exact')),
            unavailable=sum(r['error_m'] is None for r in group),score_exceedances=sum(r['score_exceeds'] for r in group),
            oracle_successes=sum(r['oracle_success'] for r in group),
            oracle_unavailable=sum(r['oracle_error_m'] is None for r in group),
            gains_vs_oracle=sum(r['success'] and not r['oracle_success'] for r in group),
            losses_vs_oracle=sum(r['oracle_success'] and not r['success'] for r in group)))
    return dict(cells=cells,scope='Marginal uniform[-1,1] Hz residual-carrier mixture, not uniform-in-carrier control.')


def run():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    if any(OUTPUT.iterdir()):raise ValueError('output must be empty')
    sources=dict(json.loads((PRIOR/'protocol.json').read_text())['source_sha256'])
    for p in (Path(__file__).resolve(),Path('python/gnss_gpu/calibrated_common_carrier_cpu.py')):sources[str(p)]=digest(p)
    assert all(digest(p)==h for p,h in sources.items())
    cal=tasks(0.,800000,299)
    evaluation=tasks(0.,810000,128)+tasks(.8,820000,64)
    protocol=dict(calibration_tasks=cal,evaluation_tasks=evaluation,source_sha256=sources,
        context='White noise,37 reference dB-Hz,PRN1,10x20ms,3.6828MHz. Common residual carrier independently uniform[-1,1]Hz for EACH observation. Random delay/phase/noise as frozen generator.',
        calibration='299 iid null mixture draws, strict maximum-order-statistic gate. P(calibration produces marginal score tail >.01) <= .99**299 < .05. Final false two-path is subset of score exceedance.',
        scope='Marginal over uniform[-1,1]carrier, random delay and phase. NOT per-carrier or worst-case control, not colored noise or different C/N0.',
        evaluation='Fresh128 null +64 echo independent IDs. Echo voltage.8,gap.1chip,relative Doppler+.7Hz,initial phase0. Gate frozen before any heldout evaluation.',
        selection='Score <=gate: guarded single path; >gate: guarded two paths; unusable requested fit: ambiguous, no fallback. No generating order or carrier supplied to selection.',
        reporting='Descriptive pointwise exact95 intervals. Oracle-order diagnostics separate. No detector superiority claim; finite heldout count does not certify1%.',
        limits='Synthetic narrowband fixed delay, no interharmonic leakage/loop/nav/PVT/forest/rain; no real RF evidence.')
    (OUTPUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    with ProcessPoolExecutor(max_workers=4) as pool:
        calibration=[]
        with (OUTPUT/'calibration.jsonl').open('w') as stream:
            for row in pool.map(evaluate,cal):
                calibration.append(row);stream.write(json.dumps(row,allow_nan=False)+'\n')
                if len(calibration)%50==0:stream.flush();print(f'Calibration {len(calibration)}/299',flush=True)
        gate=calibrate_path_gate([r['uncalibrated_score'] for r in calibration])
        gatepath=OUTPUT/'frozen_gate.json';gatepath.write_text(json.dumps(asdict(gate),indent=2));gatehash=digest(gatepath)
        print(f'Frozen gate: {gate.score}',flush=True)
        rows=[]
        with (OUTPUT/'evaluation.jsonl').open('w') as stream:
            for raw in pool.map(evaluate,evaluation):
                row=apply_gate(raw,gate);rows.append(row);stream.write(json.dumps(row,allow_nan=False)+'\n')
                if len(rows)%32==0:stream.flush();print(f'Evaluation {len(rows)}/192',flush=True)
    (OUTPUT/'report.json').write_text(json.dumps(summarize(rows),indent=2))
    assert all(digest(p)==h for p,h in sources.items()) and digest(gatepath)==gatehash
    (OUTPUT/'completion.json').write_text(json.dumps(dict(calibration=299,evaluation=192,gate_sha256=gatehash,
        calibration_sha256=digest(OUTPUT/'calibration.jsonl'),evaluation_sha256=digest(OUTPUT/'evaluation.jsonl'),report_sha256=digest(OUTPUT/'report.json')),indent=2))


if __name__=='__main__':run()
