"""Frozen common-carrier selection under paired spectral-noise mismatch."""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import numpy as np
from gnss_gpu.common_carrier_paths_cpu import score_common_carrier_paths
from gnss_gpu.spectral_noise_cpu import spectral_noise_shape
from gnss_gpu.multipath_information_cpu import ca_passband_spectrum, CHIP_RATE, LIGHT_SPEED
from gnss_gpu.calibrated_path_selection_cpu import PathSelectionGate
from experiments.stress_cpu_residual_carrier import synthesize as white_signal
from experiments.run_cpu_common_carrier_selection import tasks, apply_gate, summarize as summarize_profile, digest

OUTPUT=Path('output/cpu-common-carrier-noise-stress')
PRIOR=Path('output/cpu-common-carrier-selection')
PROFILES=('white','center','edge')


def synthesize(task):
    profile,ratio,offset,replica=task
    blocks,signal,delay=white_signal((ratio,offset,replica))
    if profile=='white':return blocks,signal,delay
    f,_=ca_passband_spectrum(3.6*CHIP_RATE)
    return signal+(blocks-signal)*np.sqrt(spectral_noise_shape(f,profile)),signal,delay


def evaluate(task):
    blocks,_,delay=synthesize(task)
    score,weights=score_common_carrier_paths(blocks)
    profile,ratio,offset,replica=task
    oracle=score['two' if ratio else 'single']
    error=(oracle.first_delay_chips-delay)*LIGHT_SPEED/CHIP_RATE if oracle.locally_usable else None
    return dict(profile=profile,echo_voltage=ratio,residual_carrier_hz=offset,replica=replica,true_delay_chips=delay,
        fits={k:asdict(score[k]) for k in ('single','two','preliminary')},uncalibrated_score=score['score'],
        oracle_error_m=error,oracle_success=error is not None and abs(error)<10,
        oracle_carrier_error_hz=oracle.common_carrier_hz-offset,
        observation_sha256=hashlib.sha256(blocks.tobytes()).hexdigest(),
        weights_sha256=hashlib.sha256(weights.tobytes()).hexdigest())


def summarize(rows):
    cells=[]
    for profile in PROFILES:
        cells.extend(dict(c,profile=profile) for c in summarize_profile([r for r in rows if r['profile']==profile])['cells'])
    pairs=[]
    for profile in ('center','edge'):
        for ratio in (0.,.8):
            white={r['replica']:r for r in rows if r['profile']=='white' and r['echo_voltage']==ratio}
            other={r['replica']:r for r in rows if r['profile']==profile and r['echo_voltage']==ratio}
            assert white.keys()==other.keys()
            pairs.append(dict(profile=profile,echo_voltage=ratio,attempts=len(white),
                gains=sum(other[k]['success'] and not white[k]['success'] for k in white),
                losses=sum(white[k]['success'] and not other[k]['success'] for k in white)))
    return dict(cells=cells,paired_range_changes_vs_white=pairs,
        scope='Exploratory paired noise mismatch; white-null calibration guarantee does not cover colored noise.')


def run():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    if any(OUTPUT.iterdir()):raise ValueError('output must be empty')
    sources=dict(json.loads((PRIOR/'protocol.json').read_text())['source_sha256'])
    for p in (Path(__file__).resolve(),Path('python/gnss_gpu/spectral_noise_cpu.py')):sources[str(p)]=digest(p)
    assert all(digest(p)==h for p,h in sources.items())
    gatepath=PRIOR/'frozen_gate.json';gatehash=digest(gatepath)
    gate=PathSelectionGate(**json.loads(gatepath.read_text()))
    evaluation=[(p,*t) for p in PROFILES for t in tasks(0.,840000,64)+tasks(.8,850000,32)]
    protocol=dict(evaluation_tasks=evaluation,source_sha256=sources,gate_path=str(gatepath),gate_sha256=gatehash,gate=asdict(gate),
        context='Frozen common-carrier receiver and white-null gate. PRN1,37 reference dB-Hz,200ms,3.6828MHz; common carrier uniform[-1,1]Hz. Echo voltage.8,gap.1chip,relative Doppler+.7Hz,phase0.',
        intervention='Only spectral noise variance changes: white1; center1+9exp(-.5*(12x)^2); edge1+9x^4. Each normalized mean1. Same expected total noise energy, not equal ranging information.',
        pairing='64 null and32 echo new IDs shared across three profiles: same signal, carrier, delay, phase and standardized noise. Null/echo IDs disjoint. Do not pool profiles as independent.',
        inference='Exploratory descriptive paired comparison; pointwise exact95 intervals, no superiority tests. No recalibration or threshold retuning.',
        limits='White calibration guarantee does NOT extend to colored noise. Synthetic independent harmonic noise, fixed code delay, not measured RF/frontend/weather, tracking or PVT.')
    (OUTPUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    rows=[]
    with ProcessPoolExecutor(max_workers=4) as pool,(OUTPUT/'evaluation.jsonl').open('w') as stream:
        for raw in pool.map(evaluate,evaluation):
            row=apply_gate(raw,gate);rows.append(row);stream.write(json.dumps(row,allow_nan=False)+'\n')
            if len(rows)%32==0:stream.flush();print(f'Evaluation {len(rows)}/288',flush=True)
    (OUTPUT/'report.json').write_text(json.dumps(summarize(rows),indent=2))
    assert all(digest(p)==h for p,h in sources.items()) and digest(gatepath)==gatehash
    (OUTPUT/'completion.json').write_text(json.dumps(dict(evaluation=len(rows),gate_sha256=gatehash,
        evaluation_sha256=digest(OUTPUT/'evaluation.jsonl'),report_sha256=digest(OUTPUT/'report.json')),indent=2))


if __name__=='__main__':run()
