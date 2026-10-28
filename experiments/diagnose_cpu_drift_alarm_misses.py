"""Post-hoc diagnosis of every available >=10m error missed by frozen alarm."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import hashlib,json
import numpy as np
from gnss_gpu.cached_drift_carrier_paths_cpu import DriftFit,estimate_weights
from gnss_gpu.drift_delay_profile_cpu import delay_profile
from gnss_gpu.multipath_information_cpu import CHIP_RATE,LIGHT_SPEED
from experiments.stress_cpu_drift_gap_amplitude import synthesize,digest
from experiments.run_cpu_drift_profile_alarm import alarm_score

OUTPUT=Path('output/cpu-drift-alarm-misses')
PRIOR=Path('output/cpu-drift-profile-alarm')
SCALE=LIGHT_SPEED/CHIP_RATE


def missed_cases(rows):
    return [r for r in rows if r['error_m'] is not None and abs(r['error_m'])>=10 and not r['warning']]


def diagnostic_grid(row):
    center=row['selected']['delay_chips']
    return np.unique(np.clip(np.r_[center+np.linspace(-.15,.15,25),
        row['fits']['single']['first_delay_chips'],row['fits']['two']['first_delay_chips'],
        [p['delay_chips'] for p in row['profile_alarm']['points']]],-1,1))


def evaluate(row):
    task=[row[k] for k in ('kind','echo_voltage','excess_chips','phase_rad','drift_hz_s','initial_carrier_hz','replica')]
    b,_,_=synthesize(task)
    assert hashlib.sha256(b.tobytes()).hexdigest()==row['observation_sha256']
    single,two,pre=[DriftFit(**row['fits'][k]) for k in ('single','two','preliminary')]
    w=estimate_weights(b,pre)
    assert hashlib.sha256(w.tobytes()).hexdigest()==row['weights_sha256']
    profile=delay_profile(b,w,diagnostic_grid(row),single,two)
    # Truth-only fit is an explicitly separate oracle diagnostic, not a detector candidate.
    oracle=delay_profile(b,w,[row['true_delay_chips']],single,two)['profiles']['two'][0]
    return dict(replica=row['replica'],profiles=profile['profiles'],reference_loss=profile['reference_loss'],
        oracle_fixed_true_delay=oracle,complex_count=b.size)


def summarize(results,cases,gate):
    cells=[]
    for result,row in zip(results,cases):
        assert result['replica']==row['replica']
        single,two=[row['fits'][k] for k in ('single','two')]
        model=row['selected']['model'];fit=single if model=='single_path' else two
        candidate=[p for p in result['profiles']['two'] if p['converged'] and abs(p['delay_chips']-single['first_delay_chips'])*SCALE>10]
        refined=alarm_score(single['loss'],[p['loss'] for p in candidate],result['complex_count'])
        refined=max(refined,row['profile_alarm']['score']) if model=='single_path' else None
        best=min([two['loss']]+[p['loss'] for p in result['profiles']['two'] if p['converged']])
        oracle=result['oracle_fixed_true_delay']
        cells.append(dict(replica=row['replica'],model=model,error_m=row['error_m'],
            local_half_width_m=1.959963984540054*fit['first_std_chips']*SCALE,
            fitted_gap_chips=two['excess_chips'],true_gap_chips=row['excess_chips'],
            old_alarm_score=row['profile_alarm']['score'],refined_alarm_score=refined,
            crosses_old_gate_posthoc=refined>gate if refined is not None else None,
            previous_unrestricted_improvement=2*result['complex_count']*float(np.log(single['loss']/two['loss'])),
            profile_improves_two_loss_relative=(two['loss']-best)/two['loss'],
            oracle_delay_relative_loss=2*result['complex_count']*float(np.log(oracle['loss']/best)),
            oracle_converged=oracle['converged'],
            nonconverged=sum(not p['converged'] for points in result['profiles'].values() for p in points)))
    return dict(cells=cells,scope='All11 retrospectively selected missed errors,not independent validation. Refined search is uncalibrated: crossing the old gate is diagnostic only. Fixed-true-delay fit uses truth and is not a deployable candidate. Finite search cannot prove global optimality or separate all model/noise effects.')


def run():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    if any(OUTPUT.iterdir()):raise ValueError('output must be empty')
    sources=dict(json.loads((PRIOR/'protocol.json').read_text())['source_sha256'])
    sources[str(Path(__file__).resolve())]=digest(__file__)
    assert all(digest(p)==h for p,h in sources.items())
    cases=missed_cases([json.loads(x) for x in (PRIOR/'evaluation.jsonl').read_text().splitlines()])
    assert len(cases)==11
    gate=json.loads((PRIOR/'frozen_gate.json').read_text())['score']
    p=dict(source_sha256=sources,prior_evaluation_sha256=digest(PRIOR/'evaluation.jsonl'),prior_gate_sha256=digest(PRIOR/'frozen_gate.json'),
        cases=cases,old_alarm_gate=gate,
        design='All available errors>=10m without warning.25-point selected-delay-centered grid +/- .15chip plus previous fitted delays and alarm points. Frozen weights/bounds/optimizer. Separate truth-fixed two-path fit for oracle diagnosis only. No gate or receiver changes.')
    (OUTPUT/'protocol.json').write_text(json.dumps(p,indent=2))
    rows=[]
    with ProcessPoolExecutor(max_workers=4) as pool,(OUTPUT/'profiles.jsonl').open('w') as stream:
        for r in pool.map(evaluate,cases):
            rows.append(r);stream.write(json.dumps(r,allow_nan=False)+'\n');stream.flush()
            print(f'Diagnosed {len(rows)}/11',flush=True)
    (OUTPUT/'report.json').write_text(json.dumps(summarize(rows,cases,gate),indent=2))
    assert all(digest(path)==h for path,h in sources.items())
    (OUTPUT/'completion.json').write_text(json.dumps({name:digest(OUTPUT/name) for name in ('protocol.json','profiles.jsonl','report.json')},indent=2))


if __name__=='__main__':run()
