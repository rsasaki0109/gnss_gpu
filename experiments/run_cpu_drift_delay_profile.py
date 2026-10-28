"""Retrospective diagnostic panel, not a population or coverage study."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import hashlib,json
import numpy as np
from gnss_gpu.cached_drift_carrier_paths_cpu import DriftFit, estimate_weights
from gnss_gpu.drift_delay_profile_cpu import delay_profile
from gnss_gpu.multipath_information_cpu import CHIP_RATE,LIGHT_SPEED
from experiments.stress_cpu_drift_gap_amplitude import synthesize,digest

PRIOR=Path('output/cpu-drift-uncertainty')
OUTPUT=Path('output/cpu-drift-delay-profile')


def panel():
    rows=[json.loads(x) for x in (PRIOR/'evaluation.jsonl').read_text().splitlines()]
    return [r for ratio in (0.,.2,.8) for r in sorted(
        [r for r in rows if r['echo_voltage']==ratio and r['selected']['model']=='single_path'],
        key=lambda r:r['replica'])[:2]]


def evaluate(row):
    task=[row[k] for k in ('kind','echo_voltage','excess_chips','phase_rad','drift_hz_s','initial_carrier_hz','replica')]
    blocks,_,_=synthesize(task)
    assert hashlib.sha256(blocks.tobytes()).hexdigest()==row['observation_sha256']
    weights=estimate_weights(blocks,DriftFit(**row['fits']['preliminary']))
    assert hashlib.sha256(weights.tobytes()).hexdigest()==row['weights_sha256']
    single,two=[DriftFit(**row['fits'][k]) for k in ('single','two')]
    grid=np.unique(np.clip(np.r_[single.first_delay_chips+np.linspace(-.15,.15,25),single.first_delay_chips,two.first_delay_chips],-1,1))
    result=delay_profile(blocks,weights,grid,single,two)
    result.update(replica=row['replica'],echo_voltage=row['echo_voltage'],
        selected_delay_chips=single.first_delay_chips,true_delay_chips=row['true_delay_chips'],
        local_half_width_m=row['uncertainty']['half_width_m'])
    # Truth is only added for plotting after profiling, never used in the search.
    return result


def summarize(rows):
    cells=[]
    scale=LIGHT_SPEED/CHIP_RATE
    for r in rows:
        distant=[p for p in r['profiles']['two'] if abs(p['delay_chips']-r['selected_delay_chips'])*scale>10 and p['converged']]
        cells.append(dict(replica=r['replica'],echo_voltage=r['echo_voltage'],
            selected_error_m=(r['selected_delay_chips']-r['true_delay_chips'])*scale,
            local_half_width_m=r['local_half_width_m'],
            min_two_path_relative_loss_beyond_10m=min((p['relative_loss'] for p in distant),default=None),
            nonconverged=sum(not p['converged'] for points in r['profiles'].values() for p in points),
            boundary=sum(p['boundary'] for points in r['profiles'].values() for p in points),
            max_dense_difference=max(p['compact_dense_abs_difference'] for points in r['profiles'].values() for p in points)))
    return dict(cells=cells,limits='Six retrospectively selected single-path cases. Finite multistart profiles are upper bounds on global optimum. Relative losses have no calibrated confidence interpretation; fixed residual weights,finite grid and search bounds restrict scope.')


def run():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    if any(OUTPUT.iterdir()):raise ValueError('output must be empty')
    sources=dict(json.loads((PRIOR/'protocol.json').read_text())['source_sha256'])
    for path in (__file__,'python/gnss_gpu/drift_delay_profile_cpu.py'):sources[str(Path(path).resolve())]=digest(path)
    assert all(digest(p)==h for p,h in sources.items())
    selected=panel()
    protocol=dict(source_sha256=sources,prior_evaluation_sha256=digest(PRIOR/'evaluation.jsonl'),
        cases=selected,selection='First two replica IDs selected as single path in each of null,weak,strong prior panels. Retrospective case analysis, not new observations.',
        search='Fixed estimated residual weights.25 uniform delay offsets[-.15,.15]chip around observed single fit plus both observed fitted delays,clipped[-1,1]. Four best deterministic nuisance seeds per point/model. Same bounds as frozen estimator.',
        normalization='2*Ncomplex*log(SSE/min_SSE_over_profiles_and_prior_fits),common reference for both models. No chi-square threshold, interval, detector, or coverage claim.')
    (OUTPUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    rows=[]
    with ProcessPoolExecutor(max_workers=4) as pool,(OUTPUT/'profiles.jsonl').open('w') as stream:
        for r in pool.map(evaluate,selected):
            rows.append(r);stream.write(json.dumps(r,allow_nan=False)+'\n');stream.flush()
            print(f'Profile {len(rows)}/{len(selected)}',flush=True)
    (OUTPUT/'report.json').write_text(json.dumps(summarize(rows),indent=2))
    assert all(digest(p)==h for p,h in sources.items())
    (OUTPUT/'completion.json').write_text(json.dumps({name+'_sha256':digest(OUTPUT/(name+ext)) for name,ext in [('protocol','.json'),('profiles','.jsonl'),('report','.json')]},indent=2))


if __name__=='__main__':run()
