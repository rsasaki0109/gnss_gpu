"""Independent calibration of model-independent delay ambiguity."""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
import json
import numpy as np
from scipy.stats import binomtest
from gnss_gpu.cached_drift_carrier_paths_cpu import DriftFit,estimate_weights
from gnss_gpu.drift_delay_profile_cpu import delay_profile
from gnss_gpu.calibrated_path_selection_cpu import calibrate_path_gate,PathSelectionGate
from gnss_gpu.multipath_information_cpu import LIGHT_SPEED,CHIP_RATE
from experiments.stress_cpu_drift_gap_amplitude import evaluate as base_evaluate,synthesize,tasks,apply_gate,digest,GATE

OUTPUT=Path('output/cpu-drift-ambiguity')
PRIOR=Path('output/cpu-drift-profile-alarm')
from gnss_gpu.drift_ambiguity_cpu import diagnose_ambiguity


def tasklist(kind,ratio,start,count):
    return [(k,ratio,.03,np.pi if kind=='echo' else 0.,a,f,i) for k,a,f,i in tasks(kind,start,count)]


def evaluate(task):
    row=apply_gate(base_evaluate(task),PathSelectionGate(**json.loads(GATE.read_text())))
    row['profile_alarm']=dict(score=0.,evaluated=False,unresolved=False,points=[])
    if row['selected']['model']=='ambiguous':return row
    b,_,_=synthesize(task)
    single,two,pre=[DriftFit(**row['fits'][k]) for k in ('single','two','preliminary')]
    w=estimate_weights(b,pre)
    row['profile_alarm']=diagnose_ambiguity(b,w,row['selected']['delay_chips'],single,two)
    return row


def apply_alarm(row,gate):
    result=dict(row)
    result['warning']=bool(row['profile_alarm']['score']>gate.score)
    return result


def summarize(rows):
    cells=[]
    for kind,ratio in [('null',0.),('echo',.2),('echo',.8)]:
        g=[r for r in rows if r['kind']==kind and r['echo_voltage']==ratio]
        n=len(g);warnings=sum(r['warning'] for r in g)
        dangerous=[r for r in g if r['error_m'] is not None and abs(r['error_m'])>=10]
        cells.append(dict(kind=kind,echo_voltage=ratio,attempts=n,warnings=warnings,
            warning_exact_95=list(binomtest(warnings,n).proportion_ci(method='exact')),
            baseline_unavailable=sum(r['error_m'] is None for r in g),baseline_success=sum(r['success'] for r in g),
            available_large_errors=len(dangerous),caught_large_errors=sum(r['warning'] for r in dangerous),
            missed_large_errors=sum(not r['warning'] for r in dangerous),
            warnings_on_success=sum(r['warning'] and r['success'] for r in g),
            profile_unresolved=sum(r['profile_alarm']['unresolved'] for r in g),
            retained_success=sum(r['success'] and not r['warning'] for r in g)))
    return dict(cells=cells,scope='Warning on both single-path and two-path selections. Does not fix range estimates. Unavailable cases are not profiled. Marginal null calibration only; no guarantee on echo missed-danger rate.')


def run():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    if any(OUTPUT.iterdir()):raise ValueError('output must be empty')
    sources=dict(json.loads((PRIOR/'protocol.json').read_text())['source_sha256'])
    sources[str(Path(__file__).resolve())]=digest(__file__)
    module=Path('python/gnss_gpu/drift_ambiguity_cpu.py').resolve();sources[str(module)]=digest(module)
    assert all(digest(p)==h for p,h in sources.items())
    cal=tasklist('null',0.,1170000,299)
    test=tasklist('null',0.,1180000,64)+tasklist('echo',.2,1190000,32)+tasklist('echo',.8,1200000,64)
    p=dict(source_sha256=sources,baseline_gate_sha256=digest(GATE),baseline_gate_path=str(GATE),calibration_tasks=cal,evaluation_tasks=test,
        diagnostic='Both selected models. Distant two-path candidates offsets +/-.035,.05,.1,.15chip around selected delay plus observed fitted delays;clip[-1,1],discard <=10m separation. Frozen weights and multistart. Reference=min(single_SSE,two_SSE,converged_distant_SSEs). Score=1/(1+max(0,2*Ncomplex*log(min_distant_SSE/reference)));0 if unavailable or unresolved. Not a probability. If gate reaches1,strict exceedance cannot warn;report this without retuning.',
        calibration='299 fresh iid nulls; maximum score,strict exceedance,1% marginal false-warning tail with95% calibration confidence. Gate saved before heldout evaluation. No echo tuning.',
        conditions='800ms,white37dBHz,PRN1,3.6828MHz one-sided. Common carrier U[-1,1]Hz,drift U[-2,2]Hz/s. Echo gap.03chip,phasepi,relative Doppler0. Disjoint IDs in all groups.',
        endpoints='All-attempt warning rate;baseline availability;available errors>=10m caught/missed;warnings on successful estimates;remaining successes. This is a diagnostic overlay, not a new range estimator.')
    (OUTPUT/'protocol.json').write_text(json.dumps(p,indent=2))
    with ProcessPoolExecutor(max_workers=4) as pool:
        scores=[]
        with (OUTPUT/'calibration.jsonl').open('w') as stream:
            for i,row in enumerate(pool.map(evaluate,cal),1):
                scores.append(row['profile_alarm']['score']);stream.write(json.dumps(row,allow_nan=False)+'\n')
                if i%25==0:stream.flush();print(f'Calibration {i}/299',flush=True)
        gate=calibrate_path_gate(scores)
        (OUTPUT/'frozen_gate.json').write_text(json.dumps(asdict(gate),indent=2))
        print(f'Frozen warning gate {gate.score}',flush=True)
        rows=[]
        with (OUTPUT/'evaluation.jsonl').open('w') as stream:
            for i,row in enumerate(pool.map(evaluate,test),1):
                row=apply_alarm(row,gate);rows.append(row);stream.write(json.dumps(row,allow_nan=False)+'\n')
                if i%16==0:stream.flush();print(f'Evaluation {i}/160',flush=True)
    (OUTPUT/'report.json').write_text(json.dumps(summarize(rows),indent=2))
    assert all(digest(path)==h for path,h in sources.items()) and digest(GATE)==p['baseline_gate_sha256']
    files=['protocol.json','calibration.jsonl','frozen_gate.json','evaluation.jsonl','report.json']
    (OUTPUT/'completion.json').write_text(json.dumps({name:digest(OUTPUT/name) for name in files},indent=2))


if __name__=='__main__':run()
