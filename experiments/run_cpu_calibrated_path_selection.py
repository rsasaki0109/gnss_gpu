"""Single-path calibration, then untouched null and echo evaluation."""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import numpy as np
from scipy.stats import binomtest
from experiments.run_cpu_coherent_multipath import synthesize_blocks
from experiments.validate_cpu_bandlimited_multipath import paired_summary
from gnss_gpu.bandlimited_multipath_v2_cpu import BandlimitedFit, fit_bandlimited_paths
from gnss_gpu.calibrated_path_selection_cpu import calibrate_path_gate, _select_default_fit
from gnss_gpu.multipath_information_cpu import CHIP_RATE, LIGHT_SPEED

OUTPUT=Path('output/cpu-calibrated-path-selection')


def evaluate(task):
    ratio,gap,phase,replica=task
    _,blocks,delay=synthesize_blocks(3.6,ratio,gap,phase,replica)
    observation=blocks.mean(axis=0)
    fit=fit_bandlimited_paths(observation,3.6*CHIP_RATE)
    return dict(echo_voltage=ratio,gap_chips=gap,phase_rad=phase,replica=replica,
        true_delay_chips=delay,fit=asdict(fit),
        observation_sha256=hashlib.sha256(observation.tobytes()).hexdigest())


def apply_gate(row,gate):
    old=BandlimitedFit(**row['fit'])
    new=_select_default_fit(old,gate)
    def error(value):
        return (value-row['true_delay_chips'])*LIGHT_SPEED/CHIP_RATE if value is not None else None
    return dict(row,calibrated_fit=asdict(new),old_error_m=error(old.selected_delay_chips),
                new_error_m=error(new.selected_delay_chips),score_exceeds_gate=old.bic_improvement>gate.score)


def summarize(rows):
    groups={}
    for row in rows:
        groups.setdefault(tuple(row[k] for k in ('echo_voltage','gap_chips','phase_rad')),[]).append(row)
    report=[]
    for key,group in groups.items():
        old=[r['old_error_m'] is not None and abs(r['old_error_m'])<10 for r in group]
        new=[r['new_error_m'] is not None and abs(r['new_error_m'])<10 for r in group]
        score_pass=sum(r['score_exceeds_gate'] for r in group)
        ci=binomtest(score_pass,len(group)).proportion_ci(confidence_level=.95,method='exact')
        paired=paired_summary(old,new)
        primary=key==(.8,.1,0.)
        report.append(dict(echo_voltage=key[0],gap_chips=key[1],phase_rad=key[2],**paired,
            score_exceedances=score_pass,score_exceedance_exact_95=list(ci),
            old_two_paths=sum(r['fit']['selected_model']=='two_path' for r in group),
            new_two_paths=sum(r['calibrated_fit']['selected_model']=='two_path' for r in group),
            old_unavailable=sum(r['old_error_m'] is None for r in group),
            new_unavailable=sum(r['new_error_m'] is None for r in group),primary=primary,
            primary_improvement_supported=paired['difference']>0 and paired['paired_exact_p']<.05 if primary else None))
    return report


def run():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    if any(OUTPUT.iterdir()): raise ValueError('output must be empty')
    previous=json.loads(Path('output/cpu-coherent-multipath/protocol.json').read_text())
    hashes=dict(previous['source_sha256'])
    for p,h in hashes.items(): assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
    for p in (Path(__file__).resolve(),Path('python/gnss_gpu/calibrated_path_selection_cpu.py')):
        hashes[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    prior_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in Path('output/cpu-coherent-multipath').iterdir() if p.is_file()}
    calibration=[(0.,0.,0.,r) for r in range(40000,40299)]
    null=[(0.,0.,0.,r) for r in range(50000,50299)]
    echoes=[(.8,g,p,r) for g in (.1,.5) for p in (0.,np.pi/2) for r in range(60000,60128)]
    protocol=dict(calibration_tasks=calibration,null_evaluation_tasks=null,echo_tasks=echoes,
        source_sha256=hashes,prior_artifact_sha256=prior_hashes,
        scope='PRN1, B=3.6 chip rates, C/N0=37, duration=200 ms, ideal coherence, default v2 assistance',
        null_distribution='Single path, delay uniform [-.3,.3] chips, phase uniform [-pi,pi], spectral Gaussian noise',
        calibration='Maximum of 299 raw BIC-improvement scores; strict exceedance; alpha=.01, delta=.05',
        guarantee='Marginal null score exceedance probability <=1% with >=95% calibration confidence for this distribution only',
        primary='Paired calibrated vs BIC selected success for .1-chip same-phase echo; one exact two-sided McNemar test alpha=.05',
        success='Available absolute direct range error <10 m; all attempted records in denominator',
        independence='Calibration, null evaluation, echo seeds disjoint; 128 echo replicas shared across four conditions',
        limits='Not an acquisition false alarm guarantee or a uniform nuisance-parameter guarantee')
    (OUTPUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    with ProcessPoolExecutor(max_workers=4) as pool:
        cal=[]
        with (OUTPUT/'calibration.jsonl').open('w') as stream:
            for row in pool.map(evaluate,calibration):
                cal.append(row);stream.write(json.dumps(row,allow_nan=False)+'\n')
                if len(cal)%100==0: print(f'Calibration {len(cal)}/299',flush=True)
        gate=calibrate_path_gate([r['fit']['bic_improvement'] for r in cal])
        gatepath=OUTPUT/'frozen_gate.json'
        gatepath.write_text(json.dumps(asdict(gate),indent=2))
        gate_hash=hashlib.sha256(gatepath.read_bytes()).hexdigest()
        # Gate is frozen before any held-out single/echo fitting starts.
        print(f'Frozen score gate: {gate.score:.6f}',flush=True)
        rows=[]
        with (OUTPUT/'evaluation.jsonl').open('w') as stream:
            for raw in pool.map(evaluate,null+echoes):
                row=apply_gate(raw,gate);rows.append(row)
                stream.write(json.dumps(row,allow_nan=False)+'\n')
                if len(rows)%128==0: print(f'Evaluation {len(rows)}/811',flush=True)
    report=summarize(rows)
    (OUTPUT/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    assert hashlib.sha256(gatepath.read_bytes()).hexdigest()==gate_hash
    for p,h in {**hashes,**prior_hashes}.items(): assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
    (OUTPUT/'completion.json').write_text(json.dumps(dict(calibration_records=len(cal),evaluation_records=len(rows),
        frozen_gate_sha256=gate_hash,source_and_prior_hashes_verified=True),indent=2))


if __name__=='__main__': run()
