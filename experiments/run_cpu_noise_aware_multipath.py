"""Estimate noise from static block differences, calibrate, then hold out."""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
import hashlib,json
import numpy as np
from scipy.stats import binomtest
from gnss_gpu.multipath_information_cpu import ca_passband_spectrum,CHIP_RATE,LIGHT_SPEED
from gnss_gpu.spectral_noise_cpu import spectral_noise_shape
from gnss_gpu.bandlimited_multipath_v2_cpu import fit_bandlimited_paths
from gnss_gpu.noise_aware_multipath_cpu import fit_noise_aware_paths
from gnss_gpu.calibrated_path_selection_cpu import PathSelectionGate,calibrate_path_gate,_select_default_fit
from experiments.validate_cpu_bandlimited_multipath import paired_summary

OUTPUT=Path('output/cpu-noise-aware-multipath')
PROFILES=('white','center','edge')
OLD_GATE=Path('output/cpu-calibrated-path-selection/frozen_gate.json')


def evaluate(task):
    profile,ratio,gap,phase,replica,compare=task
    rng=np.random.default_rng(np.random.SeedSequence([20261026,0x4E415745,replica]))
    delay=float(rng.uniform(-.3,.3));common_phase=float(rng.uniform(-np.pi,np.pi))
    f,s=ca_passband_spectrum(3.6*CHIP_RATE)
    signal=s*np.exp(-2j*np.pi*f*delay/CHIP_RATE+1j*common_phase)
    signal*=1+ratio*np.exp(1j*phase-2j*np.pi*f*gap/CHIP_RATE)
    noise=rng.normal(size=(10,len(f)))+1j*rng.normal(size=(10,len(f)))
    blocks=signal+noise*np.sqrt(spectral_noise_shape(f,profile)/(2*10**3.7*.020))
    fit=fit_noise_aware_paths(blocks)
    old=fit_bandlimited_paths(blocks.mean(axis=0),3.6*CHIP_RATE) if compare else None
    return dict(profile=profile,echo_voltage=ratio,gap_chips=gap,phase_rad=phase,replica=replica,
        true_delay_chips=delay,fit=asdict(fit),old_fit=asdict(old) if old else None,
        observation_sha256=hashlib.sha256(blocks.tobytes()).hexdigest())


def select(row,gate,old_gate):
    from gnss_gpu.noise_aware_multipath_cpu import BandlimitedFit
    new=_select_default_fit(BandlimitedFit(**row['fit']),gate)
    old=_select_default_fit(BandlimitedFit(**row['old_fit']),old_gate)
    def error(value): return (value-row['true_delay_chips'])*LIGHT_SPEED/CHIP_RATE if value is not None else None
    return dict(row,new_selected=asdict(new),old_selected=asdict(old),
        new_error_m=error(new.selected_delay_chips),old_error_m=error(old.selected_delay_chips),
        new_score_pass=row['fit']['bic_improvement']>gate.score)


def summarize(rows):
    groups={}
    for r in rows: groups.setdefault((r['profile'],r['echo_voltage'],r['phase_rad']),[]).append(r)
    report=[]
    for (profile,ratio,phase),group in groups.items():
        a=[r['old_error_m'] is not None and abs(r['old_error_m'])<10 for r in group]
        b=[r['new_error_m'] is not None and abs(r['new_error_m'])<10 for r in group]
        paired=paired_summary(a,b)
        count=sum(r['new_selected']['selected_model']=='two_path' for r in group)
        ci=binomtest(count,len(group)).proportion_ci(confidence_level=.95,method='exact')
        primary=ratio==.8 and phase==0 and profile in ('white','center')
        report.append(dict(profile=profile,echo_voltage=ratio,phase_rad=phase,**paired,
            old_two_paths=sum(r['old_selected']['selected_model']=='two_path' for r in group),
            new_two_paths=count,new_two_path_exact_95=list(ci),
            new_score_passes=sum(r['new_score_pass'] for r in group),
            old_unavailable=sum(r['old_error_m'] is None for r in group),
            new_unavailable=sum(r['new_error_m'] is None for r in group),primary=primary,
            adjusted_primary_p=min(1.,2*paired['paired_exact_p']) if primary else None))
    return report


def run():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    if any(OUTPUT.iterdir()): raise ValueError('output must be empty')
    previous=json.loads(Path('output/cpu-path-selection-stress/protocol.json').read_text())
    hashes=dict(previous['source_sha256'])
    for p,h in hashes.items(): assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
    for p in (Path(__file__).resolve(),Path('python/gnss_gpu/noise_aware_multipath_cpu.py')):
        hashes[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    prior={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('output/cpu-path-selection-stress').iterdir() if p.is_file()}
    prior[str(OLD_GATE)]=hashlib.sha256(OLD_GATE.read_bytes()).hexdigest()
    cal=[(p,0.,0.,0.,70000+j*1000+r,False) for j,p in enumerate(PROFILES) for r in range(408)]
    null=[(p,0.,0.,0.,80000+r,True) for p in PROFILES for r in range(128)]
    echoes=[(p,.8,.1,phase,90000+r,True) for p in PROFILES for phase in (0.,np.pi/2) for r in range(64)]
    protocol=dict(calibration_tasks=cal,evaluation_tasks=null+echoes,source_sha256=hashes,prior_sha256=prior,
        scope='PRN1, 37 reference dB-Hz, 200 ms, 3.683 MHz one-sided, ideal static carrier/nav-wiped signal',
        noise_estimation='Across-block sample variance, 101-harmonic moving average with edge-count normalization, relative floor 1e-6',
        calibration='408 iid single-path records per profile; max score per stratum; combined gate=max of 3 stratum gates',
        guarantee='Each stratum null exceedance <=1% with joint calibration confidence >=95%; union bound 3*.99**408 <= .05',
        primary='Same-phase .1-chip echo success new vs old, white and center profiles; paired two-sided exact tests, Bonferroni two-test family',
        pairing='Evaluation replicas shared across profiles/echo phases; calibration, null and echoes use disjoint IDs',
        limitation='Noise estimation requires static aligned blocks. Joint whitening+recalibration change, not isolation of whitening alone.')
    (OUTPUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    old_gate=PathSelectionGate(**json.loads(OLD_GATE.read_text()))
    with ProcessPoolExecutor(max_workers=4) as pool:
        calibration=[]
        with (OUTPUT/'calibration.jsonl').open('w') as stream:
            for row in pool.map(evaluate,cal):
                calibration.append(row);stream.write(json.dumps(row,allow_nan=False)+'\n')
                if len(calibration)%204==0: print(f'Calibration {len(calibration)}/1224',flush=True)
        strata={p:calibrate_path_gate([r['fit']['bic_improvement'] for r in calibration if r['profile']==p],confidence_failure=.05/3) for p in PROFILES}
        gate=PathSelectionGate(max(g.score for g in strata.values()),1224,.01,.05)
        gates=dict(combined=asdict(gate),strata={p:asdict(g) for p,g in strata.items()})
        gatepath=OUTPUT/'frozen_gate.json';gatepath.write_text(json.dumps(gates,indent=2))
        gatehash=hashlib.sha256(gatepath.read_bytes()).hexdigest()
        print(f'Frozen gate {gate.score:.6f}',flush=True)
        rows=[]
        with (OUTPUT/'evaluation.jsonl').open('w') as stream:
            for raw in pool.map(evaluate,null+echoes):
                row=select(raw,gate,old_gate);rows.append(row);stream.write(json.dumps(row,allow_nan=False)+'\n')
                if len(rows)%128==0: print(f'Evaluation {len(rows)}/768',flush=True)
    (OUTPUT/'report.json').write_text(json.dumps(summarize(rows),indent=2,allow_nan=False))
    for p,h in {**hashes,**prior}.items(): assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
    assert hashlib.sha256(gatepath.read_bytes()).hexdigest()==gatehash
    (OUTPUT/'completion.json').write_text(json.dumps(dict(calibration=1224,evaluation=768,frozen_gate_sha256=gatehash),indent=2))


if __name__=='__main__': run()
