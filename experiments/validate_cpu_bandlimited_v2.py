"""Fresh paired validation of numerical tie handling; preserve frozen v1."""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import numpy as np
from experiments.run_cpu_bandlimited_multipath import evaluate
from experiments.validate_cpu_bandlimited_multipath import paired_summary, KEYS
from gnss_gpu.bandlimited_multipath_v2_cpu import fit_bandlimited_paths
from gnss_gpu.multipath_information_cpu import ca_passband_spectrum, CHIP_RATE, LIGHT_SPEED

OUTPUT = Path('output/cpu-bandlimited-v2')


def evaluate_pair(task):
    old = evaluate(task)
    b, cn0, ratio, gap, phase, replica = task
    rng = np.random.default_rng(np.random.SeedSequence([20261026,0x424C4D50,replica]))
    delay = float(rng.uniform(-.3,.3))
    common_phase = float(rng.uniform(-np.pi,np.pi))
    frequency, spectrum = ca_passband_spectrum(3.6*CHIP_RATE)
    noise = rng.normal(size=len(frequency))+1j*rng.normal(size=len(frequency))
    mask = abs(frequency) <= b*CHIP_RATE
    frequency, spectrum = frequency[mask], spectrum[mask]
    y = spectrum*np.exp(-2j*np.pi*frequency*delay/CHIP_RATE)*np.exp(1j*common_phase)
    y *= 1+ratio*np.exp(1j*phase-2j*np.pi*frequency*gap/CHIP_RATE)
    y += noise[mask]/np.sqrt(2*10**(cn0/10)*.020)
    assert hashlib.sha256(y.tobytes()).hexdigest() == old['observation_sha256']
    fit = fit_bandlimited_paths(y,b*CHIP_RATE)
    error = ((fit.selected_delay_chips-delay)*LIGHT_SPEED/CHIP_RATE
             if fit.selected_delay_chips is not None else None)
    return dict(old=old, new_fit=asdict(fit), new_error_m=error)


def report_records(records):
    groups = {}
    for row in records:
        groups.setdefault(tuple(row['old'][k] for k in KEYS),[]).append(row)
    report = []
    for key, rows in groups.items():
        old = [r['old']['selected_error_m'] is not None and abs(r['old']['selected_error_m'])<10 for r in rows]
        new = [r['new_error_m'] is not None and abs(r['new_error_m'])<10 for r in rows]
        report.append(dict(zip(KEYS,key),**paired_summary(old,new),
            old_unavailable=sum(r['old']['selected_error_m'] is None for r in rows),
            new_unavailable=sum(r['new_error_m'] is None for r in rows),
            old_nonconverged=sum(r['old']['fit']['reason']=='optimizer did not converge' for r in rows),
            new_nonconverged=sum(r['new_fit']['reason']=='optimizer did not converge' for r in rows)))
    return report


def run():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    if any(OUTPUT.iterdir()): raise ValueError('output must be empty')
    previous = json.loads(Path('output/cpu-bandlimited-multipath-replication/protocol.json').read_text())
    hashes = dict(previous['source_sha256'])
    hashes.update(previous['previous_artifact_sha256'])
    for path,digest in hashes.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
    for path in (Path(__file__).resolve(),Path('python/gnss_gpu/bandlimited_multipath_v2_cpu.py')):
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    for path in Path('output/cpu-bandlimited-multipath-replication').iterdir():
        if path.is_file(): hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    cells = [(0.,0.,0.)]+[(.8,g,p) for g in (.1,.5) for p in (0.,np.pi/2)]
    tasks = [(b,c,a,g,p,r) for b in (.9,3.6) for c in (37.,55.)
             for a,g,p in cells for r in range(20000,20064)]
    protocol = dict(tasks=tasks, source_and_prior_sha256=hashes,
        replicas_per_cell=64, old_replica_sets='0..7 and 10000..10127',
        scope='Same assisted spectral simulator; v1/v2 on identical fresh IQ; no acquisition/PVT',
        change='Only converged numerical tie preference: 64 eps objective tolerance, 1e-7 chip parameter tolerance',
        success='Available absolute range error <10 m, all attempts retained',
        criteria='Descriptive paired regression screen: report every gain/loss and availability; no population noninferiority claim',
        statistical_scope='Cell intervals/tests pointwise exploratory; no multiple-comparison significance claim')
    (OUTPUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    records=[]
    with (OUTPUT/'records.jsonl').open('w') as stream, ProcessPoolExecutor(max_workers=4) as pool:
        for record in pool.map(evaluate_pair,tasks):
            records.append(record)
            stream.write(json.dumps(record,allow_nan=False)+'\n')
            if len(records)%128==0:
                stream.flush();print(f'{len(records)}/{len(tasks)} paired records',flush=True)
    report=report_records(records)
    (OUTPUT/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    assert len(records)==1280 and len(report)==20
    identities=[tuple(r['old'][k] for k in KEYS)+(r['old']['replica'],) for r in records]
    assert identities==tasks and len(set(identities))==1280
    for path,digest in hashes.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
    # Every formerly available result should remain unchanged: tie replacement
    # only operates when the old two-path winning optimizer failed.
    for row in records:
        if row['old']['selected_error_m'] is not None:
            assert row['new_error_m']==row['old']['selected_error_m']
    for index in (0,1279):
        assert json.loads(json.dumps(evaluate_pair(tasks[index])))==json.loads(json.dumps(records[index]))
    audit=dict(records=1280,cells=20,all_iq_pair_hashes_equal=True,
        prior_sources_and_artifacts_unchanged=True,previously_available_outputs_exactly_preserved=True,
        replay_indices=[0,1279],gains=sum(r['gains'] for r in report),losses=sum(r['losses'] for r in report),
        note='Pooled counts are descriptive; 64 shared replica IDs, not 1280 independent trials')
    (OUTPUT/'audit.json').write_text(json.dumps(audit,indent=2))
    print(json.dumps(audit,indent=2))


if __name__=='__main__': run()
