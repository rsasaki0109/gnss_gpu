"""Independent paired bandwidth replication with a single prespecified contrast."""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
import hashlib,json
import numpy as np
from scipy.stats import binomtest
from experiments.run_cpu_drift_bandwidth import synthesize as echo_synthesize
from experiments.run_cpu_drift_selection import tasks,digest
from experiments.stress_cpu_carrier_drift import chirp_blocks
from gnss_gpu.bandwidth_drift_paths_cpu import score_drift_paths
from gnss_gpu.multipath_information_cpu import CHIP_RATE,LIGHT_SPEED,ca_passband_spectrum

OUTPUT=Path('output/cpu-drift-bandwidth-replication')
PRIOR=Path('output/cpu-drift-bandwidth')
BANDS=(3.6,7.2)


def tasklist():
    return [(kind,band,a,f,i) for kind,start,count in [('null',1250000,32),('echo',1260000,64)]
            for band in BANDS for _,a,f,i in tasks(kind,start,count)]


def synthesize(task):
    kind,band,drift,offset,replica=task
    b,old,truth=echo_synthesize((band,drift,offset,replica))
    if kind=='echo':return b,old,truth
    if kind!='null':raise ValueError('null or echo required')
    rng=np.random.default_rng(np.random.SeedSequence([20261027,0x42414E44,replica]))
    assert float(rng.uniform(-.3,.3))==truth
    phase=float(rng.uniform(-np.pi,np.pi))
    f,s=ca_passband_spectrum(band*CHIP_RATE)
    signal=s*np.exp(-2j*np.pi*f*truth/CHIP_RATE+1j*phase)*chirp_blocks(offset,drift,40)[:,None]
    return signal+(b-old),signal,truth


def evaluate(task):
    b,_,truth=synthesize(task);score,w=score_drift_paths(b,task[1]*CHIP_RATE)
    fit=score['single' if task[0]=='null' else 'two']
    error=(fit.first_delay_chips-truth)*LIGHT_SPEED/CHIP_RATE
    return dict(task=task,kind=task[0],bandwidth_chips=task[1],replica=task[-1],true_delay_chips=truth,
        fits={k:asdict(score[k]) for k in ('single','two','preliminary')},uncalibrated_score=score['score'],
        raw_error_m=error,available=fit.locally_usable,success=bool(fit.locally_usable and abs(error)<10),
        observation_sha256=hashlib.sha256(b.tobytes()).hexdigest(),weights_sha256=hashlib.sha256(w.tobytes()).hexdigest())


def paired_outcomes(narrow,wide):
    a={r['replica']:r['success'] for r in narrow};b={r['replica']:r['success'] for r in wide}
    if not a or a.keys()!=b.keys() or len(a)!=len(narrow) or len(b)!=len(wide):
        raise ValueError('nonempty uniquely matched replicas required')
    gains=sum(b[i] and not a[i] for i in a);losses=sum(a[i] and not b[i] for i in a)
    discordant=gains+losses
    return dict(pairs=len(a),gains=gains,losses=losses,both_success=sum(a[i] and b[i] for i in a),
        both_failure=sum(not a[i] and not b[i] for i in a),success_rate_difference=(gains-losses)/len(a),
        exact_two_sided_p=binomtest(gains,discordant,.5,alternative='two-sided').pvalue if discordant else 1.)


def summarize(rows):
    cells=[]
    for kind in ('null','echo'):
        for band in BANDS:
            g=[r for r in rows if r['kind']==kind and r['bandwidth_chips']==band];n=len(g);k=sum(r['success'] for r in g)
            cells.append(dict(kind=kind,bandwidth_chips=band,attempts=n,successes=k,
                success_exact_95=list(binomtest(k,n).proportion_ci(method='exact')),unavailable=sum(not r['available'] for r in g),
                available_large_errors=sum(r['available'] and abs(r['raw_error_m'])>=10 for r in g)))
    primary=paired_outcomes([r for r in rows if r['kind']=='echo' and r['bandwidth_chips']==3.6],
                            [r for r in rows if r['kind']=='echo' and r['bandwidth_chips']==7.2])
    return dict(cells=cells,primary_echo_comparison=primary,
        scope='One prespecified exact two-sided paired test,alpha.05,wide vs baseline echo success. Null cells descriptive known-single-path controls,not false-detection estimates. Truth used only to choose known order for evaluation,not to fit parameters. No prior pilot pooling.')


def run():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    if any(OUTPUT.iterdir()):raise ValueError('output must be empty')
    sources=dict(json.loads((PRIOR/'protocol.json').read_text())['source_sha256'])
    sources[str(Path(__file__).resolve())]=digest(__file__)
    assert all(digest(path)==h for path,h in sources.items())
    ts=tasklist()
    p=dict(source_sha256=sources,tasks=ts,
        design='Fresh64 echo pairs and32 null pairs;one-sided3.6vs7.2chip-rate.800ms,PRN1,pre-filter direct37dBHz,constant noise PSD. Echo voltage.8,gap.03chip,phasepi,relative Doppler0. Frozen generalized receiver,weights/guards unchanged. No calibration gate reused.',
        primary='Only confirmatory contrast:paired echo success(available and abs(error)<10m),wide vs baseline.Exact two-sided McNemar via binomial discordances,alpha.05. All64 pairs including unavailable;no early stopping or pilot pooling.',
        controls='32 fresh null pairs with known single-path order;descriptive range success/availability only,not detector false-alarm. Echo evaluation uses known two-path order. True delay never passed to optimizer.',
        limits='Synthetic fixed paths,white noise,ideal nav wipeoff,common linear drift. No RF/rain/forest/PVT or general operational guarantee.')
    (OUTPUT/'protocol.json').write_text(json.dumps(p,indent=2))
    rows=[]
    with ProcessPoolExecutor(max_workers=4) as pool,(OUTPUT/'evaluation.jsonl').open('w') as stream:
        for r in pool.map(evaluate,ts):
            rows.append(r);stream.write(json.dumps(r,allow_nan=False)+'\n');stream.flush()
            if len(rows)%16==0:print(f'Replication {len(rows)}/192',flush=True)
    (OUTPUT/'report.json').write_text(json.dumps(summarize(rows),indent=2))
    assert all(digest(path)==h for path,h in sources.items())
    (OUTPUT/'completion.json').write_text(json.dumps({name:digest(OUTPUT/name) for name in ('protocol.json','evaluation.jsonl','report.json')},indent=2))


if __name__=='__main__':run()
