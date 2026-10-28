"""Paired aperture-input range refinement; conditional on frozen acquisition."""
from dataclasses import asdict
import hashlib,json
from pathlib import Path
import numpy as np
from gnss_gpu import CpuSignalSimulator
from gnss_gpu.wall_signal_sim import C_LIGHT,VerticalWall,simulate_wall_signal
from gnss_gpu.acquisition_cpu import CaSearch
from gnss_gpu.bit_edge_acquisition_cpu import CaBitEdgeSearch,acquire_ca_bit_edge
from gnss_gpu.weak_range_refinement_cpu import refine_weak_ca_delay


def run():
    out=Path('output/cpu-weak-range-refinement');out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):raise ValueError('output must be empty')
    fs=2046000.;cfg=CaBitEdgeSearch(CaSearch(fs,(-250.,0.,250.)))
    artifact=Path('output/cpu-weak-positioning/frozen_thresholds.json')
    frozen=json.loads(artifact.read_text())['1']['detection']
    assert json.loads(json.dumps(asdict(cfg)))==frozen['config']
    sources=[Path(__file__),Path('python/gnss_gpu/weak_range_refinement_cpu.py'),Path('python/gnss_gpu/code_delay_cpu.py'),Path('python/gnss_gpu/wall_signal_sim.py'),Path('python/gnss_gpu/bit_edge_acquisition_cpu.py')]
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    protocol=dict(replicas=64,cn0=[37,55],sampling='centered integrate-and-dump',carrier_hz=0,
        calibration_scope='Unchanged white-null acquisition gate; post-gate estimation only.',
        frozen_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),source_sha256=hashes,
        limitation='Single isolated PRN, static delay, no positioning/forest/rain claim. Not comparable to earlier point-sampled IQ as a pure estimator change.')
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    records=[]
    for replica in range(64):
        rng=np.random.default_rng(np.random.SeedSequence([20261026,0x46524143,replica]))
        delay=float(rng.uniform(-20,20));phase=float(rng.uniform(0,2*np.pi))
        sim=CpuSignalSimulator(fs,noise_floor_db=-10000)
        block=simulate_wall_signal(sim,VerticalWall(0.,-100.,100.,0.,100.),[-10.,0.,2.],[0.,0.,0.],[-100.,0.,2.],2046,
            reference_range_m=90-delay*C_LIGHT/fs,include_reflection=False,integrate_samples=True)
        z=block.iq.reshape(-1,2);z=(z[:,0]+1j*z[:,1])*np.exp(1j*phase)
        clean=np.tile(np.column_stack((z.real,z.imag)).ravel(),(20,1))
        edge=int(rng.integers(1,19));offset=int(rng.integers(1,2046))*2
        clean[edge,offset:]*=-1;clean[edge+1:]*=-1
        noise=rng.normal(size=clean.shape)
        for cn0 in (37,55):
            iq=clean+noise*np.sqrt(fs/(2*10**(cn0/10)))
            candidate=acquire_ca_bit_edge(iq,cfg);accepted=candidate.statistic>frozen['score']
            fitted=refine_weak_ca_delay(iq,cfg,candidate) if accepted else None
            def error(value):return abs((value-delay+1023)%2046-1023)*C_LIGHT/fs
            records.append(dict(replica=replica,cn0=cn0,accepted=accepted,true_delay_samples=delay,
                candidate=asdict(candidate),integer_error_m=error(candidate.delay_samples) if accepted else None,
                fractional_error_m=error(fitted.delay_samples) if fitted else None,
                iq_sha256=hashlib.sha256(iq.tobytes()).hexdigest()))
    summary=[]
    for cn0 in (37,55):
        rows=[r for r in records if r['cn0']==cn0];ok=[r for r in rows if r['accepted']]
        summary.append(dict(cn0=cn0,accepted=len(ok),total=len(rows),**{key:dict(median=float(np.median([r[key] for r in ok])),p95=float(np.quantile([r[key] for r in ok],.95))) for key in ('integer_error_m','fractional_error_m')}))
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    (out/'records.json').write_text(json.dumps(records,indent=2));(out/'report.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))
if __name__=='__main__':run()
