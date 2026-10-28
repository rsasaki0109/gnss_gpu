"""Paired direct+echo sweep at the weak receiver's sample rate, no truth in inference."""
from dataclasses import asdict
from pathlib import Path
import hashlib,json
import numpy as np
from gnss_gpu.validation.tracking_residuals import generate_ca_code
from gnss_gpu.acquisition_cpu import CaSearch
from gnss_gpu.bit_edge_acquisition_cpu import CaBitEdgeSearch,acquire_ca_bit_edge
from gnss_gpu.weak_range_refinement_cpu import refine_weak_ca_delay
from gnss_gpu.weak_multipath_cpu import fit_weak_ca_two_paths

FS=2046000.; C=299792458.

def aperture(delay):
    code=generate_ca_code(1); prefix=np.r_[0.,np.cumsum(code)]
    def primitive(x):
        chip=(x-delay)/2; whole=np.floor(chip).astype(int)
        return 2*(whole//1023*prefix[-1]+prefix[whole%1023]+(chip-whole)*code[whole%1023])
    centers=np.arange(2046)
    return primitive(centers+.5)-primitive(centers-.5)


def run():
    output=Path('output/cpu-weak-multipath');output.mkdir(parents=True,exist_ok=True)
    if any(output.iterdir()):raise ValueError('output must be empty')
    cfg=CaBitEdgeSearch(CaSearch(FS,(-250.,0.,250.)))
    gatepath=Path('output/cpu-fractional-positioning/frozen_thresholds.json')
    gate=json.loads(gatepath.read_text())['1']['detection']
    assert json.loads(json.dumps(asdict(cfg)))==gate['config']
    cells=[(0.,0.,0.)]+[(a,d,p) for a in (.3,.8,1.5) for d in (.1,.5,1.) for p in (0.,np.pi/2,np.pi)]
    paths=[Path(__file__)]+[Path('python/gnss_gpu')/(n+'.py') for n in ('weak_multipath_cpu','weak_range_refinement_cpu','multipath_fit_cpu','code_delay_cpu','bit_edge_acquisition_cpu')]
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    protocol=dict(replicas=8,cn0=[37,55],cells=cells,amplitude_definition='echo/direct voltage',delay_unit='chips',
        phase_definition='net echo/direct complex phase',polarity='constant over 20 ms',source_sha256=hashes,
        gate_sha256=hashlib.sha256(gatepath.read_bytes()).hexdigest(),seed_domain=0x4D505354,
        fit_settings=dict(search_radius_chips=.5,max_excess_chips=1.,max_delay_std_samples=.25),
        scope='Isolated PRN, static coherent two-path aperture IQ. Echo stronger than direct violates estimator prior. No forest/rain or positioning claim.')
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2))
    records=[]
    for cn0 in (37,55):
        for amplitude,gap,phase in cells:
            for replica in range(8):
                rng=np.random.default_rng(np.random.SeedSequence([20261026,0x4D505354,replica]))
                delay=float(rng.uniform(-20,20));angle=float(rng.uniform(0,2*np.pi))
                z=(aperture(delay)+amplitude*np.exp(1j*phase)*aperture(delay+2*gap))*np.exp(1j*angle)
                iq=np.tile(np.column_stack((z.real,z.imag)).ravel(),(20,1))
                iq+=rng.normal(size=iq.shape)*np.sqrt(FS/(2*10**(cn0/10)))
                candidate=acquire_ca_bit_edge(iq,cfg);accepted=candidate.statistic>gate['score']
                single=refine_weak_ca_delay(iq,cfg,candidate) if accepted else None
                fit=fit_weak_ca_two_paths(iq,cfg,candidate,**protocol['fit_settings']) if accepted else None
                def error(x):return abs((x-delay+1023)%2046-1023)*C/FS if x is not None else None
                records.append(dict(cn0=cn0,amplitude=amplitude,gap_chips=gap,phase_rad=phase,replica=replica,
                    accepted=accepted,candidate=asdict(candidate),true_delay_samples=delay,
                    single_error_m=error(single.delay_samples) if single else None,
                    selected_error_m=error(fit.delay_samples) if fit else None,
                    selected_model=fit.selected_model if fit else 'unavailable',reason=fit.reason if fit else 'no accepted fit',
                    iq_sha256=hashlib.sha256(iq.tobytes()).hexdigest()))
            print(cn0,amplitude,gap,phase,flush=True)
    summaries=[]
    for cn0 in (37,55):
        for amplitude,gap,phase in cells:
            rows=[r for r in records if (r['cn0'],r['amplitude'],r['gap_chips'],r['phase_rad'])==(cn0,amplitude,gap,phase)]
            available=[r for r in rows if r['selected_error_m'] is not None]
            summaries.append(dict(cn0=cn0,amplitude=amplitude,gap_chips=gap,phase_rad=phase,total=len(rows),
                accepted=sum(r['accepted'] for r in rows),selected=len(available),
                ambiguous=sum(r['selected_model']=='ambiguous' for r in rows),
                single_under_10m=sum(r['single_error_m'] is not None and r['single_error_m']<10 for r in rows),
                selected_under_10m=sum(r['selected_error_m'] is not None and r['selected_error_m']<10 for r in rows),
                single_median_same_selected=float(np.median([r['single_error_m'] for r in available])) if available else None,
                selected_median=float(np.median([r['selected_error_m'] for r in available])) if available else None))
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    (output/'records.json').write_text(json.dumps(records,indent=2));(output/'report.json').write_text(json.dumps(summaries,indent=2))
if __name__=='__main__':run()
