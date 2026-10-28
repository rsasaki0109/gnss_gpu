"""Explain saturated null scores by reconstructing fitted complex amplitudes."""
import json
import numpy as np
from gnss_gpu.cached_drift_carrier_paths_cpu import DriftFit,estimate_weights,_problem
from gnss_gpu.residual_dynamic_paths_cpu import _inverse_std
from gnss_gpu.multipath_information_cpu import LIGHT_SPEED,CHIP_RATE
from experiments.run_cpu_drift_ambiguity import OUTPUT,synthesize,digest


def run():
    rows=[json.loads(x) for x in (OUTPUT/'calibration.jsonl').read_text().splitlines()]
    cells=[]
    for r in rows:
        if r['profile_alarm']['score']!=1:continue
        task=[r[k] for k in ('kind','echo_voltage','excess_chips','phase_rad','drift_hz_s','initial_carrier_hz','replica')]
        b,_,_=synthesize(task);w=estimate_weights(b,DriftFit(**r['fits']['preliminary']))
        point=min([p for p in r['profile_alarm']['points'] if p['converged']],key=lambda p:p['loss'])
        dense,a,_=_problem(b,w,True)(np.r_[point['delay_chips'],point['nuisance']],True)
        y=b*_inverse_std(w,b.shape[1]);y=y/np.max(np.abs(y));y=y.ravel()
        amp=np.linalg.lstsq(a,y,rcond=None)[0]
        residual=y-a@amp
        np.testing.assert_allclose(np.vdot(residual,residual).real,point['loss'],rtol=0,atol=1e-9)
        scale=LIGHT_SPEED/CHIP_RATE
        cells.append(dict(replica=r['replica'],first_to_second_voltage_ratio=float(abs(amp[0])/abs(amp[1])),
            first_delay_error_m=(point['delay_chips']-r['true_delay_chips'])*scale,
            second_delay_error_m=(point['delay_chips']+point['nuisance'][2]-r['true_delay_chips'])*scale,
            far_loss=point['loss'],previous_two_loss=r['fits']['two']['loss']))
    report=dict(calibration_sha256=digest(OUTPUT/'calibration.jsonl'),source_sha256=digest(__file__),cells=cells,
        scope='Post-hoc reconstruction of all exactly saturated null scores; truth used only to measure fitted path offsets. No amplitude constraint or recalibration applied.')
    (OUTPUT/'saturation.json').write_text(json.dumps(report,indent=2))
    print(f'Reconstructed amplitudes for {len(cells)} saturated nulls')


if __name__=='__main__':run()
