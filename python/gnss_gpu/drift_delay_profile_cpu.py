"""Finite multistart delay profiles with frozen observation-derived weights.

These losses are diagnostics, not confidence bounds or calibrated protection levels.
No generating delay, amplitude or model order is accepted by this API.
"""
import numpy as np
from scipy.optimize import minimize
from gnss_gpu.cached_drift_carrier_paths_cpu import _problem


def fixed_delay_objective(solve, delay):
    def objective(nuisance):
        loss, gradient = solve(np.r_[delay, nuisance])
        return loss, gradient[1:]
    return objective


def delay_profile(blocks, weights, delays, single, two):
    delays = np.asarray(delays, dtype=float)
    if delays.ndim != 1 or not len(delays) or not np.all(np.isfinite(delays)) or np.any(np.abs(delays)>1):
        raise ValueError('nonempty finite delay grid within [-1,1] chips required')
    results = {}
    for is_two, name in ((False,'single'), (True,'two')):
        solve = _problem(blocks, weights, is_two)
        bounds = [(-2.,2.),(-3.,3.)] + ([(.01,1.2),(-12.,12.)] if is_two else [])
        base = [(s.common_carrier_hz,s.common_drift_hz_s) for s in (single,two)]
        base += [(float(np.clip(single.common_carrier_hz+c,-2,2)),single.common_drift_hz_s) for c in (-.5,.5)]
        if is_two:
            seeds = [(*p,g,r) for p in base for g in (.01,.03,.1,.3,1.) for r in (-1.,0.,1.)]
            seeds.append((two.common_carrier_hz,two.common_drift_hz_s,two.excess_chips,two.relative_doppler_hz))
        else:
            seeds = base + [(0.,0.)]
        points = []
        for delay in delays:
            objective = fixed_delay_objective(solve, delay)
            starts = sorted(seeds,key=lambda p:objective(p)[0])[:4]
            fits = [minimize(objective,p,jac=True,method='L-BFGS-B',bounds=bounds,
                    options=dict(maxiter=300,ftol=1e-13,gtol=1e-9)) for p in starts]
            # Retain the smallest loss even if nonconverged, and expose that fact.
            best = min(fits,key=lambda r:r.fun)
            dense = solve(np.r_[delay,best.x],True)[0]
            points.append(dict(delay_chips=float(delay),loss=dense,nuisance=best.x.tolist(),
                converged=bool(best.success),boundary=bool(min(min(x-lo,hi-x) for x,(lo,hi) in zip(best.x,bounds))<1e-4),
                compact_dense_abs_difference=abs(float(best.fun)-dense)))
        results[name] = points
    reference = min(single.loss,two.loss,*(p['loss'] for points in results.values() for p in points))
    for points in results.values():
        for p in points:
            p['relative_loss'] = float(2*np.asarray(blocks).size*np.log(p['loss']/reference))
    return dict(reference_loss=reference,profiles=results)
