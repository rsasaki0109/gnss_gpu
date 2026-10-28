"""Broader observation-only initialization for the frozen relative-drift model."""

from __future__ import annotations
import numpy as np
from scipy.optimize import minimize, OptimizeResult
from gnss_gpu.relative_drift_paths_cpu import _problem, RelativeDriftFit
from gnss_gpu.noise_aware_multipath_cpu import _select_equivalent_converged
from gnss_gpu.multipath_information_cpu import CHIP_RATE

BOUNDS = [(-1.,1.),(-2.,2.),(-3.,3.),(.01,1.2),(-12.,12.),(-4.,4.)]
KEYS = ('first_delay_chips','common_carrier_hz','common_drift_hz_s',
        'excess_chips','relative_doppler_hz','relative_drift_hz_s')


def search_seeds(solve, seed):
    """729 grid candidates; retain best per carrier/common-drift/echo-drift cell.

    Spatial coordinates are relative to the observation-derived delay. Selection
    uses only residual loss; no generating parameters or true delay are inputs.
    """
    delays=sorted(set(float(np.clip(seed.first_delay_chips+x,-1,1)) for x in (-.05,0.,.05)))
    gaps=sorted(set((seed.excess_chips,.05,.2)))
    starts=[]
    for c in (-1.5,0.,1.5):
        for a in (-2.,0.,2.):
            for q in (-3.,0.,3.):
                candidates=[(d,c,a,g,r,q) for d in delays for g in gaps for r in (-2.,0.,2.)]
                starts.append(min(candidates,key=lambda p:solve(p)[0]))
    return starts


def fit_broad_relative_drift(blocks,weights,seed,bandwidth_hz=3.6*CHIP_RATE):
    solve=_problem(blocks,weights,True,bandwidth_hz)
    bounds=BOUNDS
    original=np.array([getattr(seed,k) for k in KEYS])
    starts=search_seeds(solve,seed)+[original]
    fits=[OptimizeResult(x=original,fun=solve(original)[0],success=seed.converged)]
    fits += [minimize(solve,p,jac=True,method='L-BFGS-B',bounds=bounds,
                     options={'maxiter':400,'ftol':1e-13,'gtol':1e-9}) for p in starts]
    best=_select_equivalent_converged(fits)
    loss, a, jac = solve(best.x, True)
    projected = jac-a@np.linalg.lstsq(a,jac,rcond=None)[0]
    info = np.real(projected.conj().T@projected)
    eig = np.linalg.eigvalsh(info)
    std = None
    if eig[0] > max(1e-20,1e-12*eig[-1]):
        std = float(np.sqrt(max(0,loss/(2*len(a)-10)*np.linalg.inv(info)[0,0])))
    if not best.success:
        reason = 'optimizer did not converge'
    elif min(min(x-lo,hi-x) for x,(lo,hi) in zip(best.x,bounds)) < 1e-4:
        reason = 'search boundary'
    elif std is None or std > .1:
        reason = 'weak local delay information'
    else:
        reason = 'interior locally conditioned fit'
    return RelativeDriftFit(float(best.x[0]), float(best.x[1]), float(best.x[2]), float(best.x[3]), float(best.x[4]), float(best.x[5]),std,bool(best.success),
                            reason=='interior locally conditioned fit',reason,loss)
