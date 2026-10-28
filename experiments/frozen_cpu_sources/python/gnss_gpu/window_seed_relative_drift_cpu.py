"""Observation-only half-window carrier seeds for a full relative-drift fit."""
import numpy as np
from scipy.optimize import minimize, OptimizeResult
from gnss_gpu.relative_drift_paths_cpu import _problem, RelativeDriftFit
from gnss_gpu.broad_relative_drift_cpu import BOUNDS, KEYS
from gnss_gpu.noise_aware_multipath_cpu import _select_equivalent_converged
from gnss_gpu.multipath_information_cpu import CHIP_RATE


def constant_carrier_objective(solve):
    """Embed [delay, common frequency, gap, relative frequency] in six variables."""
    def objective(p):
        loss,grad=solve((p[0],p[1],0.,p[2],p[3],0.))
        return loss,grad[[0,1,3,4]]
    return objective


def join_windows(left,right,centers):
    """Convert midpoint frequencies to initial frequency and slope (unclipped)."""
    t0,t1=centers
    if not np.isfinite(t0+t1) or t1<=t0:
        raise ValueError('finite increasing window centers required')
    a=(right[1]-left[1])/(t1-t0)
    q=(right[3]-left[3])/(t1-t0)
    return np.array([.5*(left[0]+right[0]),left[1]-a*t0,a,
                     .5*(left[2]+right[2]),left[3]-q*t0,q])


def half_candidates(blocks,weights,seed,center,bandwidth_hz):
    solve=constant_carrier_objective(_problem(blocks,weights,True,bandwidth_hz))
    # Wider midpoint-frequency bounds cover every full-model bound at t<=.6s.
    bounds=[(-1.,1.),(-4.,4.),(.01,1.2),(-15.,15.)]
    c0=float(np.clip(seed.common_carrier_hz+seed.common_drift_hz_s*center,-4,4))
    r0=float(np.clip(seed.relative_doppler_hz+seed.relative_drift_hz_s*center,-15,15))
    delays=sorted(set(float(np.clip(seed.first_delay_chips+x,-1,1)) for x in (-.05,0.,.05)))
    gaps=sorted(set((seed.excess_chips,.05,.2)))
    starts=[]
    # Preserve diversity in common frequency instead of taking only pooled best seeds.
    for c in sorted(set((-2.,0.,2.,c0))):
        candidates=[(d,c,g,r) for d in delays for g in gaps for r in (-3.,0.,3.,r0)]
        starts.extend(sorted(candidates,key=lambda p:solve(p)[0])[:2])
    fits=[minimize(solve,p,jac=True,method='L-BFGS-B',bounds=bounds,
                   options={'maxiter':300,'ftol':1e-13,'gtol':1e-9}) for p in starts]
    # Only converged windows seed the full fit; no truth-based availability filtering.
    return [f.x for f in sorted(fits,key=lambda f:f.fun) if f.success][:3]


def fit_window_seeded_drift(blocks,weights,seed,bandwidth_hz=3.6*CHIP_RATE):
    if np.asarray(blocks).shape[0]!=40:
        raise ValueError('40 blocks required for two 400ms windows')
    halves=[half_candidates(blocks[:20],weights,seed,.2,bandwidth_hz),
            half_candidates(blocks[20:],weights,seed,.6,bandwidth_hz)]
    original=np.array([getattr(seed,k) for k in KEYS])
    starts=[original];clipped=0
    for left in halves[0]:
        for right in halves[1]:
            raw=join_windows(left,right,(.2,.6))
            p=np.clip(raw,np.array(BOUNDS)[:,0],np.array(BOUNDS)[:,1])
            clipped+=int(np.any(p!=raw))
            starts.append(p)
            spatial=p.copy();spatial[[0,3]]=original[[0,3]];starts.append(spatial)
    solve=_problem(blocks,weights,True,bandwidth_hz)
    bounds=BOUNDS
    fits=[OptimizeResult(x=original,fun=solve(original)[0],success=seed.converged)]
    fits += [minimize(solve,p,jac=True,method='L-BFGS-B',bounds=bounds,
                      options={'maxiter':400,'ftol':1e-13,'gtol':1e-9}) for p in starts]
    best=_select_equivalent_converged(fits)
    loss,a,jac=solve(best.x,True)
    projected=jac-a@np.linalg.lstsq(a,jac,rcond=None)[0]
    info=np.real(projected.conj().T@projected);eig=np.linalg.eigvalsh(info)
    std=None
    if eig[0]>max(1e-20,1e-12*eig[-1]):
        std=float(np.sqrt(max(0,loss/(2*len(a)-10)*np.linalg.inv(info)[0,0])))
    reason=('optimizer did not converge' if not best.success else
            'search boundary' if min(min(x-lo,hi-x) for x,(lo,hi) in zip(best.x,bounds))<1e-4 else
            'weak local delay information' if std is None or std>.1 else
            'interior locally conditioned fit')
    fit=RelativeDriftFit(*map(float,best.x),std,bool(best.success),
                         reason=='interior locally conditioned fit',reason,loss)
    return fit,dict(window_candidates=[len(h) for h in halves],clipped_pairs=clipped,
                    full_starts=len(starts),window_parameters=[[p.tolist() for p in h] for h in halves])
