"""Known two-path refinement with independent echo carrier drift.

Fixed code delays remain an approximation. Reuse observation-derived weights
and seeds from the existing receiver; no model-order selection is calibrated.
"""
from dataclasses import dataclass
from functools import lru_cache
import numpy as np
from scipy.optimize import minimize
from gnss_gpu.multipath_information_cpu import ca_passband_spectrum, CHIP_RATE
from gnss_gpu.residual_dynamic_paths_cpu import _inverse_std
from gnss_gpu.noise_aware_multipath_cpu import _select_equivalent_converged
from gnss_gpu.bandwidth_drift_paths_cpu import temporal_chirp


@dataclass(frozen=True)
class RelativeDriftFit:
    first_delay_chips: float
    common_carrier_hz: float
    common_drift_hz_s: float
    excess_chips: float
    relative_doppler_hz: float
    relative_drift_hz_s: float
    first_std_chips: float | None
    converged: bool
    locally_usable: bool
    reason: str
    loss: float


def _problem(blocks, weights, two, bandwidth_hz=3.6*CHIP_RATE):
    f, s = ca_passband_spectrum(bandwidth_hz)
    y = np.asarray(blocks, dtype=complex)
    if y.ndim != 2 or y.shape[0] not in (10,20,40) or y.shape[1] != len(s) or not np.all(np.isfinite(y)) or np.max(np.abs(y)) == 0:
        raise ValueError('finite nonzero 10/20/40-block full-passband observation required')
    inv = _inverse_std(weights, len(s))
    y = y*inv
    y = y/np.max(np.abs(y))
    s = s*inv
    slope = -2j*np.pi*f/CHIP_RATE
    energy = float(np.vdot(y, y).real)

    @lru_cache(maxsize=64)
    def spatial(delay):
        # Owned by this observation/weight problem; never shared across signals.
        # Cache only identical scalar delays, without rounding or approximation.
        v = s*np.exp(slope*delay)
        dv = slope*v
        return v, dv, y@v.conj(), y@dv.conj()

    def solve(p, details=False):
        first, common, drift = p[:3]
        c, dc, ac = temporal_chirp(common, drift, y.shape[0])
        v, dv, q, dq = spatial(first)
        projections = {id(v): q, id(dv): dq}
        templates = [(c, v)]
        derivatives = [[(c, dv)], [(dc, v)], [(ac, v)]]
        if two:
            gap, relative, relative_drift = p[3:]
            e, de, ae = temporal_chirp(common+relative, drift+relative_drift, y.shape[0])
            w, dw, qw, dqw = spatial(first+gap)
            projections.update({id(w): qw, id(dw): dqw})
            templates.append((e, w))
            derivatives[0].append((e, dw))
            derivatives[1].append((de, w))
            derivatives[2].append((ae, w))
            derivatives += [[None, (e, dw)], [None, (de, w)], [None, (ae, w)]]
        inner = lambda a, b: np.vdot(a[0], b[0])*np.vdot(a[1], b[1])
        corr = lambda a: np.vdot(a[0], projections[id(a[1])])
        gram = np.array([[inner(a,b) for b in templates] for a in templates])
        rhs = np.array([corr(a) for a in templates])
        amp = np.linalg.solve(gram, rhs)
        loss = float(energy-np.vdot(rhs, amp).real)
        if loss < 1e-8*energy:
            residual = y-sum(u[:,None]*v*amp[i] for i,(u,v) in enumerate(templates))
            loss = float(np.vdot(residual, residual).real)
        grad = []
        for terms in derivatives:
            value = sum(amp[i].conjugate()*(corr(a)-sum(inner(a,b)*amp[j] for j,b in enumerate(templates)))
                        for i,a in enumerate(terms) if a is not None)
            grad.append(-2*value.real)
        if not details:
            return loss, np.array(grad)
        a = np.column_stack([(u[:,None]*v).ravel() for u,v in templates])
        jac = np.column_stack([sum((term[0][:,None]*term[1]*amp[i]).ravel()
              for i,term in enumerate(terms) if term is not None) for terms in derivatives])
        residual = y.ravel()-a@amp
        return float(np.vdot(residual, residual).real), a, jac
    return solve


def fit_relative_drift(blocks, weights, seed, bandwidth_hz=3.6*CHIP_RATE):
    """Refine an observation-derived two-path seed, keeping weights fixed.

    15 starts: echo drift {-3,-1.5,0,1.5,3} Hz/s, gap at fitted/.05/.2
    chips. Correct initial relative frequency to preserve midpoint frequency.
    Ten real parameters include four complex-amplitude components. The local
    uncertainty diagnostic is not a calibrated confidence interval.
    """
    solve = _problem(blocks, weights, True, bandwidth_hz)
    bounds = [(-1.,1.),(-2.,2.),(-3.,3.),(.01,1.2),(-12.,12.),(-4.,4.)]
    midpoint = np.asarray(blocks).shape[0]*.01
    seeds = [(seed.first_delay_chips,seed.common_carrier_hz,seed.common_drift_hz_s,g,
              float(np.clip(seed.relative_doppler_hz-midpoint*q,-12,12)),q)
             for q in (-3.,-1.5,0.,1.5,3.) for g in (seed.excess_chips,.05,.2)]
    fits = [minimize(solve, p, jac=True, method='L-BFGS-B', bounds=bounds,
                    options={'maxiter':400,'ftol':1e-13,'gtol':1e-9}) for p in seeds]
    best = _select_equivalent_converged(fits)
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
    return RelativeDriftFit(*map(float,best.x),std,bool(best.success),
                            reason=='interior locally conditioned fit',reason,loss)
