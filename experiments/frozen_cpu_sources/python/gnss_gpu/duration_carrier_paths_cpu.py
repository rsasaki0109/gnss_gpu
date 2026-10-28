"""Variable-duration common-carrier diagnostic; each duration requires new calibration.

Single path has 4 real parameters, two paths 8 including complex amplitudes.
The direct frequency is common; echo frequency is common + relative Doppler.
No model order is selected here. Local usability does not imply detection.
"""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize
from gnss_gpu.dynamic_multipath_cpu import temporal_echo as frozen_temporal_echo
from gnss_gpu.multipath_information_cpu import ca_passband_spectrum, CHIP_RATE
from gnss_gpu.residual_dynamic_paths_cpu import _inverse_std
from gnss_gpu.noise_aware_multipath_cpu import _select_equivalent_converged


@dataclass(frozen=True)
class CarrierFit:
    first_delay_chips: float
    common_carrier_hz: float
    excess_chips: float | None
    relative_doppler_hz: float | None
    first_std_chips: float | None
    converged: bool
    locally_usable: bool
    reason: str
    loss: float


def temporal_echo(fd, count):
    """20 ms rectangular block integrals; preserve the frozen ten-block case."""
    if count == 10:
        return frozen_temporal_echo(fd)
    t = (np.arange(count)+.5)*.020
    x = fd*.020
    sinc = np.sinc(x)
    ds = (-np.pi**2*x/3+np.pi**4*x**3/30) if abs(x)<1e-4 else (np.pi*x*np.cos(np.pi*x)-np.sin(np.pi*x))/(np.pi*x*x)
    phase = np.exp(2j*np.pi*fd*t)
    return sinc*phase, phase*(.020*ds+2j*np.pi*t*sinc)


def _problem(blocks, weights, two):
    f, s = ca_passband_spectrum(3.6*CHIP_RATE)
    y = np.asarray(blocks, dtype=complex)
    if y.ndim != 2 or y.shape[0] not in (10,20,40) or y.shape[1] != len(s) or not np.all(np.isfinite(y)) or np.max(np.abs(y)) == 0:
        raise ValueError('finite nonzero 10/20/40-block full-passband observation required')
    inv = _inverse_std(weights, len(s))
    y = y*inv
    y = y/np.max(np.abs(y))
    s = s*inv
    slope = -2j*np.pi*f/CHIP_RATE
    energy = float(np.vdot(y, y).real)

    def solve(p, details=False):
        first, common = p[:2]
        c, dc = temporal_echo(common, y.shape[0])
        v = s*np.exp(slope*first)
        templates = [(c, v)]
        derivatives = [[(c, slope*v)], [(dc, v)]]
        if two:
            gap, relative = p[2:]
            e, de = temporal_echo(common+relative, y.shape[0])
            w = s*np.exp(slope*(first+gap))
            templates.append((e, w))
            derivatives[0].append((e, slope*w))
            derivatives[1].append((de, w))
            derivatives += [[None, (e, slope*w)], [None, (de, w)]]
        inner = lambda a, b: np.vdot(a[0], b[0])*np.vdot(a[1], b[1])
        corr = lambda a: np.vdot(a[0], y@a[1].conj())
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


def fit_carrier_paths(blocks, weights, *, two, single_seed=None):
    """Fit one or two paths. Common carrier bounded to +/-2 Hz, relative +/-12.

    Deterministic coarse search plus 12 local starts. Optional single-path fit
    is an observation-derived initializer, never true delay/carrier information.
    """
    solve = _problem(blocks, weights, two)
    bounds = [(-1.,1.),(-2.,2.)] if two else [(-1.,2.2),(-2.,2.)]
    if two:
        bounds += [(.01,1.2),(-12.,12.)]
        seeds = [(d,c,g,r) for d in np.linspace(-1,1,9) for c in (-1.5,0.,1.5)
                 for g in (.05,.2,.5,1.) for r in (-10.,-5.,-1.,0.,1.,5.,10.)]
        if single_seed is not None:
            d = float(np.clip(single_seed.first_delay_chips,-1,1))
            seeds += [(d,single_seed.common_carrier_hz,g,r) for g in (.05,.2,.5,1.)
                      for r in (-5.,-1.,0.,1.,5.)]
    else:
        seeds = [(d,c) for d in np.linspace(-1,2.2,49) for c in (-1.5,0.,1.5)]
    seeds = sorted(seeds, key=lambda p:solve(p)[0])[:12]
    fits = [minimize(solve, seed, jac=True, method='L-BFGS-B', bounds=bounds,
                    options={'maxiter':250,'ftol':1e-13,'gtol':1e-9}) for seed in seeds]
    best = _select_equivalent_converged(fits)
    loss, a, jac = solve(best.x, True)
    projected = jac-a@np.linalg.lstsq(a,jac,rcond=None)[0]
    info = np.real(projected.conj().T@projected)
    eig = np.linalg.eigvalsh(info)
    std = None
    if eig[0] > max(1e-20, 1e-12*eig[-1]):
        std = float(np.sqrt(max(0,loss/(2*len(a)-(8 if two else 4))*np.linalg.inv(info)[0,0])))
    if not best.success:
        reason = 'optimizer did not converge'
    elif min(min(x-lo,hi-x) for x,(lo,hi) in zip(best.x,bounds)) < 1e-4:
        reason = 'search boundary'
    elif std is None or std > .1:
        reason = 'weak local delay information'
    else:
        reason = 'interior locally conditioned fit'
    d,c = map(float,best.x[:2])
    g,r = map(float,best.x[2:]) if two else (None,None)
    return CarrierFit(d,c,g,r,std,bool(best.success),reason=='interior locally conditioned fit',reason,loss)


def estimate_weights(blocks, preliminary):
    """Use both fitted carriers in residual prediction; one refit only."""
    p = preliminary
    _, a, _ = _problem(blocks, np.ones(np.asarray(blocks).shape[1]), True)(
        (p.first_delay_chips,p.common_carrier_hz,p.excess_chips,p.relative_doppler_hz), True)
    y = np.asarray(blocks).ravel()
    residual = (y-a@np.linalg.lstsq(a,y,rcond=None)[0]).reshape(np.asarray(blocks).shape)
    variance = np.mean(np.abs(residual)**2,axis=0)
    kernel = np.ones(101)
    smooth = np.convolve(variance,kernel,mode='same')/np.convolve(np.ones_like(variance),kernel,mode='same')
    mean = float(smooth.mean())
    if not np.isfinite(mean) or mean <= 0:
        raise ValueError('positive residual noise energy required')
    weights = np.maximum(smooth, mean*1e-6)
    return weights/weights.mean()


def score_common_carrier_paths(blocks):
    """Uncalibrated ranking score, not a detector or posterior probability."""
    ones = np.ones(np.asarray(blocks).shape[-1])
    initial_single = fit_carrier_paths(blocks,ones,two=False)
    preliminary = fit_carrier_paths(blocks,ones,two=True,single_seed=initial_single)
    weights = estimate_weights(blocks,preliminary)
    single = fit_carrier_paths(blocks,weights,two=False)
    two = fit_carrier_paths(blocks,weights,two=True,single_seed=single)
    n = 2*np.asarray(blocks).size
    tiny = np.finfo(float).tiny
    score = float(n*np.log(max(single.loss,tiny)/max(two.loss,tiny))-4*np.log(n))
    return dict(single=single,two=two,preliminary=preliminary,score=score),weights
