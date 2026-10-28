"""Blind other-PRN discovery followed by joint, per-period nuisance projection.

Point-sampled rectangular C/A model, not the aperture fractional-delay fitter.
Discovery gates are fixed heuristics; calibrate the COMPLETE residual detector.
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import numpy as np

from gnss_gpu.acquisition_cpu import CaSearch
from gnss_gpu.input_validation import positive_float, positive_int
from gnss_gpu.signal_sim_cpu import CHIP_RATE, _sample_index
from gnss_gpu.validation.tracking_residuals import generate_ca_code


@dataclass(frozen=True)
class CaCancellationConfig:
    sampling_freq: float
    target_prn: int = 1
    nuisance_prns: tuple = tuple(range(2, 13))
    carrier_grid_hz: tuple = tuple(float(x) for x in range(-2000, 2001, 250))
    discovery_periods: int = 4
    min_statistic: float = 35.
    min_phase_coherence: float = .5
    max_frequency_correction_hz: float = 125.
    max_gram_condition: float = 100.

    def __post_init__(self):
        search = CaSearch(self.sampling_freq, self.carrier_grid_hz, prn=self.target_prn)
        prns = tuple(_sample_index("nuisance_prn", value) for value in self.nuisance_prns)
        if len(set(prns)) != len(prns) or any(not 1 <= value <= 32 for value in prns) or self.target_prn in prns:
            raise ValueError("nuisance PRNs must be unique valid codes excluding target_prn")
        for name in ("min_statistic", "min_phase_coherence", "max_frequency_correction_hz", "max_gram_condition"):
            object.__setattr__(self, name, positive_float(name, getattr(self, name)))
        if self.min_phase_coherence > 1 or self.max_gram_condition < 1:
            raise ValueError("invalid coherence or condition limit")
        gap = min(np.diff(search.carrier_grid_hz)) if len(search.carrier_grid_hz) > 1 else float("inf")
        if self.max_frequency_correction_hz >= 250. or self.max_frequency_correction_hz > gap/2:
            raise ValueError("frequency correction must be below 250 Hz and within half a grid gap")
        object.__setattr__(self, "sampling_freq", search.sampling_freq)
        object.__setattr__(self, "target_prn", search.prn)
        object.__setattr__(self, "nuisance_prns", prns)
        object.__setattr__(self, "carrier_grid_hz", search.carrier_grid_hz)
        object.__setattr__(self, "discovery_periods", positive_int("discovery_periods", self.discovery_periods))


@lru_cache(maxsize=32)
def _point_reference(prn, samples_per_chip):
    local = np.repeat(generate_ca_code(prn), samples_per_chip).astype(np.float64)
    spectrum = np.conj(np.fft.fft(local))
    local.setflags(write=False); spectrum.setflags(write=False)
    return local, spectrum


@dataclass(frozen=True)
class CancelledCaComponent:
    prn: int
    coarse_carrier_hz: float
    carrier_hz: float
    delay_samples: int
    discovery_statistic: float
    phase_coherence: float
    fitted_rms_amplitude: float


@dataclass(frozen=True)
class CaCancellationResult:
    iq: np.ndarray
    components: tuple
    removed_energy_fraction: float
    gram_condition: Optional[float]
    status: str


def cancel_other_ca(iq, config: CaCancellationConfig):
    """Estimate nuisance parameters from IQ, then subtract their joint LS fit.

    Each row is one contiguous 1 ms interleaved-real period. A first, fixed
    subset discovers carrier/delay by noncoherent point-template FFT search.
    Whole-record squared projection phases refine carrier without known bit
    signs. This has a +/-250 Hz phase alias; the correction bound rejects large
    estimates but cannot rule out false aliases or wrong PRN peaks.

    A shared per-period dictionary fits one complex amplitude per selected PRN
    and row. Subtracting its orthogonal projection removes noise and possibly
    part of the target too. No target-preservation or detection guarantee follows
    from reduced residual energy. Calibration must include discovery and fit.
    No code Doppler, fractional phase, within-period bit edge or multipath fit.
    """
    count = 1023*round(config.sampling_freq/CHIP_RATE)
    raw = np.asarray(iq)
    if raw.ndim != 2 or raw.shape[0] < max(3, config.discovery_periods) or raw.shape[1] != 2*count or np.iscomplexobj(raw):
        raise ValueError("iq must contain enough real interleaved 1 ms rows")
    data = np.array(raw, dtype=np.float64, copy=True)
    if not np.all(np.isfinite(data)):
        raise ValueError("iq must be finite")
    scale = float(np.max(np.abs(data)))
    if not scale:
        return CaCancellationResult(data, (), 0., None, "zero")
    pairs = data.reshape(-1, count, 2)
    signal = pairs[:, :, 0]/scale+1j*(pairs[:, :, 1]/scale)
    periods = len(signal)
    samples = np.arange(count)
    discovery = signal[:config.discovery_periods]
    discovery_energy = float(np.sum(abs(discovery)**2))
    if not discovery_energy:
        return CaCancellationResult(data, (), 0., None, "no_discovery_energy")
    entries, columns = [], []
    for prn in config.nuisance_prns:
        spc = round(config.sampling_freq/CHIP_RATE)
        # Oversized periods bypass retained reference arrays.
        local, spectrum = (_point_reference if count <= 8192 else _point_reference.__wrapped__)(prn, spc)
        best = None
        for carrier in config.carrier_grid_hz:
            wipe = np.exp(-2j*np.pi*carrier*samples/config.sampling_freq)
            corr = np.fft.ifft(np.fft.fft(discovery*wipe, axis=1)*spectrum, axis=1)
            powers = np.sum(abs(corr)**2, axis=0)
            lag = int(np.argmax(powers))
            score = float(config.discovery_periods*powers[lag]/discovery_energy)
            if best is None or score > best[0]:
                best = score, carrier, lag
        score, carrier, lag = best
        if score < config.min_statistic:
            continue
        shifted = np.roll(local, lag)
        coarse_basis = shifted*np.exp(2j*np.pi*carrier*samples/config.sampling_freq)
        projections = signal@np.conj(coarse_basis)
        projections *= np.exp(-2j*np.pi*carrier*np.arange(periods)*.001)
        increments = (projections[1:]*np.conj(projections[:-1]))**2
        weight = float(np.sum(abs(increments)))
        if not weight:
            continue
        total = np.sum(increments)
        coherence = float(abs(total)/weight)
        correction = float(np.angle(total)/(4*np.pi*.001))
        refined = carrier+correction
        if (coherence < config.min_phase_coherence or abs(correction) > config.max_frequency_correction_hz
                or not config.carrier_grid_hz[0] <= refined <= config.carrier_grid_hz[-1]):
            continue
        columns.append(shifted*np.exp(2j*np.pi*refined*samples/config.sampling_freq))
        entries.append((prn, carrier, refined, int((lag+count//2) % count-count//2), score, coherence))
    if not columns:
        return CaCancellationResult(data, (), 0., None, "no_components")
    design = np.column_stack(columns)
    gram = design.conj().T@design
    condition = float(np.linalg.cond(gram))
    if not np.isfinite(condition) or condition > config.max_gram_condition:
        return CaCancellationResult(data, (), 0., condition, "ill_conditioned")
    amplitudes = np.linalg.solve(gram, design.conj().T@signal.T)
    residual = (signal.T-design@amplitudes).T
    energy, residual_energy = float(np.sum(abs(signal)**2)), float(np.sum(abs(residual)**2))
    if not np.all(np.isfinite(residual)) or residual_energy > energy*(1+1e-12):
        raise RuntimeError("invalid nuisance projection residual")
    output = np.empty_like(pairs)
    output[:, :, 0], output[:, :, 1] = residual.real*scale, residual.imag*scale
    fits = tuple(CancelledCaComponent(*entry, float(np.sqrt(np.mean(abs(amplitudes[i])**2)))*scale)
                 for i, entry in enumerate(entries))
    return CaCancellationResult(output.reshape(data.shape), fits,
        float(np.clip(1-residual_energy/energy, 0., 1.)), condition, "projected")
