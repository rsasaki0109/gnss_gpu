"""Multiple coarse-carrier hypotheses for blind other-PRN cancellation.

Experimental alternative to the frozen single-seed model. Recalibrate the whole
residual detector before applying any gate; old gates do not transfer.
"""

import numpy as np

from gnss_gpu.ca_cancellation_cpu import (
    CaCancellationConfig, CaCancellationResult, CancelledCaComponent,
    _point_reference, cancel_other_ca,
)
from gnss_gpu.input_validation import positive_int
from gnss_gpu.signal_sim_cpu import CHIP_RATE


def cancel_other_ca_multiseed(iq, config: CaCancellationConfig, *, coarse_candidates=3):
    """Refine several coarse seeds per PRN and retain one by full-record energy.

    Retain the best delay for each coarse carrier, shortlist by discovery score,
    then apply the same phase/coherence/refinement-cell gates to every seed.
    Among surviving seeds, select the largest whole-record projection energy.
    Ties retain discovery-score order, then ascending coarse carrier. Only one
    carrier/delay column per PRN enters the joint per-period amplitude fit.

    No true signal parameters enter selection. Navigation sign reversals between
    rows are tolerated by squared phase increments; within-row edges and phase
    aliasing remain model limitations. Projection can remove target/noise too.
    The increased search must be included in fresh full-pipeline calibration.
    coarse_candidates=1 calls the historical algorithm exactly.
    """
    coarse_candidates = positive_int("coarse_candidates", coarse_candidates)
    if coarse_candidates == 1:
        return cancel_other_ca(iq, config)
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
        coarse = []
        for carrier in config.carrier_grid_hz:
            wipe = np.exp(-2j*np.pi*carrier*samples/config.sampling_freq)
            corr = np.fft.ifft(np.fft.fft(discovery*wipe, axis=1)*spectrum, axis=1)
            powers = np.sum(abs(corr)**2, axis=0)
            lag = int(np.argmax(powers))
            score = float(config.discovery_periods*powers[lag]/discovery_energy)
            coarse.append((score, carrier, lag))
        coarse.sort(key=lambda item: (-item[0], item[1]))
        best = None
        for score, carrier, lag in coarse[:coarse_candidates]:
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
            column = shifted*np.exp(2j*np.pi*refined*samples/config.sampling_freq)
            projected_energy = float(np.sum(abs(signal@np.conj(column))**2)/count)
            if best is None or projected_energy > best[0]:
                entry = (prn, carrier, refined, int((lag+count//2) % count-count//2), score, coherence)
                best = projected_energy, column, entry
        if best is not None:
            columns.append(best[1])
            entries.append(best[2])
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
