"""Local delay-subspace refinement of blindly discovered other-PRN signals."""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from gnss_gpu.ca_cancellation_cpu import cancel_other_ca, _point_reference
from gnss_gpu.signal_sim_cpu import CHIP_RATE, _sample_index


@dataclass(frozen=True)
class CaLocalDelayResult:
    iq: np.ndarray
    discovery_components: tuple
    removed_energy_fraction: float
    original_removed_energy_fraction: float
    gram_condition: Optional[float]
    delay_radius_samples: int
    status: str

    @property
    def components(self):
        """Original blind discoveries, not the refined subspace coefficients."""
        return self.discovery_components


def _refine_local_delays(iq, config, discovery, delay_radius_samples=1):
    radius = _sample_index("delay_radius_samples", delay_radius_samples)
    if radius > 2:
        raise ValueError("delay radius must be at most two samples")
    if not discovery.components:
        return CaLocalDelayResult(
            discovery.iq,
            (),
            discovery.removed_energy_fraction,
            discovery.removed_energy_fraction,
            discovery.gram_condition,
            radius,
            "no_components",
        )
    raw = np.asarray(iq, dtype=float)
    periods, count = raw.shape[0], raw.shape[1] // 2
    pairs = raw.reshape(periods, count, 2)
    scale = float(np.max(abs(pairs)))
    signal = pairs[:, :, 0] / scale + 1j * (pairs[:, :, 1] / scale)
    samples = np.arange(count)
    columns = []
    for fit in discovery.components:
        spc = round(config.sampling_freq / CHIP_RATE)
        local, _ = (
            _point_reference if count <= 8192 else _point_reference.__wrapped__
        )(fit.prn, spc)
        carrier = np.exp(2j * np.pi * fit.carrier_hz * samples / config.sampling_freq)
        for offset in range(-radius, radius + 1):
            columns.append(np.roll(local, fit.delay_samples + offset) * carrier)
    design = np.column_stack(columns)
    gram = design.conj().T @ design
    condition = float(np.linalg.cond(gram))
    if not np.isfinite(condition) or condition > config.max_gram_condition:
        return CaLocalDelayResult(
            discovery.iq,
            discovery.components,
            discovery.removed_energy_fraction,
            discovery.removed_energy_fraction,
            condition,
            radius,
            "fallback_ill_conditioned",
        )
    amplitudes = np.linalg.solve(gram, design.conj().T @ signal.T)
    residual = (signal.T - design @ amplitudes).T
    energy = float(np.sum(abs(signal) ** 2))
    remaining = float(np.sum(abs(residual) ** 2))
    if not np.all(np.isfinite(residual)) or remaining > energy * (1 + 1e-12):
        raise RuntimeError("invalid local-delay projection")
    output = np.empty_like(pairs)
    output[:, :, 0], output[:, :, 1] = residual.real * scale, residual.imag * scale
    return CaLocalDelayResult(
        output.reshape(raw.shape),
        discovery.components,
        float(np.clip(1 - remaining / energy, 0.0, 1.0)),
        discovery.removed_energy_fraction,
        condition,
        radius,
        "local_delay_projected",
    )


def cancel_other_ca_local_delay(iq, config, *, delay_radius_samples=1):
    """Blindly discover nuisances, then jointly fit nearby delay columns per row.

    Adjacent integer-delay columns allow their mixture to change between rows,
    covering small delay drift and a one-sample epoch slip without true clock
    parameters. This is not a physical multipath decomposition. Extra columns
    also project out more noise/target energy; calibrate the complete pipeline.
    """
    radius = _sample_index("delay_radius_samples", delay_radius_samples)
    if radius > 2:
        raise ValueError("delay radius must be at most two samples")
    discovery = cancel_other_ca(iq, config)
    return _refine_local_delays(iq, config, discovery, radius)
