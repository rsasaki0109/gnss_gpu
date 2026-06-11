"""Quantitative benchmark: which diffraction model reproduces real residuals?

This isolates the *diffraction amplitude model* (knife-edge ITU-R P.526 vs the
Kouyoumjian-Pathak UTD) and asks which one yields a simulated pseudorange-bias
distribution closer to a measured one (e.g. real UrbanNav residuals), using the
Wasserstein-1 and Kolmogorov-Smirnov distances.

The geometry (excess delay, which edge, which satellite) is identical between
the two models; only the diffracted-replica amplitude differs. So a difference
in the resulting bias distribution is attributable purely to the diffraction
amplitude model -- the quantity Zhang & Hsu (NAVIGATION 2021) showed UTD models
more accurately than the knife edge.

The DLL code bias for a single delayed replica is predicted analytically with
the standard ideal C/A triangular autocorrelation and a non-coherent
early-minus-late discriminator, swept open-loop for its zero crossing (the same
principle as the GPU ``measure_multipath_bias`` but vectorisable on the CPU).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gnss_gpu.validation.residuals import ks_statistic, wasserstein1
from gnss_gpu.validation.tracking_residuals import (
    CA_CHIP_M,
    CA_CHIP_RATE,
    C_LIGHT,
    discriminator_zero_crossing,
)


def triangle_acf(x) -> np.ndarray:
    """Ideal C/A code autocorrelation R(x) = 1 - |x| for |x|<=1, else 0."""
    ax = np.abs(np.asarray(x, dtype=float))
    return np.where(ax <= 1.0, 1.0 - ax, 0.0)


def code_multipath_bias_chips(
    alpha: float,
    tau_chips: float,
    theta: float,
    *,
    spacing: float = 0.5,
    sweep_chips: tuple[float, float] = (-0.75, 1.0),
    n_points: int = 351,
) -> float | None:
    """Steady-state non-coherent EML DLL code bias from one delayed replica.

    ``alpha`` is the replica/direct amplitude ratio, ``tau_chips`` its excess
    delay in chips, ``theta`` the carrier phase difference (rad). Returns the
    DLL lock offset in chips (the multipath-induced code bias), or None if the
    S-curve has no on-time zero crossing.
    """
    offsets = np.linspace(float(sweep_chips[0]), float(sweep_chips[1]), int(n_points))
    half = 0.5 * float(spacing)
    ct = float(alpha) * np.cos(float(theta))
    st = float(alpha) * np.sin(float(theta))
    tau = float(tau_chips)

    def corr_power(off):
        ri = triangle_acf(off) + ct * triangle_acf(off - tau)
        rq = st * triangle_acf(off - tau)
        return ri * ri + rq * rq

    # Early-minus-late power, signed so the stable lock is a falling zero
    # crossing (positive discriminator on the early side, off < 0).
    disc = corr_power(offsets + half) - corr_power(offsets - half)
    return discriminator_zero_crossing(offsets, disc)


@dataclass
class DiffractionCandidate:
    """One diffraction multipath component relative to the direct path."""

    amplitude_ratio: float   # diffracted amplitude / direct amplitude
    excess_delay_m: float     # extra path length over the direct ray (m)


def predict_bias_samples_m(
    candidates,
    *,
    n_phase: int = 16,
    spacing: float = 0.5,
    max_alpha: float = 0.99,
    **bias_kwargs,
) -> np.ndarray:
    """Predict a distribution of code-bias samples (meters) from candidates.

    Each candidate contributes ``n_phase`` samples by sweeping the (unknown)
    carrier phase difference uniformly over [0, 2*pi); this traces the classic
    multipath error envelope, whose spread is the per-candidate bias
    distribution. Candidates with no DLL zero crossing are skipped.
    """
    thetas = np.linspace(0.0, 2.0 * np.pi, int(n_phase), endpoint=False)
    out: list[float] = []
    for cand in candidates:
        alpha = float(min(float(cand.amplitude_ratio), float(max_alpha)))
        if alpha <= 0.0:
            continue
        tau_chips = float(cand.excess_delay_m) / CA_CHIP_M
        for theta in thetas:
            bias = code_multipath_bias_chips(
                alpha, tau_chips, float(theta), spacing=spacing, **bias_kwargs)
            if bias is not None:
                out.append(bias * CA_CHIP_M)
    return np.asarray(out, dtype=float)


def benchmark_models(real_values, sim_by_model: dict) -> dict:
    """Compare each model's simulated bias distribution to the real one.

    Returns per-model Wasserstein-1 / KS distances plus the winning model name
    under each metric (smaller distance == closer to the real distribution).
    """
    real = np.asarray(real_values, dtype=float)
    real = real[np.isfinite(real)]
    per_model: dict[str, dict] = {}
    for name, sim in sim_by_model.items():
        sim_arr = np.asarray(sim, dtype=float)
        sim_arr = sim_arr[np.isfinite(sim_arr)]
        per_model[name] = {
            "n_sim": int(sim_arr.size),
            "wasserstein": wasserstein1(sim_arr, real),
            "ks": ks_statistic(sim_arr, real),
            "mean_bias_m": float(np.mean(sim_arr)) if sim_arr.size else float("nan"),
        }

    def _best(metric):
        valid = {k: v[metric] for k, v in per_model.items() if np.isfinite(v[metric])}
        return min(valid, key=valid.get) if valid else None

    return {
        "n_real": int(real.size),
        "real_mean_bias_m": float(np.mean(real)) if real.size else float("nan"),
        "models": per_model,
        "best_wasserstein": _best("wasserstein"),
        "best_ks": _best("ks"),
    }


def candidates_from_paths(paths, direct_amplitude: float = 1.0):
    """Build DiffractionCandidate list from diffraction path objects.

    Accepts knife-edge ``DiffractionPath`` or ``UTDDiffractionPath`` objects
    (both expose ``amplitude`` and ``excess_delay``).
    """
    direct = float(direct_amplitude) if direct_amplitude else 1.0
    out = []
    for p in paths:
        out.append(DiffractionCandidate(
            amplitude_ratio=float(p.amplitude) / direct,
            excess_delay_m=float(p.excess_delay)))
    return out


__all__ = [
    "triangle_acf",
    "code_multipath_bias_chips",
    "DiffractionCandidate",
    "predict_bias_samples_m",
    "benchmark_models",
    "candidates_from_paths",
    "CA_CHIP_M",
    "CA_CHIP_RATE",
    "C_LIGHT",
]
