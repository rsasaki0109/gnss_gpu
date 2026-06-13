from __future__ import annotations

from typing import Any

import numpy as np


C_LIGHT = 299792458.0
CA_CHIP_RATE = 1.023e6
CA_CODE_LENGTH = 1023
CA_CHIP_M = C_LIGHT / CA_CHIP_RATE


def _maybe_scalar(value: Any):
    arr = np.asarray(value)
    if arr.ndim == 0:
        return arr.item()
    return value


def wrap_code_phase_chips(delta_chips):
    """Wrap code phase difference to [-511.5, +511.5) chips."""
    delta = np.asarray(delta_chips, dtype=float)
    wrapped = (delta + CA_CODE_LENGTH / 2.0) % float(CA_CODE_LENGTH) - CA_CODE_LENGTH / 2.0
    return _maybe_scalar(wrapped)


def code_phase_to_residual_m(tracked_chips, reference_chips):
    """Convert tracked-reference code phase difference to pseudorange residual in meters."""
    residual_chips = wrap_code_phase_chips(np.asarray(tracked_chips, dtype=float) - reference_chips)
    residual_m = np.asarray(residual_chips, dtype=float) * CA_CHIP_M
    return _maybe_scalar(residual_m)


def eml_discriminator(
    early_i,
    early_q,
    prompt_i,
    prompt_q,
    late_i,
    late_q,
    spacing=0.5,
    normalize=True,
):
    """Non-coherent Early-minus-Late power discriminator in chips."""
    del prompt_i, prompt_q, spacing

    early_power = np.hypot(np.asarray(early_i, dtype=float), np.asarray(early_q, dtype=float))
    late_power = np.hypot(np.asarray(late_i, dtype=float), np.asarray(late_q, dtype=float))

    if normalize:
        denom = early_power + late_power
        safe_denom = np.where(denom != 0.0, denom, 1.0)
        disc = np.where(denom != 0.0, 0.5 * (early_power - late_power) / safe_denom, 0.0)
    else:
        disc = 0.5 * (early_power - late_power)

    return _maybe_scalar(disc)


def correlations_to_discriminator(correlations, spacing=0.5, normalize=True) -> np.ndarray:
    """Convert [n_ch, 6] E/P/L correlations to EML discriminator values."""
    corr = np.asarray(correlations, dtype=float).reshape(-1, 6)
    return np.asarray(
        eml_discriminator(
            corr[:, 0],
            corr[:, 1],
            corr[:, 2],
            corr[:, 3],
            corr[:, 4],
            corr[:, 5],
            spacing=spacing,
            normalize=normalize,
        ),
        dtype=float,
    )


def cn0_from_prompt(prompt_i, prompt_q, noise_power, integration_time):
    """Simple prompt-power C/N0 estimate in dB-Hz."""
    prompt_power = np.asarray(prompt_i, dtype=float) ** 2 + np.asarray(prompt_q, dtype=float) ** 2
    cn0_linear = (prompt_power / np.asarray(noise_power, dtype=float)) / np.asarray(
        integration_time, dtype=float
    )
    cn0_db = 10.0 * np.log10(np.maximum(cn0_linear, np.finfo(float).tiny))
    return _maybe_scalar(cn0_db)


def settled_value(history, settle_fraction=0.5) -> float:
    """Mean of the final settle_fraction portion of a 1-D history."""
    if not (0.0 < settle_fraction <= 1.0):
        raise ValueError("settle_fraction must be in (0, 1].")

    arr = np.asarray(history, dtype=float).reshape(-1)
    n = arr.size
    if n == 0:
        raise ValueError("history must not be empty.")
    if n == 1:
        return float(arr[0])

    count = max(1, int(round(n * settle_fraction)))
    return float(np.mean(arr[-count:]))


# --- GPS C/A code synthesis (pure numpy, matches acquisition.cu LFSR) ---

_G2_TAPS = [
    (2, 6), (3, 7), (4, 8), (5, 9), (1, 9), (2, 10), (1, 8), (2, 9),
    (3, 10), (2, 3), (3, 4), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10),
    (1, 4), (2, 5), (3, 6), (4, 7), (5, 8), (6, 9), (1, 3), (4, 6),
    (5, 7), (6, 8), (7, 9), (8, 10), (1, 6), (2, 7), (3, 8), (4, 9),
]

_CA_CODE_CACHE: dict = {}


def generate_ca_code(prn: int) -> np.ndarray:
    """Return the 1023-chip GPS L1 C/A Gold code for ``prn`` as +/-1 floats."""
    prn = int(prn)
    if prn < 1 or prn > 32:
        raise ValueError("prn must be in 1..32")
    if prn in _CA_CODE_CACHE:
        return _CA_CODE_CACHE[prn]

    g1 = [1] * 10
    g2 = [1] * 10
    tap1, tap2 = _G2_TAPS[prn - 1]
    tap1 -= 1
    tap2 -= 1

    code = np.zeros(CA_CODE_LENGTH, dtype=np.float64)
    for i in range(CA_CODE_LENGTH):
        g1_out = g1[9]
        g2_delayed = g2[tap1] ^ g2[tap2]
        ca_bit = g1_out ^ g2_delayed
        code[i] = 2 * ca_bit - 1

        g1_fb = g1[2] ^ g1[9]
        g2_fb = g2[1] ^ g2[2] ^ g2[5] ^ g2[7] ^ g2[8] ^ g2[9]
        for j in range(9, 0, -1):
            g1[j] = g1[j - 1]
            g2[j] = g2[j - 1]
        g1[0] = g1_fb
        g2[0] = g2_fb

    _CA_CODE_CACHE[prn] = code
    return code


def generate_ca_signal(
    prn,
    code_phase,
    carrier_freq,
    n_samples,
    sampling_freq,
    intermediate_freq,
    amplitude=1.0,
    noise_std=0.0,
    rng=None,
):
    """Synthesise a real IF block of a single C/A path (matches the tracker)."""
    code = generate_ca_code(prn)
    n = int(n_samples)
    t = np.arange(n, dtype=np.float64) / float(sampling_freq)
    code_chips = np.mod(
        float(code_phase) + CA_CHIP_RATE * t, float(CA_CODE_LENGTH)
    ).astype(np.int64) % CA_CODE_LENGTH
    code_samples = code[code_chips]
    carrier = np.cos(2.0 * np.pi * float(carrier_freq) * t)
    signal = float(amplitude) * code_samples * carrier
    if noise_std and noise_std > 0.0:
        gen = rng if rng is not None else np.random.default_rng(0)
        signal = signal + gen.normal(0.0, float(noise_std), size=n)
    return signal.astype(np.float64)


def generate_multipath_signal(
    paths,
    prn,
    n_samples,
    sampling_freq,
    intermediate_freq,
    noise_std=0.0,
    rng=None,
):
    """Sum of C/A paths into one real IF block.

    Each path is a mapping with keys ``code_phase`` [chips], ``carrier_freq``
    [Hz] (default = intermediate_freq) and ``amplitude`` (default 1.0).
    """
    n = int(n_samples)
    signal = np.zeros(n, dtype=np.float64)
    for path in paths:
        signal = signal + generate_ca_signal(
            prn,
            path["code_phase"],
            path.get("carrier_freq", intermediate_freq),
            n,
            sampling_freq,
            intermediate_freq,
            amplitude=path.get("amplitude", 1.0),
            noise_std=0.0,
        )
    if noise_std and noise_std > 0.0:
        gen = rng if rng is not None else np.random.default_rng(0)
        signal = signal + gen.normal(0.0, float(noise_std), size=n)
    return signal.astype(np.float64)


def discriminator_zero_crossing(offsets, disc_values):
    """Return the code offset where a falling DLL S-curve crosses zero.

    ``offsets`` and ``disc_values`` are paired 1-D arrays describing the EML
    discriminator as a function of local code offset (chips). The stable DLL
    lock point is the zero crossing with a negative slope nearest to zero
    offset (positive discriminator on the early side, negative on the late
    side). Returns ``None`` if no falling crossing exists.
    """
    x = np.asarray(offsets, dtype=float).reshape(-1)
    y = np.asarray(disc_values, dtype=float).reshape(-1)
    if x.size != y.size or x.size < 2:
        raise ValueError("offsets and disc_values must be equal-length (>=2)")

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    sign = np.sign(y)
    falling = np.where((sign[:-1] > 0) & (sign[1:] <= 0))[0]
    if falling.size == 0:
        return None

    # Crossing nearest to zero local offset (the on-time lock point).
    mid = 0.5 * (x[falling] + x[falling + 1])
    pick = falling[int(np.argmin(np.abs(mid)))]

    x0, x1 = x[pick], x[pick + 1]
    y0, y1 = y[pick], y[pick + 1]
    if y1 == y0:
        return float(0.5 * (x0 + x1))
    return float(x0 - y0 * (x1 - x0) / (y1 - y0))


def measure_multipath_bias(
    paths,
    prn,
    *,
    config=None,
    doppler_hz=0.0,
    ref_code_phase=None,
    sweep_chips=(-0.5, 1.5),
    n_points=121,
    correlator_spacing=None,
):
    """Measure the steady-state DLL code bias from a composite signal (GPU).

    ``paths[0]`` is the direct (reference) path. A composite IF block is
    synthesised, then the GPU correlator evaluates the non-coherent EML
    discriminator S-curve over a sweep of local code offsets. The zero crossing
    of that S-curve is the code tracking point the DLL would settle on; its
    offset from the direct path is the multipath-induced pseudorange residual.

    This open-loop S-curve method is deterministic and avoids the divergence of
    re-feeding a static block through the closed loop. Requires the compiled GPU
    tracking extension.

    Returns a dict with ``bias_chips``, ``residual_m`` (and ``offsets`` /
    ``disc`` arrays for inspection / plotting). ``bias_chips`` is ``None`` when
    the S-curve has no on-time zero crossing.
    """
    from gnss_gpu.tracking import TrackingConfig
    from gnss_gpu._gnss_gpu_tracking import (
        ChannelState as _NativeChannelState,
        TrackingConfig as _NativeTrackingConfig,
        batch_correlate as _batch_correlate,
    )

    if not paths:
        raise ValueError("paths must contain at least the direct path")

    if config is None:
        config = TrackingConfig(
            sampling_freq=4.092e6,
            intermediate_freq=4.092e6,
            integration_time=1e-3,
            dll_bandwidth=2.0,
            pll_bandwidth=15.0,
            correlator_spacing=0.5 if correlator_spacing is None else correlator_spacing,
        )

    spacing = config.correlator_spacing if correlator_spacing is None else correlator_spacing
    ncfg = _NativeTrackingConfig(
        config.sampling_freq, config.intermediate_freq, config.integration_time,
        config.dll_bandwidth, config.pll_bandwidth, spacing,
    )

    direct = paths[0]
    if ref_code_phase is None:
        ref_code_phase = float(direct["code_phase"])
    carrier_freq = config.intermediate_freq + float(doppler_hz)
    enriched = [
        {
            "code_phase": float(p["code_phase"]),
            "carrier_freq": p.get("carrier_freq", carrier_freq),
            "amplitude": float(p.get("amplitude", 1.0)),
        }
        for p in paths
    ]

    n_samples = int(round(config.sampling_freq * config.integration_time))
    signal = generate_multipath_signal(
        enriched, prn, n_samples,
        config.sampling_freq, config.intermediate_freq, noise_std=0.0,
    ).astype(np.float32)

    offsets = np.linspace(float(sweep_chips[0]), float(sweep_chips[1]), int(n_points))
    disc = np.empty(offsets.size, dtype=float)
    for k, off in enumerate(offsets):
        ch = _NativeChannelState()
        ch.prn = int(prn)
        ch.code_phase = ref_code_phase + float(off)
        ch.code_freq = CA_CHIP_RATE
        ch.carrier_phase = 0.0
        ch.carrier_freq = carrier_freq
        ch.cn0 = 45.0
        ch.locked = True
        corr = np.asarray(
            _batch_correlate(signal, [ch], 1, n_samples, ncfg), dtype=float
        ).reshape(6)
        disc[k] = float(eml_discriminator(
            corr[0], corr[1], corr[2], corr[3], corr[4], corr[5], spacing=spacing))

    bias_chips = discriminator_zero_crossing(offsets, disc)
    residual_m = None if bias_chips is None else float(bias_chips * CA_CHIP_M)
    return {
        "bias_chips": bias_chips,
        "residual_m": residual_m,
        "offsets": offsets,
        "disc": disc,
        "ref_code_phase": float(ref_code_phase),
    }


def convergence_residual(code_phase_history, reference_chips, settle_fraction=0.5) -> dict:
    """Estimate settled tracking residual against a reference code phase."""
    steady = settled_value(code_phase_history, settle_fraction=settle_fraction)
    residual_chips = float(wrap_code_phase_chips(steady - reference_chips))
    residual_m = float(code_phase_to_residual_m(steady, reference_chips))

    return {
        "steady_code_phase": float(steady),
        "residual_m": residual_m,
        "residual_chips": residual_chips,
    }


__all__ = [
    "C_LIGHT",
    "CA_CHIP_RATE",
    "CA_CODE_LENGTH",
    "CA_CHIP_M",
    "wrap_code_phase_chips",
    "code_phase_to_residual_m",
    "eml_discriminator",
    "correlations_to_discriminator",
    "cn0_from_prompt",
    "settled_value",
    "convergence_residual",
    "generate_ca_code",
    "generate_ca_signal",
    "generate_multipath_signal",
    "discriminator_zero_crossing",
    "measure_multipath_bias",
]
