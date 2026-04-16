"""Helpers for E2E signal-sim/acquisition experiments."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

C_LIGHT = 299792458.0
CA_CHIP_RATE = 1.023e6
GPS_CA_CODE_LENGTH = 1023.0
GPS_CA_PERIOD_S = 1e-3
GPS_CA_PERIOD_M = C_LIGHT * GPS_CA_PERIOD_S
GPS_L1_FREQ = 1575.42e6
DEFAULT_CODE_LOCK_MAX_ERROR_M = 2000.0

__all__ = [
    "C_LIGHT",
    "CA_CHIP_RATE",
    "GPS_CA_CODE_LENGTH",
    "GPS_CA_PERIOD_S",
    "GPS_CA_PERIOD_M",
    "GPS_L1_FREQ",
    "DEFAULT_CODE_LOCK_MAX_ERROR_M",
    "compute_e2e_wls_weights",
    "acquisition_lag_to_code_phase_chips",
    "code_phase_chips_to_acquisition_lag",
    "refine_acquisition_code_lag_dll",
    "refine_acquisition_code_lags_dll_batch",
    "refine_acquisition_code_lags_diagnostic_batch",
    "dump_e2e_diagnostics_csv",
    "pseudorange_to_code_phase_chips",
    "acquisition_code_phase_to_pseudorange",
]

_DIAGNOSTIC_ARRAY_KEYS = [
    "lag_samples",
    "code_phase_chips",
    "carrier_phase_cycles",
    "code_freq_hz",
    "carrier_freq_hz",
    "E_I",
    "E_Q",
    "P_I",
    "P_Q",
    "L_I",
    "L_Q",
    "prompt_power",
    "dll_abs",
    "cn0_est_db",
]

_DIAGNOSTIC_CSV_COLUMNS = [
    "prn",
    "lag_samples",
    "code_phase_chips",
    "carrier_phase_cycles",
    "code_freq_hz",
    "carrier_freq_hz",
    "E_I",
    "E_Q",
    "P_I",
    "P_Q",
    "L_I",
    "L_Q",
    "prompt_power",
    "dll_abs",
    "cn0_est_db",
]


def _zero_diagnostics(prns, code_phase_lags, n_iter):
    prn_arr = np.asarray(list(prns), dtype=np.int32).ravel()
    n_ch = int(prn_arr.size)
    diag = {
        key: np.zeros(n_ch, dtype=np.float64)
        for key in _DIAGNOSTIC_ARRAY_KEYS
    }
    lags = np.asarray(code_phase_lags, dtype=np.float64).ravel()
    if lags.size == n_ch:
        diag["lag_samples"] = lags.copy()
    elif n_ch:
        n_copy = min(n_ch, int(lags.size))
        diag["lag_samples"][:n_copy] = lags[:n_copy]
    diag["prn"] = prn_arr
    diag["n_iter_used"] = int(n_iter)
    return diag


def compute_e2e_wls_weights(prompt_power, snr_acquisition_ratio, dll_abs):
    """Per-satellite WLS weights from lock quality (mean-normalized precision).

    ``wls_position`` uses weights as diagonal precision (larger = more trusted).

    - *prompt_power*: |P_I|^2+|P_Q|^2 after DLL refinement
    - *snr_acquisition_ratio*: acquisition peak / second-peak (from search)
    - *dll_abs*: |E-L| / (E+L) power discriminator magnitude in [0, 1]
    """
    pp = np.maximum(np.asarray(prompt_power, dtype=np.float64).ravel(), 1e-30)
    snr = np.maximum(np.asarray(snr_acquisition_ratio, dtype=np.float64).ravel(), 1e-6)
    d = np.clip(np.asarray(dll_abs, dtype=np.float64).ravel(), 0.0, 1.0)
    if pp.size != snr.size or snr.size != d.size:
        raise ValueError("prompt_power, snr_acquisition_ratio, and dll_abs must match")
    lock = 1.0 / (0.02 + d * d)
    w = pp * snr * lock
    m = float(np.mean(w))
    if m > 0.0:
        w /= m
    return w


def _maybe_scalar(value):
    arr = np.asarray(value)
    if arr.ndim == 0:
        return float(arr)
    return arr


def acquisition_lag_to_code_phase_chips(code_phase_lag, sampling_freq):
    """Map acquisition circular lag (samples) to C/A code phase at t=0 [chips]."""
    fs = float(sampling_freq)
    n_samples = int(round(fs * GPS_CA_PERIOD_S))
    lag = float(code_phase_lag)
    delay_samples = float(np.mod(n_samples - lag, n_samples))
    return (delay_samples / fs) * CA_CHIP_RATE


def code_phase_chips_to_acquisition_lag(code_phase_chips, sampling_freq):
    """Map code phase [chips] back to acquisition lag (samples), circular."""
    fs = float(sampling_freq)
    n_samples = int(round(fs * GPS_CA_PERIOD_S))
    delay_samples = (float(code_phase_chips) / CA_CHIP_RATE) * fs
    delay_samples = float(np.mod(delay_samples, n_samples))
    return float(np.mod(n_samples - delay_samples, n_samples))


def refine_acquisition_code_lag_dll(
    signal_i,
    prn,
    code_phase_lag,
    doppler_hz,
    sampling_freq,
    intermediate_freq=0.0,
    n_iter=15,
    dll_gain=0.22,
    pll_gain=0.18,
    correlator_spacing=0.5,
):
    """Refine acquisition lag with GPU E/P/L correlations + DLL/PLL on one epoch.

    Re-correlates the same 1 ms IF buffer without advancing time: uses
    ``batch_correlate`` only. DLL nudges ``code_phase`` (chips); PLL nudges
    ``carrier_phase`` (cycles) via ``atan2(PQ, PI)`` (same discriminator as
    ``tracking.cu``).

    Returns the refined lag in sample units, or the input lag if bindings
    are unavailable or the prompt correlation is near zero.
    """
    out = refine_acquisition_code_lags_dll_batch(
        signal_i,
        [prn],
        [code_phase_lag],
        [doppler_hz],
        sampling_freq,
        intermediate_freq=intermediate_freq,
        n_iter=n_iter,
        dll_gain=dll_gain,
        pll_gain=pll_gain,
        correlator_spacing=correlator_spacing,
    )
    return float(out[0]) if out.size else float(code_phase_lag)


def refine_acquisition_code_lags_dll_batch(
    signal_i,
    prns,
    code_phase_lags,
    doppler_hz_list,
    sampling_freq,
    intermediate_freq=0.0,
    n_iter=15,
    dll_gain=0.22,
    pll_gain=0.18,
    correlator_spacing=0.5,
    return_lock_metrics=False,
):
    """DLL + optional PLL refinement for multiple PRNs on one IF buffer.

    One ``batch_correlate`` per iteration. DLL updates code phase; if
    ``pll_gain > 0``, PLL updates carrier phase (cycles) with ``atan2(PQ, PI)``
    each iteration. Time is not advanced (same 1 ms buffer).

    If ``return_lock_metrics`` is True, also returns prompt power and |DLL| for
    each channel (after one final correlate at the refined code/carrier phase).

    On import failure, returns the input lags unchanged (and dummy metrics if
    requested).
    """
    try:
        from gnss_gpu._gnss_gpu_tracking import (
            TrackingConfig as _TrackingConfig,
            ChannelState as _ChannelState,
            batch_correlate as _batch_correlate,
        )
    except ImportError:
        lag = np.asarray(code_phase_lags, dtype=np.float64)
        if return_lock_metrics:
            n = int(lag.size)
            return lag, np.ones(n, dtype=np.float64), np.zeros(n, dtype=np.float64)
        return lag

    prns = list(prns)
    n_ch = len(prns)
    if n_ch == 0:
        empty = np.asarray([], dtype=np.float64)
        if return_lock_metrics:
            return empty, empty, empty
        return empty

    lags_in = np.asarray(code_phase_lags, dtype=np.float64).ravel()
    dops = np.asarray(doppler_hz_list, dtype=np.float64).ravel()
    if lags_in.size != n_ch or dops.size != n_ch:
        raise ValueError("prns, code_phase_lags, and doppler_hz_list must match in length")

    sig = np.asarray(signal_i, dtype=np.float32).ravel()
    n_samples = int(sig.shape[0])
    fs = float(sampling_freq)
    if n_iter <= 0:
        out = lags_in.copy()
        if return_lock_metrics:
            return out, np.ones(n_ch, dtype=np.float64), np.zeros(n_ch, dtype=np.float64)
        return out

    cfg = _TrackingConfig(
        sampling_freq=fs,
        intermediate_freq=float(intermediate_freq),
        integration_time=GPS_CA_PERIOD_S,
        dll_bandwidth=2.0,
        pll_bandwidth=15.0,
        correlator_spacing=float(correlator_spacing),
    )

    ch_list = []
    for k in range(n_ch):
        ch = _ChannelState()
        ch.prn = int(prns[k])
        ch.code_phase = float(acquisition_lag_to_code_phase_chips(lags_in[k], fs))
        dop = float(dops[k])
        ch.code_freq = CA_CHIP_RATE + dop * CA_CHIP_RATE / GPS_L1_FREQ
        ch.carrier_phase = 0.0
        ch.carrier_freq = float(intermediate_freq) + dop
        ch.cn0 = 0.0
        ch.dll_integrator = 0.0
        ch.pll_integrator = 0.0
        ch.locked = True
        ch_list.append(ch)

    for _ in range(n_iter):
        corr = _batch_correlate(sig, ch_list, n_ch, n_samples, cfg).reshape(n_ch, 6)
        for k in range(n_ch):
            ei, eq = float(corr[k, 0]), float(corr[k, 1])
            li, lq = float(corr[k, 4]), float(corr[k, 5])
            pi_v, pq = float(corr[k, 2]), float(corr[k, 3])
            e_pow = ei * ei + eq * eq
            l_pow = li * li + lq * lq
            p_pow = pi_v * pi_v + pq * pq
            if p_pow < 1e-30:
                continue
            denom = e_pow + l_pow
            if denom < 1e-30:
                continue
            dll_disc = (e_pow - l_pow) / denom
            ch_list[k].code_phase += dll_gain * dll_disc
            ch_list[k].code_phase = float(
                np.mod(ch_list[k].code_phase, GPS_CA_CODE_LENGTH))
            if ch_list[k].code_phase < 0.0:
                ch_list[k].code_phase += GPS_CA_CODE_LENGTH

            if pll_gain > 0.0:
                pll_disc = float(np.arctan2(pq, pi_v))
                ch_list[k].carrier_phase += (pll_gain * pll_disc) / (2.0 * np.pi)
                ch_list[k].carrier_phase = float(
                    np.mod(ch_list[k].carrier_phase, 1.0))
                if ch_list[k].carrier_phase < 0.0:
                    ch_list[k].carrier_phase += 1.0

    prompt_pow = np.empty(n_ch, dtype=np.float64)
    dll_abs = np.empty(n_ch, dtype=np.float64)
    if return_lock_metrics:
        corr_f = _batch_correlate(sig, ch_list, n_ch, n_samples, cfg).reshape(n_ch, 6)
        for k in range(n_ch):
            ei, eq = float(corr_f[k, 0]), float(corr_f[k, 1])
            li, lq = float(corr_f[k, 4]), float(corr_f[k, 5])
            pi_v, pq = float(corr_f[k, 2]), float(corr_f[k, 3])
            e_pow = ei * ei + eq * eq
            l_pow = li * li + lq * lq
            prompt_pow[k] = pi_v * pi_v + pq * pq
            denom = e_pow + l_pow
            if denom < 1e-30:
                dll_abs[k] = 1.0
            else:
                dll_abs[k] = abs((e_pow - l_pow) / denom)

    out = np.empty(n_ch, dtype=np.float64)
    for k in range(n_ch):
        out[k] = code_phase_chips_to_acquisition_lag(ch_list[k].code_phase, fs)
    if return_lock_metrics:
        return out, prompt_pow, dll_abs
    return out


def refine_acquisition_code_lags_diagnostic_batch(
    signal_i,
    prns,
    code_phase_lags,
    doppler_hz_list,
    sampling_freq,
    intermediate_freq=0.0,
    n_iter=15,
    dll_gain=0.22,
    pll_gain=0.18,
    correlator_spacing=0.5,
):
    """DLL/PLL refinement with final per-channel correlator diagnostics."""
    prns = list(prns)
    n_ch = len(prns)
    if n_ch == 0:
        return _zero_diagnostics(prns, [], n_iter)

    try:
        from gnss_gpu._gnss_gpu_tracking import (
            TrackingConfig as _TrackingConfig,
            ChannelState as _ChannelState,
            batch_correlate as _batch_correlate,
        )
    except ImportError:
        return _zero_diagnostics(prns, code_phase_lags, n_iter)

    lags_in = np.asarray(code_phase_lags, dtype=np.float64).ravel()
    dops = np.asarray(doppler_hz_list, dtype=np.float64).ravel()
    if lags_in.size != n_ch or dops.size != n_ch:
        raise ValueError("prns, code_phase_lags, and doppler_hz_list must match in length")

    sig = np.asarray(signal_i, dtype=np.float32).ravel()
    n_samples = int(sig.shape[0])
    fs = float(sampling_freq)

    cfg = _TrackingConfig(
        sampling_freq=fs,
        intermediate_freq=float(intermediate_freq),
        integration_time=GPS_CA_PERIOD_S,
        dll_bandwidth=2.0,
        pll_bandwidth=15.0,
        correlator_spacing=float(correlator_spacing),
    )

    ch_list = []
    for k in range(n_ch):
        ch = _ChannelState()
        ch.prn = int(prns[k])
        ch.code_phase = float(acquisition_lag_to_code_phase_chips(lags_in[k], fs))
        dop = float(dops[k])
        ch.code_freq = CA_CHIP_RATE + dop * CA_CHIP_RATE / GPS_L1_FREQ
        ch.carrier_phase = 0.0
        ch.carrier_freq = float(intermediate_freq) + dop
        ch.cn0 = 0.0
        ch.dll_integrator = 0.0
        ch.pll_integrator = 0.0
        ch.locked = True
        ch_list.append(ch)

    for _ in range(n_iter):
        corr = _batch_correlate(sig, ch_list, n_ch, n_samples, cfg).reshape(n_ch, 6)
        for k in range(n_ch):
            ei, eq = float(corr[k, 0]), float(corr[k, 1])
            li, lq = float(corr[k, 4]), float(corr[k, 5])
            pi_v, pq = float(corr[k, 2]), float(corr[k, 3])
            e_pow = ei * ei + eq * eq
            l_pow = li * li + lq * lq
            p_pow = pi_v * pi_v + pq * pq
            if p_pow < 1e-30:
                continue
            denom = e_pow + l_pow
            if denom < 1e-30:
                continue
            dll_disc = (e_pow - l_pow) / denom
            ch_list[k].code_phase += dll_gain * dll_disc
            ch_list[k].code_phase = float(
                np.mod(ch_list[k].code_phase, GPS_CA_CODE_LENGTH))
            if ch_list[k].code_phase < 0.0:
                ch_list[k].code_phase += GPS_CA_CODE_LENGTH

            if pll_gain > 0.0:
                pll_disc = float(np.arctan2(pq, pi_v))
                ch_list[k].carrier_phase += (pll_gain * pll_disc) / (2.0 * np.pi)
                ch_list[k].carrier_phase = float(
                    np.mod(ch_list[k].carrier_phase, 1.0))
                if ch_list[k].carrier_phase < 0.0:
                    ch_list[k].carrier_phase += 1.0

    corr_f = _batch_correlate(sig, ch_list, n_ch, n_samples, cfg).reshape(n_ch, 6)
    diag = {
        key: np.empty(n_ch, dtype=np.float64)
        for key in _DIAGNOSTIC_ARRAY_KEYS
    }
    diag["prn"] = np.asarray(prns, dtype=np.int32)
    diag["n_iter_used"] = int(n_iter)

    for k in range(n_ch):
        ei, eq = float(corr_f[k, 0]), float(corr_f[k, 1])
        pi_v, pq = float(corr_f[k, 2]), float(corr_f[k, 3])
        li, lq = float(corr_f[k, 4]), float(corr_f[k, 5])
        e_pow = ei * ei + eq * eq
        l_pow = li * li + lq * lq
        prompt_pow = pi_v * pi_v + pq * pq
        denom = e_pow + l_pow

        diag["lag_samples"][k] = (
            lags_in[k] if n_iter <= 0
            else code_phase_chips_to_acquisition_lag(ch_list[k].code_phase, fs)
        )
        diag["code_phase_chips"][k] = float(
            np.mod(ch_list[k].code_phase, GPS_CA_CODE_LENGTH))
        diag["carrier_phase_cycles"][k] = float(np.mod(ch_list[k].carrier_phase, 1.0))
        diag["code_freq_hz"][k] = float(ch_list[k].code_freq)
        diag["carrier_freq_hz"][k] = float(ch_list[k].carrier_freq)
        diag["E_I"][k] = ei
        diag["E_Q"][k] = eq
        diag["P_I"][k] = pi_v
        diag["P_Q"][k] = pq
        diag["L_I"][k] = li
        diag["L_Q"][k] = lq
        diag["prompt_power"][k] = prompt_pow
        diag["dll_abs"][k] = 1.0 if denom < 1e-30 else abs((e_pow - l_pow) / denom)

    # Rough relative CN0 diagnostic for 1 ms coherent integration, not calibrated.
    # signal_plus_noise=P_I^2+P_Q^2; noise_proxy=max(P_Q^2, 1e-30); -30 dB is 1 ms.
    noise_proxy = np.maximum(diag["P_Q"] * diag["P_Q"], 1e-30)
    with np.errstate(divide="ignore", invalid="ignore"):
        diag["cn0_est_db"] = 10.0 * np.log10(diag["prompt_power"] / noise_proxy) - 30.0

    return diag


def dump_e2e_diagnostics_csv(path, diagnostics):
    """Write per-channel E2E diagnostics to CSV."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_ch = int(np.asarray(diagnostics["prn"]).size)

    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(_DIAGNOSTIC_CSV_COLUMNS)
        for i in range(n_ch):
            row = [str(int(np.asarray(diagnostics["prn"])[i]))]
            for key in _DIAGNOSTIC_CSV_COLUMNS[1:]:
                row.append("%.9g" % float(np.asarray(diagnostics[key])[i]))
            writer.writerow(row)


def pseudorange_to_code_phase_chips(pseudorange_m):
    """Convert pseudorange to GPS C/A code phase in chips."""
    pr = np.asarray(pseudorange_m, dtype=np.float64)
    code_phase = np.mod((pr / C_LIGHT) * CA_CHIP_RATE, GPS_CA_CODE_LENGTH)
    return _maybe_scalar(code_phase)


def acquisition_code_phase_to_pseudorange(
    code_phase_samples,
    sampling_freq,
    approx_pseudorange_m,
):
    """Recover pseudorange from acquisition lag using nearest-ms ambiguity.

    Acquisition returns the circular correlation lag in samples, not the
    absolute code delay. We map that lag back to the code delay within the
    current 1 ms C/A period, then choose the nearest full-millisecond
    ambiguity around an approximate pseudorange.
    """
    fs = float(sampling_freq)
    n_samples = int(round(fs * GPS_CA_PERIOD_S))

    lag = np.asarray(code_phase_samples, dtype=np.float64)
    approx = np.asarray(approx_pseudorange_m, dtype=np.float64)

    delay_samples = np.mod(n_samples - lag, n_samples)
    pseudorange_mod = (delay_samples / fs) * C_LIGHT
    ambiguity = np.rint((approx - pseudorange_mod) / GPS_CA_PERIOD_M)
    pseudorange = pseudorange_mod + ambiguity * GPS_CA_PERIOD_M
    return _maybe_scalar(pseudorange)
