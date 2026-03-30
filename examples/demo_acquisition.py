#!/usr/bin/env python3
"""Demo: GPU-accelerated GPS signal acquisition.

This script demonstrates the signal acquisition pipeline:
  1. Generate synthetic GPS L1 IF signals with known PRNs, code phases, and Dopplers
  2. Add noise to achieve a target C/N0
  3. Run the GPU acquisition engine
  4. Verify that detected parameters match the injected ones

No external data files are required.
"""

from __future__ import annotations

import sys
import numpy as np


# ---------------------------------------------------------------------------
# Pure-Python C/A code generator (Gold code, fallback)
# ---------------------------------------------------------------------------

_G2_TAP = {
    1: (2, 6), 2: (3, 7), 3: (4, 8), 4: (5, 9), 5: (1, 9),
    6: (2, 10), 7: (1, 8), 8: (2, 9), 9: (3, 10), 10: (2, 3),
    11: (3, 4), 12: (5, 6), 13: (6, 7), 14: (7, 8), 15: (8, 9),
    16: (9, 10), 17: (1, 4), 18: (2, 5), 19: (3, 6), 20: (4, 7),
    21: (5, 8), 22: (6, 9), 23: (1, 3), 24: (4, 6), 25: (5, 7),
    26: (6, 8), 27: (7, 9), 28: (8, 10), 29: (1, 6), 30: (2, 7),
    31: (3, 8), 32: (4, 9),
}


def _generate_ca_code_py(prn: int) -> np.ndarray:
    """Generate 1023-chip C/A code for the given PRN (1-32)."""
    g1 = np.ones(10, dtype=int)
    g2 = np.ones(10, dtype=int)
    tap1, tap2 = _G2_TAP[prn]
    code = np.zeros(1023, dtype=np.float32)
    for i in range(1023):
        g1_out = g1[9]
        g2_out = g2[tap1 - 1] ^ g2[tap2 - 1]
        code[i] = 1.0 - 2.0 * (g1_out ^ g2_out)
        fb1 = g1[2] ^ g1[9]
        g1 = np.roll(g1, 1)
        g1[0] = fb1
        fb2 = g2[1] ^ g2[2] ^ g2[5] ^ g2[7] ^ g2[8] ^ g2[9]
        g2 = np.roll(g2, 1)
        g2[0] = fb2
    return code


# ---------------------------------------------------------------------------
# Pure-Python acquisition (serial code-phase/Doppler search, fallback)
# ---------------------------------------------------------------------------

def _acquire_py(signal, sampling_freq, intermediate_freq, prn_list,
                doppler_range=5000, doppler_step=500, threshold=2.5):
    """Brute-force serial acquisition using FFT-based circular correlation."""
    n_samples = len(signal)
    chip_rate = 1.023e6
    results = []

    doppler_bins = np.arange(-doppler_range, doppler_range + doppler_step, doppler_step)

    for prn in prn_list:
        code_1023 = _generate_ca_code_py(prn)
        t = np.arange(n_samples) / sampling_freq
        chip_idx = (t * chip_rate).astype(int) % 1023
        code_sampled = code_1023[chip_idx]

        best_snr = 0.0
        best_cp = 0
        best_dop = 0.0

        for dop in doppler_bins:
            carrier_freq = intermediate_freq + dop
            carrier = np.cos(2.0 * np.pi * carrier_freq * t).astype(np.float32)
            mixed = signal * carrier

            # FFT-based circular correlation
            S = np.fft.fft(mixed)
            C = np.fft.fft(code_sampled)
            corr = np.abs(np.fft.ifft(S * np.conj(C))) ** 2

            peak_idx = int(np.argmax(corr))
            peak_val = corr[peak_idx]

            # Remove peak region for noise estimation
            mask = np.ones(len(corr), dtype=bool)
            margin = max(1, n_samples // 100)
            lo = max(0, peak_idx - margin)
            hi = min(len(corr), peak_idx + margin + 1)
            mask[lo:hi] = False
            noise_mean = np.mean(corr[mask])

            snr = peak_val / noise_mean if noise_mean > 0 else 0.0
            if snr > best_snr:
                best_snr = snr
                best_cp = peak_idx
                best_dop = dop

        results.append({
            "prn": prn,
            "acquired": bool(best_snr >= threshold),
            "code_phase": best_cp,
            "doppler_hz": best_dop,
            "snr": float(best_snr),
        })

    return results


# ---------------------------------------------------------------------------
# Signal generation (shared between GPU and fallback paths)
# ---------------------------------------------------------------------------

def _generate_composite_signal(sat_params, sampling_freq, intermediate_freq,
                               duration_s=1e-3, seed=42):
    """Generate a composite IF signal with multiple satellites + noise.

    Parameters
    ----------
    sat_params : list of dict
        Each dict has keys: prn, code_phase, doppler, snr_db
    """
    rng = np.random.default_rng(seed)
    n_samples = int(sampling_freq * duration_s)
    chip_rate = 1.023e6
    t = np.arange(n_samples) / sampling_freq

    signal = np.zeros(n_samples, dtype=np.float32)

    for sp in sat_params:
        code_1023 = _generate_ca_code_py(sp["prn"])
        chip_idx = (t * chip_rate).astype(int) % 1023
        code_sampled = code_1023[chip_idx]
        code_sampled = np.roll(code_sampled, int(sp["code_phase"]))

        carrier_freq = intermediate_freq + sp["doppler"]
        carrier = np.cos(2.0 * np.pi * carrier_freq * t).astype(np.float32)

        amplitude = np.sqrt(10.0 ** (sp["snr_db"] / 10.0))
        signal += amplitude * code_sampled * carrier

    noise = rng.standard_normal(n_samples).astype(np.float32)
    signal += noise

    return signal


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("  gnss_gpu Demo: GPS Signal Acquisition")
    print("=" * 70)

    # --- Signal parameters ---
    sampling_freq = 4.0e6       # 4 MHz
    intermediate_freq = 1.25e6  # 1.25 MHz IF
    duration_s = 1e-3           # 1 ms (one C/A code period)

    injected = [
        {"prn":  1, "code_phase":  200, "doppler":  1500.0, "snr_db": 35.0},
        {"prn":  5, "code_phase": 1500, "doppler": -2000.0, "snr_db": 33.0},
        {"prn": 10, "code_phase": 3000, "doppler":   500.0, "snr_db": 30.0},
    ]

    print(f"\n[1] Signal parameters")
    print(f"    Sampling freq  : {sampling_freq / 1e6:.1f} MHz")
    print(f"    IF             : {intermediate_freq / 1e6:.2f} MHz")
    print(f"    Duration       : {duration_s * 1e3:.1f} ms")
    print(f"\n    Injected satellites:")
    print(f"    {'PRN':>5s} {'Code Phase':>12s} {'Doppler [Hz]':>14s} {'C/N0 [dB-Hz]':>14s}")
    for sp in injected:
        print(f"    {sp['prn']:5d} {sp['code_phase']:12d} {sp['doppler']:14.1f} {sp['snr_db']:14.1f}")

    # --- Generate composite signal ---
    print("\n[2] Generating composite IF signal ...")
    signal = _generate_composite_signal(injected, sampling_freq, intermediate_freq,
                                        duration_s=duration_s, seed=42)
    print(f"    Signal length  : {len(signal)} samples")
    print(f"    Signal power   : {10 * np.log10(np.mean(signal ** 2)):.1f} dB")

    # --- Run acquisition ---
    prn_search = list(range(1, 33))
    print(f"\n[3] Running acquisition (searching PRN 1-32) ...")

    try:
        from gnss_gpu import Acquisition
        acq = Acquisition(sampling_freq, intermediate_freq,
                          doppler_range=5000, doppler_step=500, threshold=2.5)
        results = acq.acquire(signal, prn_list=prn_search)
        acq_source = "GPU"
    except (ImportError, RuntimeError, Exception) as e:
        print(f"    CUDA Acquisition unavailable ({e}), using Python fallback")
        results = _acquire_py(signal, sampling_freq, intermediate_freq, prn_search,
                              doppler_range=5000, doppler_step=500, threshold=2.5)
        acq_source = "Python"

    print(f"    Acquisition engine: {acq_source}")

    # --- Print results ---
    acquired = [r for r in results if r["acquired"]]
    not_acquired = [r for r in results if not r["acquired"]]

    print(f"\n[4] Acquisition results: {len(acquired)} satellites acquired")
    print(f"    {'PRN':>5s} {'Code Phase':>12s} {'Doppler [Hz]':>14s} {'SNR':>8s} {'Status':>10s}")
    print(f"    {'-' * 51}")
    for r in sorted(results, key=lambda x: -x["snr"]):
        status = "ACQUIRED" if r["acquired"] else "-"
        if r["acquired"]:
            print(f"    {r['prn']:5d} {r['code_phase']:12d} {r['doppler_hz']:14.1f}"
                  f" {r['snr']:8.2f} {status:>10s}")

    # --- Verify against injected parameters ---
    print(f"\n[5] Verification against injected parameters")
    injected_prns = {sp["prn"]: sp for sp in injected}
    acquired_prns = {r["prn"]: r for r in acquired}

    all_ok = True
    for prn, sp in injected_prns.items():
        if prn not in acquired_prns:
            print(f"    PRN {prn:2d}: NOT DETECTED")
            all_ok = False
            continue
        r = acquired_prns[prn]
        cp_err = abs(r["code_phase"] - sp["code_phase"])
        dop_err = abs(r["doppler_hz"] - sp["doppler"])
        cp_ok = cp_err < 2
        dop_ok = dop_err <= 500  # within one Doppler bin
        ok = cp_ok and dop_ok
        if not ok:
            all_ok = False
        print(f"    PRN {prn:2d}: code_phase_err={cp_err:4d} samples ({'OK' if cp_ok else 'FAIL'}), "
              f"doppler_err={dop_err:7.1f} Hz ({'OK' if dop_ok else 'FAIL'})")

    # Check for false acquisitions
    false_acq = [r["prn"] for r in acquired if r["prn"] not in injected_prns]
    if false_acq:
        print(f"    False acquisitions: PRN {false_acq}")
        all_ok = False
    else:
        print(f"    No false acquisitions.")

    print(f"\n    Overall: {'PASS' if all_ok else 'FAIL'}")

    print("\n" + "=" * 70)
    print("  Demo complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
