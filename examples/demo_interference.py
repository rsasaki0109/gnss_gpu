#!/usr/bin/env python3
"""Demo: GPU-accelerated GNSS interference detection and excision.

This script demonstrates the interference detection pipeline:
  1. Generate a clean GNSS-like wideband signal
  2. Inject a CW (continuous wave) jammer at a known frequency
  3. Run interference detection and print results
  4. Run interference excision and compare before/after spectra
  5. Save spectrogram data to file

No external data files are required.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Pure-Python STFT and interference routines (fallback)
# ---------------------------------------------------------------------------

def _stft_py(signal, fft_size, hop_size):
    """Compute power spectrogram via short-time FFT (Hann window)."""
    n = len(signal)
    window = np.hanning(fft_size).astype(np.float32)
    n_frames = max(1, (n - fft_size) // hop_size + 1)
    n_bins = fft_size // 2 + 1
    spectrogram = np.zeros((n_frames, n_bins), dtype=np.float32)

    for i in range(n_frames):
        start = i * hop_size
        segment = signal[start:start + fft_size] * window
        spectrum = np.fft.rfft(segment)
        spectrogram[i] = 10.0 * np.log10(np.abs(spectrum) ** 2 + 1e-20)

    return spectrogram


def _detect_interference_py(spectrogram, fft_size, sampling_freq, threshold_db):
    """Detect interference peaks above threshold relative to noise floor."""
    n_bins = spectrogram.shape[1]
    mean_power = np.mean(spectrogram, axis=0)
    noise_floor = np.median(mean_power)
    freq_resolution = sampling_freq / fft_size

    detections = []
    above = mean_power - noise_floor > threshold_db
    # Find contiguous groups of bins above threshold
    in_peak = False
    start_bin = 0
    for b in range(n_bins):
        if above[b] and not in_peak:
            in_peak = True
            start_bin = b
        elif not above[b] and in_peak:
            in_peak = False
            end_bin = b
            bw_bins = end_bin - start_bin
            center_bin = (start_bin + end_bin) // 2
            center_freq = center_bin * freq_resolution
            bandwidth = bw_bins * freq_resolution
            peak_power = float(np.max(mean_power[start_bin:end_bin]))

            if bw_bins <= 3:
                int_type = "CW"
            elif bw_bins <= n_bins // 4:
                int_type = "narrowband"
            else:
                int_type = "wideband"

            detections.append({
                "type_name": int_type,
                "center_freq_hz": float(center_freq),
                "bandwidth_hz": float(bandwidth),
                "power_db": peak_power,
                "noise_floor_db": float(noise_floor),
                "start_bin": start_bin,
                "end_bin": end_bin,
            })
    if in_peak:
        end_bin = n_bins
        bw_bins = end_bin - start_bin
        center_bin = (start_bin + end_bin) // 2
        center_freq = center_bin * freq_resolution
        bandwidth = bw_bins * freq_resolution
        peak_power = float(np.max(mean_power[start_bin:end_bin]))
        int_type = "CW" if bw_bins <= 3 else ("narrowband" if bw_bins <= n_bins // 4 else "wideband")
        detections.append({
            "type_name": int_type,
            "center_freq_hz": float(center_freq),
            "bandwidth_hz": float(bandwidth),
            "power_db": peak_power,
            "noise_floor_db": float(noise_floor),
            "start_bin": start_bin,
            "end_bin": end_bin,
        })

    return detections


def _excise_interference_py(signal, fft_size, hop_size, threshold_db):
    """Remove interference via frequency-domain notch filtering (overlap-add)."""
    n = len(signal)
    window = np.hanning(fft_size).astype(np.float32)
    n_frames = max(1, (n - fft_size) // hop_size + 1)
    output = np.zeros(n, dtype=np.float32)
    win_sum = np.zeros(n, dtype=np.float32)

    # Estimate noise floor from spectrogram
    spec = _stft_py(signal, fft_size, hop_size)
    mean_power = np.mean(spec, axis=0)
    noise_floor = np.median(mean_power)

    for i in range(n_frames):
        start = i * hop_size
        end = start + fft_size
        segment = signal[start:end] * window
        spectrum = np.fft.rfft(segment)
        power_db = 10.0 * np.log10(np.abs(spectrum) ** 2 + 1e-20)

        # Zero out bins that exceed threshold
        mask = power_db - noise_floor > threshold_db
        spectrum[mask] = 0.0

        cleaned = np.fft.irfft(spectrum, n=fft_size).astype(np.float32)
        output[start:end] += cleaned * window
        win_sum[start:end] += window ** 2

    # Normalise by window overlap
    nonzero = win_sum > 1e-8
    output[nonzero] /= win_sum[nonzero]

    return output


# ---------------------------------------------------------------------------
# Signal generation
# ---------------------------------------------------------------------------

def _generate_gnss_like_signal(sampling_freq, duration_s, seed=42):
    """Generate a wideband GNSS-like signal (spread-spectrum + noise)."""
    rng = np.random.default_rng(seed)
    n_samples = int(sampling_freq * duration_s)
    t = np.arange(n_samples) / sampling_freq

    # Spread-spectrum component (PRN-like)
    chip_rate = 1.023e6
    chip_idx = (t * chip_rate).astype(int) % 1023
    prn_seq = 2.0 * (rng.integers(0, 2, 1023).astype(np.float32)) - 1.0
    code = prn_seq[chip_idx]

    carrier = np.cos(2.0 * np.pi * 1.25e6 * t).astype(np.float32)
    signal = 0.1 * code * carrier

    # Add thermal noise
    noise = rng.standard_normal(n_samples).astype(np.float32)
    signal += noise

    return signal


def _inject_cw_jammer(signal, sampling_freq, jammer_freq, jammer_power_db):
    """Add a CW jammer at a given frequency and power level."""
    n_samples = len(signal)
    t = np.arange(n_samples) / sampling_freq
    amplitude = np.sqrt(2.0 * 10.0 ** (jammer_power_db / 10.0))
    jammer = amplitude * np.cos(2.0 * np.pi * jammer_freq * t).astype(np.float32)
    return signal + jammer


def _compute_psd(signal, fft_size, sampling_freq):
    """Compute average power spectral density in dB."""
    n_segments = max(1, len(signal) // fft_size)
    window = np.hanning(fft_size).astype(np.float32)
    psd = np.zeros(fft_size // 2 + 1, dtype=np.float64)
    for i in range(n_segments):
        start = i * fft_size
        segment = signal[start:start + fft_size] * window
        spectrum = np.fft.rfft(segment)
        psd += np.abs(spectrum) ** 2
    psd /= n_segments
    psd_db = 10.0 * np.log10(psd + 1e-20)
    freqs = np.linspace(0, sampling_freq / 2, len(psd_db))
    return freqs, psd_db


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("  gnss_gpu Demo: Interference Detection and Excision")
    print("=" * 70)

    # Parameters
    sampling_freq = 10.0e6     # 10 MHz
    duration_s = 10e-3         # 10 ms
    fft_size = 1024
    hop_size = 256
    threshold_db = 15.0

    jammer_freq = 3.5e6       # 3.5 MHz (within Nyquist band)
    jammer_power_db = 30.0    # 30 dB above noise

    print(f"\n[1] Signal parameters")
    print(f"    Sampling freq  : {sampling_freq / 1e6:.1f} MHz")
    print(f"    Duration       : {duration_s * 1e3:.1f} ms")
    print(f"    FFT size       : {fft_size}")
    print(f"    Hop size       : {hop_size}")
    print(f"    Threshold      : {threshold_db:.1f} dB")

    # --- Step 1: Generate clean GNSS signal ---
    print(f"\n[2] Generating clean GNSS-like signal ...")
    clean_signal = _generate_gnss_like_signal(sampling_freq, duration_s)
    clean_power = 10.0 * np.log10(np.mean(clean_signal ** 2) + 1e-20)
    print(f"    Samples : {len(clean_signal)}")
    print(f"    Power   : {clean_power:.1f} dB")

    # --- Step 2: Inject CW jammer ---
    print(f"\n[3] Injecting CW jammer")
    print(f"    Frequency : {jammer_freq / 1e6:.2f} MHz")
    print(f"    Power     : {jammer_power_db:.1f} dB above noise")
    jammed_signal = _inject_cw_jammer(clean_signal, sampling_freq, jammer_freq, jammer_power_db)
    jammed_power = 10.0 * np.log10(np.mean(jammed_signal ** 2) + 1e-20)
    print(f"    Jammed signal power: {jammed_power:.1f} dB")

    # --- Step 3: Detect interference ---
    print(f"\n[4] Running interference detection ...")
    try:
        from gnss_gpu import InterferenceDetector
        detector = InterferenceDetector(sampling_freq, fft_size, hop_size, threshold_db)
        detections = detector.detect(jammed_signal)
        det_source = "GPU"
    except (ImportError, RuntimeError, Exception) as e:
        print(f"    CUDA InterferenceDetector unavailable ({e}), using Python fallback")
        spec = _stft_py(jammed_signal, fft_size, hop_size)
        detections = _detect_interference_py(spec, fft_size, sampling_freq, threshold_db)
        det_source = "Python"

    print(f"    Detection engine: {det_source}")
    print(f"    Detected {len(detections)} interference source(s):")
    for i, d in enumerate(detections):
        print(f"      [{i}] Type: {d['type_name']}, "
              f"Freq: {d['center_freq_hz'] / 1e6:.3f} MHz, "
              f"BW: {d['bandwidth_hz'] / 1e3:.1f} kHz, "
              f"Power: {d['power_db']:.1f} dB")

    # --- Step 4: Excise interference ---
    print(f"\n[5] Running interference excision ...")
    try:
        from gnss_gpu import InterferenceDetector
        detector = InterferenceDetector(sampling_freq, fft_size, hop_size, threshold_db)
        cleaned_signal = detector.excise(jammed_signal)
        exc_source = "GPU"
    except (ImportError, RuntimeError, Exception) as e:
        print(f"    CUDA excision unavailable ({e}), using Python fallback")
        cleaned_signal = _excise_interference_py(jammed_signal, fft_size, hop_size, threshold_db)
        exc_source = "Python"

    cleaned_power = 10.0 * np.log10(np.mean(cleaned_signal ** 2) + 1e-20)
    print(f"    Excision engine : {exc_source}")
    print(f"    Cleaned power   : {cleaned_power:.1f} dB")

    # --- Step 5: Compare power spectra ---
    print(f"\n[6] Power spectrum comparison")
    freqs, psd_clean = _compute_psd(clean_signal, fft_size, sampling_freq)
    freqs, psd_jammed = _compute_psd(jammed_signal, fft_size, sampling_freq)
    freqs, psd_cleaned = _compute_psd(cleaned_signal, fft_size, sampling_freq)

    # Find peak near jammer frequency
    jammer_bin = int(jammer_freq / (sampling_freq / 2) * (len(freqs) - 1))
    search_lo = max(0, jammer_bin - 5)
    search_hi = min(len(freqs), jammer_bin + 6)

    peak_jammed = float(np.max(psd_jammed[search_lo:search_hi]))
    peak_cleaned = float(np.max(psd_cleaned[search_lo:search_hi]))
    peak_clean = float(np.max(psd_clean[search_lo:search_hi]))

    print(f"    PSD at jammer freq ({jammer_freq / 1e6:.2f} MHz):")
    print(f"      Clean signal  : {peak_clean:8.1f} dB")
    print(f"      Jammed signal : {peak_jammed:8.1f} dB")
    print(f"      After excision: {peak_cleaned:8.1f} dB")
    suppression = peak_jammed - peak_cleaned
    print(f"      Suppression   : {suppression:8.1f} dB")

    # --- Step 6: Save spectrogram data ---
    tmpdir = tempfile.mkdtemp(prefix="gnss_gpu_interference_")
    spec_path = Path(tmpdir) / "spectrogram_jammed.npy"
    psd_path = Path(tmpdir) / "psd_comparison.npz"

    print(f"\n[7] Saving data")
    spec = _stft_py(jammed_signal, fft_size, hop_size)
    np.save(str(spec_path), spec)
    print(f"    Spectrogram : {spec_path}  shape={spec.shape}")

    np.savez(str(psd_path),
             freqs=freqs,
             psd_clean=psd_clean,
             psd_jammed=psd_jammed,
             psd_cleaned=psd_cleaned)
    print(f"    PSD data    : {psd_path}")

    print(f"\n    Spectrogram info:")
    print(f"      Frames         : {spec.shape[0]}")
    print(f"      Freq bins      : {spec.shape[1]}")
    print(f"      Time resolution: {hop_size / sampling_freq * 1e3:.3f} ms")
    print(f"      Freq resolution: {sampling_freq / fft_size / 1e3:.3f} kHz")

    print("\n" + "=" * 70)
    print("  Demo complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
