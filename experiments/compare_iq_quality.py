#!/usr/bin/env python3
"""Compare IQ output quality: gnss_gpu (GPU) vs gps-sdr-sim (CPU).

Generates side-by-side comparison of:
  1. Time-domain waveform
  2. Power spectrum (FFT)
  3. Acquisition correlation peak
  4. I/Q histogram
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from gnss_gpu.acquisition import Acquisition


def load_iq_int8(path, n_samples=None):
    """Load interleaved I/Q int8 binary."""
    raw = np.fromfile(path, dtype=np.int8)
    if n_samples is not None:
        raw = raw[: n_samples * 2]
    i = raw[0::2].astype(np.float32)
    q = raw[1::2].astype(np.float32)
    return i, q


def compute_spectrum(signal, fs, nfft=4096):
    """Power spectral density via Welch's method."""
    from scipy.signal import welch
    try:
        f, psd = welch(signal, fs=fs, nperseg=nfft, return_onesided=False)
        # shift to center
        f = np.fft.fftshift(f)
        psd = np.fft.fftshift(psd)
        return f, 10 * np.log10(psd + 1e-20)
    except ImportError:
        # Fallback: simple FFT
        n = min(len(signal), nfft * 16)
        sig = signal[:n]
        S = np.fft.fftshift(np.abs(np.fft.fft(sig, nfft)) ** 2 / nfft)
        f = np.fft.fftshift(np.fft.fftfreq(nfft, 1.0 / fs))
        return f, 10 * np.log10(S + 1e-20)


def compute_correlation(signal, prn, fs):
    """Compute code correlation for a given PRN."""
    from gnss_gpu._gnss_gpu_acq import generate_ca_code
    code = np.array(generate_ca_code(prn), dtype=np.float32)

    n = int(fs * 1e-3)  # 1ms
    sig = signal[:n]

    # Resample code
    chip_rate = 1.023e6
    t = np.arange(n) / fs
    chip_idx = (t * chip_rate).astype(int) % 1023
    code_sampled = code[chip_idx]

    # Cross-correlation via FFT
    S = np.fft.fft(sig)
    C = np.fft.fft(code_sampled)
    corr = np.abs(np.fft.ifft(S * np.conj(C))) ** 2
    return corr / np.max(corr)


def main():
    out_dir = os.path.join(os.path.dirname(__file__), "results", "iq_comparison")
    os.makedirs(out_dir, exist_ok=True)

    fs = 2.6e6

    # Load both IQ files
    print("Loading IQ data...")
    gpssdr_i, gpssdr_q = load_iq_int8("/tmp/gpssim_compare.bin")
    gnss_gpu_i, gnss_gpu_q = load_iq_int8("/tmp/gnss_gpu_compare.bin")

    print(f"  gps-sdr-sim: {len(gpssdr_i)} samples")
    print(f"  gnss_gpu:    {len(gnss_gpu_i)} samples")

    # --- Figure ---
    fig, axes = plt.subplots(2, 4, figsize=(20, 9), facecolor="#1a1a2e")
    tc, bg = "#e0e0e0", "#16213e"

    datasets = [
        ("gps-sdr-sim (CPU)", gpssdr_i, gpssdr_q, 0),
        ("gnss_gpu (GPU)", gnss_gpu_i, gnss_gpu_q, 1),
    ]

    for label, sig_i, sig_q, row in datasets:
        # 1. Time domain (first 200 samples)
        ax = axes[row, 0]
        ax.set_facecolor(bg)
        n_show = 200
        t_us = np.arange(n_show) / fs * 1e6
        ax.plot(t_us, sig_i[:n_show], color="#6c5ce7", linewidth=0.8, alpha=0.9, label="I")
        ax.plot(t_us, sig_q[:n_show], color="#fd79a8", linewidth=0.8, alpha=0.6, label="Q")
        ax.set_title(f"{label}\nTime Domain", color=tc, fontsize=10)
        ax.set_xlabel("Time [μs]", color=tc, fontsize=8)
        ax.set_ylabel("Amplitude", color=tc, fontsize=8)
        ax.tick_params(colors=tc, labelsize=7)
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, color="#333355", alpha=0.3)
        for s in ax.spines.values():
            s.set_color("#333355")

        # 2. Power spectrum
        ax = axes[row, 1]
        ax.set_facecolor(bg)
        f, psd = compute_spectrum(sig_i, fs, nfft=4096)
        ax.plot(f / 1e6, psd, color="#00d4aa", linewidth=0.8)
        ax.set_title("Power Spectrum", color=tc, fontsize=10)
        ax.set_xlabel("Frequency [MHz]", color=tc, fontsize=8)
        ax.set_ylabel("PSD [dB]", color=tc, fontsize=8)
        ax.tick_params(colors=tc, labelsize=7)
        ax.grid(True, color="#333355", alpha=0.3)
        for s in ax.spines.values():
            s.set_color("#333355")

        # 3. Correlation peak (PRN 15 — common in both)
        ax = axes[row, 2]
        ax.set_facecolor(bg)
        prn_test = 15
        corr = compute_correlation(sig_i, prn_test, fs)
        n_corr = len(corr)
        chips = np.arange(n_corr) * CA_CHIP_RATE / fs
        ax.plot(chips, corr, color="#ffd93d", linewidth=0.8)
        peak_idx = np.argmax(corr)
        ax.axvline(chips[peak_idx], color="#ff6b6b", linestyle="--", alpha=0.7)
        ax.set_title(f"Correlation (PRN {prn_test})", color=tc, fontsize=10)
        ax.set_xlabel("Code Phase [chips]", color=tc, fontsize=8)
        ax.set_ylabel("Normalized", color=tc, fontsize=8)
        ax.tick_params(colors=tc, labelsize=7)
        ax.grid(True, color="#333355", alpha=0.3)
        for s in ax.spines.values():
            s.set_color("#333355")

        # 4. I/Q histogram
        ax = axes[row, 3]
        ax.set_facecolor(bg)
        ax.hist(sig_i[:50000], bins=50, alpha=0.7, color="#6c5ce7", label="I", density=True)
        ax.hist(sig_q[:50000], bins=50, alpha=0.5, color="#fd79a8", label="Q", density=True)
        rms_i = np.sqrt(np.mean(sig_i[:50000].astype(np.float64) ** 2))
        ax.set_title(f"I/Q Distribution (RMS={rms_i:.1f})", color=tc, fontsize=10)
        ax.set_xlabel("Amplitude", color=tc, fontsize=8)
        ax.set_ylabel("Density", color=tc, fontsize=8)
        ax.tick_params(colors=tc, labelsize=7)
        ax.legend(fontsize=7)
        ax.grid(True, color="#333355", alpha=0.3)
        for s in ax.spines.values():
            s.set_color("#333355")

    fig.suptitle("IQ Output Quality: gps-sdr-sim (CPU) vs gnss_gpu (GPU)",
                 color="white", fontsize=14, fontweight="bold")
    plt.tight_layout()

    out_path = os.path.join(out_dir, "iq_comparison.png")
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved: {out_path}")

    # Acquisition comparison
    print("\nAcquisition comparison (PRN 1-32):")
    acq = Acquisition(sampling_freq=fs, intermediate_freq=0, threshold=2.0)
    for label, sig_i in [("gps-sdr-sim", gpssdr_i), ("gnss_gpu", gnss_gpu_i)]:
        results = acq.acquire(sig_i[:int(fs * 1e-3)], prn_list=list(range(1, 33)))
        acquired = [(r["prn"], f"SNR={r['snr']:.1f}") for r in results if r["acquired"]]
        print(f"  {label}: {len(acquired)} acquired — {acquired[:6]}...")


CA_CHIP_RATE = 1.023e6

if __name__ == "__main__":
    main()
