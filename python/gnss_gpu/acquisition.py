"""GPU-accelerated GNSS signal acquisition."""

import numpy as np

from gnss_gpu.input_validation import (
    finite_float,
    nonnegative_float,
    positive_float,
)


def _validate_prn(name, value):
    try:
        prn = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if prn != value:
        raise ValueError(f"{name} must be an integer")
    if prn < 1 or prn > 32:
        raise ValueError(f"{name} must be in [1, 32]")
    return prn


def _as_signal_array(signal):
    arr = np.asarray(signal, dtype=np.float32)
    if arr.ndim != 1:
        raise ValueError("signal must be 1-D")
    if arr.size == 0:
        raise ValueError("signal must contain at least one sample")
    if not np.all(np.isfinite(arr)):
        raise ValueError("signal must be finite")
    return np.ascontiguousarray(arr, dtype=np.float32)


def _as_prn_array(prn_list):
    arr = np.asarray(prn_list)
    if arr.ndim != 1:
        raise ValueError("prn_list must be 1-D")
    if arr.size == 0:
        raise ValueError("prn_list must contain at least one PRN")

    if np.issubdtype(arr.dtype, np.integer):
        prns = arr.astype(np.int64, copy=False)
    else:
        try:
            numeric = arr.astype(np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("prn_list must contain integers") from exc
        if not np.all(np.isfinite(numeric)):
            raise ValueError("prn_list must contain integers")
        if not np.all(numeric == np.floor(numeric)):
            raise ValueError("prn_list must contain integers")
        prns = numeric.astype(np.int64)

    if np.any((prns < 1) | (prns > 32)):
        raise ValueError("prn_list values must be in [1, 32]")
    return np.ascontiguousarray(prns, dtype=np.int32)


class Acquisition:
    """Parallel code-phase / Doppler search for GPS C/A signals."""

    def __init__(self, sampling_freq, intermediate_freq=0,
                 doppler_range=5000, doppler_step=500, threshold=2.5):
        self.sampling_freq = positive_float("sampling_freq", sampling_freq)
        self.intermediate_freq = finite_float("intermediate_freq", intermediate_freq)
        self.doppler_range = nonnegative_float("doppler_range", doppler_range)
        self.doppler_step = positive_float("doppler_step", doppler_step)
        self.threshold = nonnegative_float("threshold", threshold)

    def acquire(self, signal, prn_list=None):
        """Run acquisition on the given signal.

        Args:
            signal: 1-D float32 array of IF samples.
            prn_list: List of PRN numbers to search (default: 1-32).

        Returns:
            List of dicts with keys: prn, acquired, code_phase, doppler_hz, snr.
        """
        if prn_list is None:
            prn_list = list(range(1, 33))

        signal = _as_signal_array(signal)
        prn_arr = _as_prn_array(prn_list)

        from gnss_gpu._gnss_gpu_acq import acquire_parallel as _acquire

        raw = _acquire(
            signal, self.sampling_freq, self.intermediate_freq,
            prn_arr, self.doppler_range, self.doppler_step, self.threshold)

        return raw

    @staticmethod
    def generate_test_signal(prn, code_phase, doppler, snr_db,
                             sampling_freq, duration_s=1e-3,
                             intermediate_freq=0):
        """Generate a synthetic GPS C/A signal for testing.

        Args:
            prn: Satellite PRN number (1-32).
            code_phase: Code phase offset in samples.
            doppler: Doppler shift in Hz.
            snr_db: Signal-to-noise ratio in dB.
            sampling_freq: Sampling frequency in Hz.
            duration_s: Signal duration in seconds.
            intermediate_freq: Intermediate frequency in Hz.

        Returns:
            1-D float32 array of IF samples.
        """
        prn = _validate_prn("prn", prn)
        code_phase = finite_float("code_phase", code_phase)
        doppler = finite_float("doppler", doppler)
        snr_db = finite_float("snr_db", snr_db)
        sampling_freq = positive_float("sampling_freq", sampling_freq)
        duration_s = positive_float("duration_s", duration_s)
        intermediate_freq = finite_float("intermediate_freq", intermediate_freq)

        n_samples = int(sampling_freq * duration_s)
        if n_samples < 1:
            raise ValueError("duration_s and sampling_freq must produce at least one sample")
        chip_rate = 1.023e6

        from gnss_gpu._gnss_gpu_acq import generate_ca_code as _gen_code

        # Generate and resample C/A code
        code_1023 = np.array(_gen_code(prn), dtype=np.float32)
        t = np.arange(n_samples) / sampling_freq
        chip_indices = (t * chip_rate).astype(int) % 1023
        code_sampled = code_1023[chip_indices]

        # Apply code phase shift (circular)
        code_sampled = np.roll(code_sampled, int(code_phase))

        # Generate carrier
        carrier_freq = intermediate_freq + doppler
        phase = 2.0 * np.pi * carrier_freq * t
        carrier = np.cos(phase).astype(np.float32)

        # Signal
        signal_power = 10.0 ** (snr_db / 10.0)
        signal = np.sqrt(signal_power) * code_sampled * carrier

        # Add noise
        rng = np.random.default_rng(42)
        noise = rng.standard_normal(n_samples).astype(np.float32)
        return (signal + noise).astype(np.float32)
