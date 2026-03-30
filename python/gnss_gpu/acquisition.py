"""GPU-accelerated GNSS signal acquisition."""

import numpy as np


class Acquisition:
    """Parallel code-phase / Doppler search for GPS C/A signals."""

    def __init__(self, sampling_freq, intermediate_freq=0,
                 doppler_range=5000, doppler_step=500, threshold=2.5):
        self.sampling_freq = float(sampling_freq)
        self.intermediate_freq = float(intermediate_freq)
        self.doppler_range = float(doppler_range)
        self.doppler_step = float(doppler_step)
        self.threshold = float(threshold)

    def acquire(self, signal, prn_list=None):
        """Run acquisition on the given signal.

        Args:
            signal: 1-D float32 array of IF samples.
            prn_list: List of PRN numbers to search (default: 1-32).

        Returns:
            List of dicts with keys: prn, acquired, code_phase, doppler_hz, snr.
        """
        from gnss_gpu._gnss_gpu_acq import acquire_parallel as _acquire

        if prn_list is None:
            prn_list = list(range(1, 33))

        signal = np.asarray(signal, dtype=np.float32).ravel()
        prn_arr = np.asarray(prn_list, dtype=np.int32)

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
        from gnss_gpu._gnss_gpu_acq import generate_ca_code as _gen_code

        n_samples = int(sampling_freq * duration_s)
        chip_rate = 1.023e6

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
