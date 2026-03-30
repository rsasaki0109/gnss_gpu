"""GPU-accelerated GNSS interference detection and excision."""

import numpy as np

try:
    from gnss_gpu._gnss_gpu_interference import (
        compute_stft,
        detect_interference,
        excise_interference,
    )
    _HAS_GPU = True
except ImportError:
    _HAS_GPU = False


class InterferenceDetector:
    """GNSS interference detector and exciser using GPU-accelerated STFT.

    Parameters
    ----------
    sampling_freq : float
        Sampling frequency in Hz.
    fft_size : int
        FFT window size (default 1024).
    hop_size : int
        Hop size between STFT frames (default 256).
    threshold_db : float
        Detection threshold above noise floor in dB (default 15).
    """

    def __init__(self, sampling_freq, fft_size=1024, hop_size=256, threshold_db=15.0):
        self.sampling_freq = float(sampling_freq)
        self.fft_size = int(fft_size)
        self.hop_size = int(hop_size)
        self.threshold_db = float(threshold_db)

    def compute_spectrogram(self, signal):
        """Compute STFT power spectrogram.

        Parameters
        ----------
        signal : array_like
            Input signal, shape (n_samples,).

        Returns
        -------
        spectrogram : ndarray
            Power spectrogram in dB, shape (n_frames, fft_size//2+1).
        """
        if not _HAS_GPU:
            raise RuntimeError("GPU bindings not available. Build with CUDA support.")
        signal = np.ascontiguousarray(signal, dtype=np.float32)
        return compute_stft(signal, self.fft_size, self.hop_size, self.sampling_freq)

    def detect(self, signal):
        """Detect interference in signal.

        Parameters
        ----------
        signal : array_like
            Input signal, shape (n_samples,).

        Returns
        -------
        detections : list of dict
            Each dict contains: type, type_name, center_freq_hz, bandwidth_hz,
            power_db, start_frame, end_frame.
        """
        if not _HAS_GPU:
            raise RuntimeError("GPU bindings not available. Build with CUDA support.")
        signal = np.ascontiguousarray(signal, dtype=np.float32)
        spectrogram = compute_stft(signal, self.fft_size, self.hop_size, self.sampling_freq)
        return detect_interference(spectrogram, self.fft_size, self.sampling_freq,
                                   self.threshold_db)

    def excise(self, signal):
        """Remove interference from signal.

        Parameters
        ----------
        signal : array_like
            Input signal, shape (n_samples,).

        Returns
        -------
        cleaned : ndarray
            Signal with interference removed, shape (n_samples,).
        """
        if not _HAS_GPU:
            raise RuntimeError("GPU bindings not available. Build with CUDA support.")
        signal = np.ascontiguousarray(signal, dtype=np.float32)
        spectrogram = compute_stft(signal, self.fft_size, self.hop_size, self.sampling_freq)
        return excise_interference(signal, spectrogram, self.fft_size, self.hop_size,
                                   self.threshold_db)
