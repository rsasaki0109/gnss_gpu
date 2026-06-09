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


def _finite_float(name, value):
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not np.isfinite(out):
        raise ValueError(f"{name} must be finite")
    return out


def _positive_float(name, value):
    out = _finite_float(name, value)
    if out <= 0.0:
        raise ValueError(f"{name} must be positive")
    return out


def _positive_int(name, value, *, minimum=1):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        out = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if out != value:
        raise ValueError(f"{name} must be an integer")
    if out < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return out


def _as_signal_array(signal, fft_size):
    arr = np.asarray(signal, dtype=np.float32)
    if arr.ndim != 1:
        raise ValueError("signal must be 1-D")
    if arr.size < fft_size:
        raise ValueError("signal length must be >= fft_size")
    if not np.all(np.isfinite(arr)):
        raise ValueError("signal must be finite")
    return np.ascontiguousarray(arr, dtype=np.float32)


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
        self.sampling_freq = _positive_float("sampling_freq", sampling_freq)
        self.fft_size = _positive_int("fft_size", fft_size, minimum=2)
        self.hop_size = _positive_int("hop_size", hop_size)
        self.threshold_db = _finite_float("threshold_db", threshold_db)

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
        signal = _as_signal_array(signal, self.fft_size)
        if not _HAS_GPU:
            raise RuntimeError("GPU bindings not available. Build with CUDA support.")
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
        signal = _as_signal_array(signal, self.fft_size)
        if not _HAS_GPU:
            raise RuntimeError("GPU bindings not available. Build with CUDA support.")
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
        signal = _as_signal_array(signal, self.fft_size)
        if not _HAS_GPU:
            raise RuntimeError("GPU bindings not available. Build with CUDA support.")
        spectrogram = compute_stft(signal, self.fft_size, self.hop_size, self.sampling_freq)
        return excise_interference(signal, spectrogram, self.fft_size, self.hop_size,
                                   self.threshold_db)
