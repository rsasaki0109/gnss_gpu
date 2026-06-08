"""CPU-side validation tests for the interference wrapper."""

import numpy as np
import pytest

from gnss_gpu.interference import InterferenceDetector


def test_interference_init_rejects_invalid_config():
    with pytest.raises(ValueError, match="sampling_freq must be positive"):
        InterferenceDetector(sampling_freq=0.0)
    with pytest.raises(ValueError, match="fft_size must be >= 2"):
        InterferenceDetector(sampling_freq=10000.0, fft_size=1)
    with pytest.raises(ValueError, match="hop_size must be >= 1"):
        InterferenceDetector(sampling_freq=10000.0, hop_size=0)
    with pytest.raises(ValueError, match="threshold_db must be finite"):
        InterferenceDetector(sampling_freq=10000.0, threshold_db=np.nan)


def test_compute_spectrogram_rejects_invalid_signal_before_gpu_check():
    detector = InterferenceDetector(sampling_freq=10000.0, fft_size=8, hop_size=4)

    with pytest.raises(ValueError, match="signal must be 1-D"):
        detector.compute_spectrogram(np.ones((2, 8), dtype=np.float32))
    with pytest.raises(ValueError, match="signal length must be >= fft_size"):
        detector.compute_spectrogram(np.ones(7, dtype=np.float32))
    with pytest.raises(ValueError, match="signal must be finite"):
        detector.compute_spectrogram([0.0, 1.0, np.nan, 0.0, 0.0, 0.0, 0.0, 0.0])


@pytest.mark.parametrize("method_name", ["detect", "excise"])
def test_detect_and_excise_reject_invalid_signal_before_gpu_check(method_name):
    detector = InterferenceDetector(sampling_freq=10000.0, fft_size=8, hop_size=4)
    method = getattr(detector, method_name)

    with pytest.raises(ValueError, match="signal must be 1-D"):
        method(np.ones((2, 8), dtype=np.float32))
    with pytest.raises(ValueError, match="signal length must be >= fft_size"):
        method(np.ones(7, dtype=np.float32))
    with pytest.raises(ValueError, match="signal must be finite"):
        method([0.0, 1.0, np.inf, 0.0, 0.0, 0.0, 0.0, 0.0])
