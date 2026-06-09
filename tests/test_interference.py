"""Tests for GPU-accelerated interference detection and excision."""

import numpy as np
import pytest

from gnss_gpu.interference import InterferenceDetector


# Skip all tests if GPU bindings are not available
try:
    from gnss_gpu._gnss_gpu_interference import (
        compute_stft,
        detect_interference,
        excise_interference,
    )
    HAS_GPU = True
except ImportError:
    HAS_GPU = False

pytestmark = pytest.mark.skipif(not HAS_GPU, reason="GPU interference bindings not built")


@pytest.fixture
def detector():
    """Default interference detector at 10 kHz sampling rate."""
    return InterferenceDetector(sampling_freq=10000.0, fft_size=1024, hop_size=256,
                                threshold_db=15.0)


class TestSTFT:
    """Test STFT computation on simple signals."""

    def test_sinusoid_peak_at_correct_bin(self, detector):
        """A pure sinusoid should produce a peak at the correct frequency bin."""
        fs = detector.sampling_freq
        freq = 1000.0  # 1 kHz tone
        n_samples = 8192
        t = np.arange(n_samples) / fs
        signal = np.sin(2 * np.pi * freq * t).astype(np.float32)

        spec = detector.compute_spectrogram(signal)

        # Expected bin for 1 kHz: bin = freq * fft_size / fs = 1000 * 1024 / 10000 = 102.4
        expected_bin = int(round(freq * detector.fft_size / fs))

        # Average power across frames
        mean_power = spec.mean(axis=0)

        # Peak should be at or very near the expected bin
        peak_bin = np.argmax(mean_power)
        assert abs(peak_bin - expected_bin) <= 1, (
            f"Peak at bin {peak_bin}, expected ~{expected_bin}"
        )

    def test_spectrogram_shape(self, detector):
        """Spectrogram should have correct shape."""
        n_samples = 4096
        signal = np.random.randn(n_samples).astype(np.float32)
        spec = detector.compute_spectrogram(signal)

        n_frames = (n_samples - detector.fft_size) // detector.hop_size + 1
        n_bins = detector.fft_size // 2 + 1
        assert spec.shape == (n_frames, n_bins)

    def test_compute_stft_rejects_invalid_inputs(self):
        """Direct binding should reject invalid STFT inputs before CUDA work."""
        signal = np.ones(16, dtype=np.float32)

        with pytest.raises(RuntimeError, match="signal must be a 1D array"):
            compute_stft(np.ones((2, 8), dtype=np.float32), 8, 4, 10000.0)
        with pytest.raises(RuntimeError, match="signal length must be >= fft_size"):
            compute_stft(np.ones(7, dtype=np.float32), 8, 4, 10000.0)
        bad_signal = signal.copy()
        bad_signal[0] = np.nan
        with pytest.raises(RuntimeError, match="signal must be finite"):
            compute_stft(bad_signal, 8, 4, 10000.0)
        with pytest.raises(RuntimeError, match="fft_size must be >= 2"):
            compute_stft(signal, 1, 4, 10000.0)
        with pytest.raises(RuntimeError, match="hop_size must be >= 1"):
            compute_stft(signal, 8, 0, 10000.0)
        with pytest.raises(RuntimeError, match="sampling_freq must be positive"):
            compute_stft(signal, 8, 4, 0.0)


class TestCWDetection:
    """Test continuous wave (CW) interference detection."""

    def test_detect_cw_tone(self, detector):
        """Injecting a strong CW tone into noise should be detected."""
        fs = detector.sampling_freq
        n_samples = 16384
        t = np.arange(n_samples) / fs

        # White noise background
        np.random.seed(42)
        noise = np.random.randn(n_samples).astype(np.float32) * 0.1

        # Strong CW interference at 1 kHz
        cw_freq = 1000.0
        cw_power = 5.0  # much stronger than noise
        signal = noise + cw_power * np.sin(2 * np.pi * cw_freq * t).astype(np.float32)

        detections = detector.detect(signal)

        # Should detect at least one interference
        assert len(detections) >= 1, "Failed to detect CW interference"

        # The detection nearest to 1 kHz
        det = min(detections, key=lambda d: abs(d["center_freq_hz"] - cw_freq))
        assert abs(det["center_freq_hz"] - cw_freq) < 50.0, (
            f"Detected frequency {det['center_freq_hz']} Hz, expected ~{cw_freq} Hz"
        )

    def test_detect_multiple_tones(self, detector):
        """Multiple CW tones should each be detected."""
        fs = detector.sampling_freq
        n_samples = 16384
        t = np.arange(n_samples) / fs

        np.random.seed(123)
        noise = np.random.randn(n_samples).astype(np.float32) * 0.1

        freqs = [500.0, 2000.0]
        signal = noise.copy()
        for f in freqs:
            signal += 5.0 * np.sin(2 * np.pi * f * t).astype(np.float32)

        detections = detector.detect(signal)
        detected_freqs = [d["center_freq_hz"] for d in detections]

        for f in freqs:
            matches = [df for df in detected_freqs if abs(df - f) < 100.0]
            assert len(matches) >= 1, f"Failed to detect tone at {f} Hz"

    def test_detect_interference_rejects_invalid_inputs(self):
        """Direct binding should reject invalid spectrogram inputs."""
        spec = np.ones((3, 5), dtype=np.float32)

        with pytest.raises(RuntimeError, match="spectrogram must be 2D"):
            detect_interference(np.ones(15, dtype=np.float32), 8, 10000.0, 15.0)
        with pytest.raises(RuntimeError, match="n_frames must be >= 1"):
            detect_interference(np.ones((0, 5), dtype=np.float32), 8, 10000.0, 15.0)
        with pytest.raises(RuntimeError, match="spectrogram must have"):
            detect_interference(np.ones((3, 4), dtype=np.float32), 8, 10000.0, 15.0)
        bad_spec = spec.copy()
        bad_spec[0, 0] = np.inf
        with pytest.raises(RuntimeError, match="spectrogram must be finite"):
            detect_interference(bad_spec, 8, 10000.0, 15.0)
        with pytest.raises(RuntimeError, match="sampling_freq must be positive"):
            detect_interference(spec, 8, 0.0, 15.0)
        with pytest.raises(RuntimeError, match="threshold_db must be finite"):
            detect_interference(spec, 8, 10000.0, np.nan)
        with pytest.raises(RuntimeError, match="max_detections must be >= 1"):
            detect_interference(spec, 8, 10000.0, 15.0, max_detections=0)


class TestExcision:
    """Test interference excision."""

    def test_excision_reduces_interference_power(self, detector):
        """After excision, power at the interference frequency should be reduced."""
        fs = detector.sampling_freq
        n_samples = 16384
        t = np.arange(n_samples) / fs

        np.random.seed(7)
        noise = np.random.randn(n_samples).astype(np.float32) * 0.1
        cw_freq = 1500.0
        signal = noise + 10.0 * np.sin(2 * np.pi * cw_freq * t).astype(np.float32)

        cleaned = detector.excise(signal)

        # Compare power at interference frequency via DFT
        freq_bin = int(round(cw_freq * n_samples / fs))
        original_power = np.abs(np.fft.rfft(signal)[freq_bin]) ** 2
        cleaned_power = np.abs(np.fft.rfft(cleaned)[freq_bin]) ** 2

        # Excision should reduce power by at least 10 dB
        reduction_db = 10 * np.log10(original_power / (cleaned_power + 1e-30))
        assert reduction_db > 10.0, (
            f"Interference power only reduced by {reduction_db:.1f} dB, expected >10 dB"
        )

    def test_excision_preserves_signal_length(self, detector):
        """Output signal should have the same length as input."""
        n_samples = 8192
        signal = np.random.randn(n_samples).astype(np.float32)
        cleaned = detector.excise(signal)
        assert len(cleaned) == n_samples

    def test_excise_interference_rejects_invalid_inputs(self):
        """Direct binding should reject inconsistent excision inputs."""
        signal = np.ones(16, dtype=np.float32)
        spec = np.ones((3, 5), dtype=np.float32)

        with pytest.raises(RuntimeError, match="input must be a 1D array"):
            excise_interference(np.ones((2, 8), dtype=np.float32), spec, 8, 4, 15.0)
        with pytest.raises(RuntimeError, match="input length must be >= fft_size"):
            excise_interference(np.ones(7, dtype=np.float32), spec, 8, 4, 15.0)
        bad_signal = signal.copy()
        bad_signal[0] = np.inf
        with pytest.raises(RuntimeError, match="input must be finite"):
            excise_interference(bad_signal, spec, 8, 4, 15.0)
        with pytest.raises(RuntimeError, match="hop_size must be >= 1"):
            excise_interference(signal, spec, 8, 0, 15.0)
        with pytest.raises(RuntimeError, match="threshold_db must be finite"):
            excise_interference(signal, spec, 8, 4, np.nan)
        with pytest.raises(RuntimeError, match="spectrogram must have"):
            excise_interference(signal, np.ones((3, 4), dtype=np.float32), 8, 4, 15.0)
        with pytest.raises(RuntimeError, match="spectrogram n_frames must match"):
            excise_interference(signal, np.ones((2, 5), dtype=np.float32), 8, 4, 15.0)
        bad_spec = spec.copy()
        bad_spec[0, 0] = np.nan
        with pytest.raises(RuntimeError, match="spectrogram must be finite"):
            excise_interference(signal, bad_spec, 8, 4, 15.0)


class TestCleanSignal:
    """Test that clean signals produce no false detections."""

    def test_no_false_detections_on_gaussian_noise(self, detector):
        """Gaussian noise with moderate threshold should produce no detections."""
        np.random.seed(99)
        n_samples = 16384
        signal = np.random.randn(n_samples).astype(np.float32)

        # Use a high threshold to avoid false positives
        detector_strict = InterferenceDetector(
            sampling_freq=detector.sampling_freq,
            fft_size=detector.fft_size,
            hop_size=detector.hop_size,
            threshold_db=20.0,
        )
        detections = detector_strict.detect(signal)

        assert len(detections) == 0, (
            f"Got {len(detections)} false detection(s) on clean Gaussian noise"
        )
