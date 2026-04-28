"""Tests for GPU-accelerated GNSS signal acquisition (requires CUDA GPU)."""

import numpy as np
import pytest

try:
    from gnss_gpu._gnss_gpu_acq import generate_ca_code, acquire_parallel
    HAS_GPU = True
except ImportError:
    HAS_GPU = False

pytestmark = pytest.mark.skipif(not HAS_GPU, reason="CUDA acquisition module not available")


class TestCACodeGeneration:
    """Verify GPS C/A code generation correctness."""

    def test_code_length(self):
        """Each PRN code must be exactly 1023 chips."""
        for prn in range(1, 33):
            code = generate_ca_code(prn)
            assert len(code) == 1023, f"PRN {prn}: length {len(code)}"

    def test_code_values(self):
        """All chips must be +1 or -1."""
        for prn in range(1, 33):
            code = generate_ca_code(prn)
            for i, chip in enumerate(code):
                assert chip in (1, -1), f"PRN {prn}, chip {i}: value {chip}"

    def test_code_balance(self):
        """C/A codes should have near-equal number of +1 and -1 chips.

        Gold codes of length 1023 have exactly 512 ones and 511 zeros
        (or vice versa), so the count of +1 chips should be 512 or 511.
        """
        for prn in range(1, 33):
            code = generate_ca_code(prn)
            n_pos = sum(1 for c in code if c == 1)
            n_neg = 1023 - n_pos
            assert abs(n_pos - n_neg) <= 1, (
                f"PRN {prn}: {n_pos} positive, {n_neg} negative"
            )

    def test_prn1_first_10_chips(self):
        """PRN 1 first 10 chips should match known reference.

        PRN 1 octal 1440 -> binary first 10 chips: [1,1,-1,-1,1,-1,-1,-1,-1,-1]
        """
        code = generate_ca_code(1)
        expected = [1, 1, -1, -1, 1, -1, -1, -1, -1, -1]
        assert code[:10] == expected, f"PRN 1 first 10: {code[:10]}"

    def test_codes_unique(self):
        """All 32 PRN codes must be distinct."""
        codes = set()
        for prn in range(1, 33):
            code = tuple(generate_ca_code(prn))
            codes.add(code)
        assert len(codes) == 32


class TestAcquisition:
    """Test signal acquisition with synthetic GPS signals."""

    def _generate_test_signal(self, prn, code_phase, doppler, snr_db,
                               sampling_freq=4.092e6, duration_s=1e-3,
                               intermediate_freq=0):
        """Generate a synthetic GPS C/A signal."""
        n_samples = int(sampling_freq * duration_s)
        chip_rate = 1.023e6

        code_1023 = np.array(generate_ca_code(prn), dtype=np.float32)
        t = np.arange(n_samples) / sampling_freq
        chip_indices = (t * chip_rate).astype(int) % 1023
        code_sampled = code_1023[chip_indices]

        # Apply code phase shift
        code_sampled = np.roll(code_sampled, int(code_phase))

        # Carrier
        carrier_freq = intermediate_freq + doppler
        phase = 2.0 * np.pi * carrier_freq * t
        carrier = np.cos(phase).astype(np.float32)

        signal_power = 10.0 ** (snr_db / 10.0)
        signal = np.sqrt(signal_power) * code_sampled * carrier

        rng = np.random.default_rng(42)
        noise = rng.standard_normal(n_samples).astype(np.float32)
        return (signal + noise).astype(np.float32)

    def test_acquire_known_signal(self):
        """Acquire a signal with known PRN, code phase, and Doppler."""
        prn = 7
        true_code_phase = 100
        true_doppler = 1500.0
        sampling_freq = 4.092e6

        signal = self._generate_test_signal(
            prn=prn, code_phase=true_code_phase, doppler=true_doppler,
            snr_db=20.0, sampling_freq=sampling_freq)

        prn_list = np.array([prn], dtype=np.int32)
        results = acquire_parallel(
            signal, sampling_freq, 0.0, prn_list,
            5000.0, 500.0, 2.0)

        r = results[0]
        assert r["acquired"], f"PRN {prn} not acquired, SNR={r['snr']:.1f}"
        assert abs(r["code_phase"] - true_code_phase) < 5, (
            f"Code phase error: {r['code_phase']} vs {true_code_phase}"
        )
        # Real-valued cosine input has a Doppler sign ambiguity.
        assert abs(abs(r["doppler_hz"]) - abs(true_doppler)) <= 500, (
            f"Doppler error: {r['doppler_hz']} vs {true_doppler}"
        )

    def test_acquire_multiple_prns(self):
        """Only the injected PRN should be acquired."""
        prn = 15
        sampling_freq = 4.092e6

        signal = self._generate_test_signal(
            prn=prn, code_phase=200, doppler=0.0,
            snr_db=20.0, sampling_freq=sampling_freq)

        prn_list = np.array([10, 15, 20], dtype=np.int32)
        results = acquire_parallel(
            signal, sampling_freq, 0.0, prn_list,
            5000.0, 500.0, 2.5)

        acquired_prns = [r["prn"] for r in results if r["acquired"]]
        assert prn in acquired_prns, f"Target PRN {prn} not found in {acquired_prns}"

    def test_noise_only(self):
        """Random noise should not produce acquisitions."""
        sampling_freq = 4.092e6
        n_samples = int(sampling_freq * 1e-3)
        rng = np.random.default_rng(99)
        noise = rng.standard_normal(n_samples).astype(np.float32)

        prn_list = np.array([1, 5, 10, 20, 31], dtype=np.int32)
        results = acquire_parallel(
            noise, sampling_freq, 0.0, prn_list,
            5000.0, 500.0, 3.0)

        acquired = [r["prn"] for r in results if r["acquired"]]
        assert len(acquired) == 0, f"False acquisitions on noise: PRNs {acquired}"
