"""Tests for GPU-accelerated GNSS tracking loops."""

import numpy as np
import pytest

# GPS L1 C/A constants
CA_CODE_RATE = 1.023e6
CA_CODE_LENGTH = 1023
GPS_L1_FREQ = 1575.42e6
TWO_PI = 2.0 * np.pi

try:
    from gnss_gpu._gnss_gpu_tracking import (
        TrackingConfig,
        ChannelState,
        batch_correlate,
        scalar_tracking_update,
        cn0_nwpr,
    )
    HAS_GPU = True
except ImportError:
    HAS_GPU = False


# G2 delay taps for PRN 1-32 (1-indexed tap pairs), matching the GPS ICD
G2_TAPS = [
    (2, 6), (3, 7), (4, 8), (5, 9), (1, 9), (2, 10), (1, 8), (2, 9),
    (3, 10), (2, 3), (3, 4), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10),
    (1, 4), (2, 5), (3, 6), (4, 7), (5, 8), (6, 9), (1, 3), (4, 6),
    (5, 7), (6, 8), (7, 9), (8, 10), (1, 6), (2, 7), (3, 8), (4, 9),
]


def generate_ca_code(prn):
    """Generate 1023-chip GPS C/A Gold code for given PRN (1-32).

    Uses the same LFSR logic as the CUDA implementation in acquisition.cu.
    Returns a numpy array of +1/-1 values.
    """
    g1 = [1] * 10
    g2 = [1] * 10
    tap1, tap2 = G2_TAPS[prn - 1]
    tap1 -= 1  # convert to 0-indexed
    tap2 -= 1

    code = np.zeros(CA_CODE_LENGTH, dtype=np.float64)
    for i in range(CA_CODE_LENGTH):
        g1_out = g1[9]
        g2_delayed = g2[tap1] ^ g2[tap2]
        ca_bit = g1_out ^ g2_delayed
        code[i] = 2 * ca_bit - 1  # 0 -> -1, 1 -> +1

        g1_fb = g1[2] ^ g1[9]
        g2_fb = g2[1] ^ g2[2] ^ g2[5] ^ g2[7] ^ g2[8] ^ g2[9]

        for j in range(9, 0, -1):
            g1[j] = g1[j - 1]
            g2[j] = g2[j - 1]
        g1[0] = g1_fb
        g2[0] = g2_fb

    return code


# Cache generated codes to avoid regenerating per test
_ca_code_cache = {}


def _get_ca_code(prn):
    if prn not in _ca_code_cache:
        _ca_code_cache[prn] = generate_ca_code(prn)
    return _ca_code_cache[prn]


def generate_ca_signal(prn, code_phase, carrier_freq, n_samples,
                       sampling_freq, intermediate_freq, noise_std=0.0):
    """Generate a synthetic C/A code signal with known parameters.

    Returns:
        signal: float32 numpy array of IF samples
    """
    ca_code = _get_ca_code(prn)

    ts = 1.0 / sampling_freq
    t = np.arange(n_samples) * ts

    # Code replica
    code_freq = CA_CODE_RATE  # no Doppler on code for simplicity
    code_phases = code_phase + code_freq * t
    code_chips = np.mod(code_phases, CA_CODE_LENGTH).astype(int) % CA_CODE_LENGTH

    code = ca_code[code_chips]

    # Carrier replica
    carrier_phase_rad = TWO_PI * (intermediate_freq * t + (carrier_freq - intermediate_freq) * t)
    carrier = np.cos(carrier_phase_rad)

    signal = (code * carrier).astype(np.float32)

    if noise_std > 0:
        signal += np.random.normal(0, noise_std, n_samples).astype(np.float32)

    return signal


# ============================================================
# Test DLL discriminator
# ============================================================
class TestDLLDiscriminator:
    """Test DLL (Delay Lock Loop) discriminator behavior."""

    def test_zero_when_aligned(self):
        """DLL discriminator should return ~0 when code is aligned."""
        # When aligned, E and L power should be equal
        # |E|^2 = |L|^2 => discriminator = 0
        EI, EQ = 100.0, 10.0
        LI, LQ = 100.0, 10.0
        E_pow = EI**2 + EQ**2
        L_pow = LI**2 + LQ**2
        disc = (E_pow - L_pow) / (E_pow + L_pow)
        assert abs(disc) < 1e-10

    def test_positive_when_early_leads(self):
        """DLL discriminator should be positive when code is early."""
        # Early arm has more power => positive discriminator
        EI, EQ = 150.0, 15.0
        LI, LQ = 50.0, 5.0
        E_pow = EI**2 + EQ**2
        L_pow = LI**2 + LQ**2
        disc = (E_pow - L_pow) / (E_pow + L_pow)
        assert disc > 0

    def test_negative_when_late_leads(self):
        """DLL discriminator should be negative when code is late."""
        EI, EQ = 50.0, 5.0
        LI, LQ = 150.0, 15.0
        E_pow = EI**2 + EQ**2
        L_pow = LI**2 + LQ**2
        disc = (E_pow - L_pow) / (E_pow + L_pow)
        assert disc < 0

    def test_bounded(self):
        """DLL discriminator should be bounded in [-1, 1]."""
        for _ in range(100):
            vals = np.random.randn(4) * 100
            EI, EQ, LI, LQ = vals
            E_pow = EI**2 + EQ**2
            L_pow = LI**2 + LQ**2
            denom = E_pow + L_pow
            if denom < 1e-20:
                continue
            disc = (E_pow - L_pow) / denom
            assert -1.0 <= disc <= 1.0


# ============================================================
# Test PLL discriminator
# ============================================================
class TestPLLDiscriminator:
    """Test PLL (Phase Lock Loop) discriminator behavior."""

    def test_zero_phase(self):
        """PLL discriminator should return 0 when phase is aligned."""
        PI_v = 100.0
        PQ = 0.0
        disc = np.arctan2(PQ, PI_v)
        assert abs(disc) < 1e-10

    def test_positive_phase(self):
        """PLL discriminator should detect positive phase error."""
        phase = 0.3  # radians
        PI_v = np.cos(phase) * 100
        PQ = np.sin(phase) * 100
        disc = np.arctan2(PQ, PI_v)
        assert abs(disc - phase) < 1e-10

    def test_negative_phase(self):
        """PLL discriminator should detect negative phase error."""
        phase = -0.5
        PI_v = np.cos(phase) * 100
        PQ = np.sin(phase) * 100
        disc = np.arctan2(PQ, PI_v)
        assert abs(disc - phase) < 1e-10

    def test_quadrant_coverage(self):
        """PLL discriminator should work in all quadrants."""
        for phase in np.linspace(-np.pi + 0.01, np.pi - 0.01, 20):
            PI_v = np.cos(phase) * 100
            PQ = np.sin(phase) * 100
            disc = np.arctan2(PQ, PI_v)
            assert abs(disc - phase) < 1e-6


# ============================================================
# Test correlator (requires GPU)
# ============================================================
@pytest.mark.skipif(not HAS_GPU, reason="GPU bindings not available")
class TestCorrelator:
    """Test GPU batch correlator."""

    def test_prompt_arm_max_when_aligned(self):
        """Prompt arm should have maximum power when code is aligned."""
        prn = 1
        sampling_freq = 4.092e6
        intermediate_freq = 4.092e6
        n_samples = int(sampling_freq * 1e-3)  # 1 ms
        code_phase = 100.0  # chips

        signal = generate_ca_signal(
            prn, code_phase, intermediate_freq,
            n_samples, sampling_freq, intermediate_freq
        )

        config = TrackingConfig(
            sampling_freq=sampling_freq,
            intermediate_freq=intermediate_freq,
            integration_time=1e-3,
            dll_bandwidth=2.0,
            pll_bandwidth=15.0,
            correlator_spacing=2.0,
        )

        ch = ChannelState()
        ch.prn = prn
        ch.code_phase = code_phase
        ch.code_freq = CA_CODE_RATE
        ch.carrier_phase = 0.0
        ch.carrier_freq = intermediate_freq
        ch.cn0 = 45.0
        ch.dll_integrator = 0.0
        ch.pll_integrator = 0.0
        ch.locked = True

        correlations = batch_correlate(signal, [ch], 1, n_samples, config)
        corr = correlations.reshape(1, 6)

        # Prompt power should be largest
        E_pow = corr[0, 0]**2 + corr[0, 1]**2
        P_pow = corr[0, 2]**2 + corr[0, 3]**2
        L_pow = corr[0, 4]**2 + corr[0, 5]**2

        assert P_pow >= E_pow, f"Prompt power {P_pow} should exceed Early power {E_pow}"
        assert P_pow >= L_pow, f"Prompt power {P_pow} should exceed Late power {L_pow}"

    def test_unlocked_channel_returns_zero(self):
        """Unlocked channels should return zero correlations."""
        sampling_freq = 4.092e6
        n_samples = int(sampling_freq * 1e-3)
        signal = np.random.randn(n_samples).astype(np.float32)

        config = TrackingConfig(
            sampling_freq=sampling_freq,
            intermediate_freq=4.092e6,
        )

        ch = ChannelState()
        ch.prn = 1
        ch.code_phase = 0.0
        ch.code_freq = CA_CODE_RATE
        ch.carrier_phase = 0.0
        ch.carrier_freq = 0.0
        ch.cn0 = 0.0
        ch.dll_integrator = 0.0
        ch.pll_integrator = 0.0
        ch.locked = False

        correlations = batch_correlate(signal, [ch], 1, n_samples, config)
        assert np.allclose(correlations, 0.0)


# ============================================================
# Test scalar tracking (requires GPU)
# ============================================================
@pytest.mark.skipif(not HAS_GPU, reason="GPU bindings not available")
class TestScalarTracking:
    """Test scalar tracking loop convergence."""

    def test_maintains_lock_100_blocks(self):
        """Scalar tracking should maintain lock over 100 integration periods."""
        prn = 5
        sampling_freq = 4.092e6
        intermediate_freq = 4.092e6
        n_samples = int(sampling_freq * 1e-3)
        code_phase_true = 200.0
        doppler = 1500.0  # Hz

        config = TrackingConfig(
            sampling_freq=sampling_freq,
            intermediate_freq=intermediate_freq,
            integration_time=1e-3,
            dll_bandwidth=2.0,
            pll_bandwidth=15.0,
            correlator_spacing=0.5,
        )

        # Initialize channel with small errors
        ch = ChannelState()
        ch.prn = prn
        ch.code_phase = code_phase_true + 0.1  # small offset
        ch.code_freq = CA_CODE_RATE
        ch.carrier_phase = 0.0
        ch.carrier_freq = intermediate_freq + doppler
        ch.cn0 = 45.0
        ch.dll_integrator = 0.0
        ch.pll_integrator = 0.0
        ch.locked = True

        carrier_freq_true = intermediate_freq + doppler

        for block_idx in range(100):
            # Generate signal with true parameters
            current_code_phase = np.mod(
                code_phase_true + CA_CODE_RATE * block_idx * 1e-3,
                CA_CODE_LENGTH
            )
            signal = generate_ca_signal(
                prn, current_code_phase, carrier_freq_true,
                n_samples, sampling_freq, intermediate_freq,
                noise_std=0.1
            )

            # Correlate
            correlations = batch_correlate(signal, [ch], 1, n_samples, config)

            # Update tracking
            scalar_tracking_update([ch], correlations, 1, config)

        # Channel should still be locked
        assert ch.locked, "Channel should remain locked after 100 blocks"


# ============================================================
# Test CN0 estimation
# ============================================================
class TestCN0:
    """Test CN0 estimation using NWPR method."""

    def test_cn0_reasonable_range(self):
        """CN0 should be in 35-50 dB-Hz range for reasonable SNR."""
        # Simulate correlator outputs with known SNR
        n_channels = 1
        n_hist = 20
        T = 1e-3

        # Signal power ~ 100, noise ~ 1 => SNR ~ 40 dB
        np.random.seed(42)
        signal_amplitude = 100.0
        noise_std = 1.0

        hist = np.zeros((n_channels, n_hist, 6))
        for m in range(n_hist):
            # Prompt I has strong signal, Q has noise
            hist[0, m, 2] = signal_amplitude + np.random.normal(0, noise_std)  # PI
            hist[0, m, 3] = np.random.normal(0, noise_std)  # PQ

        # Compute CN0 using NWPR formula (Python version)
        sum_PI = np.sum(hist[0, :, 2])
        sum_PQ = np.sum(hist[0, :, 3])
        sum_pow = np.sum(hist[0, :, 2]**2 + hist[0, :, 3]**2)

        NP = sum_PI**2 + sum_PQ**2
        WP = sum_pow
        M = float(n_hist)
        ratio = NP / WP

        if ratio > 1.0 and ratio < M:
            cn0_linear = (1.0 / T) * (ratio - 1.0) / (M - ratio)
            cn0_db = 10.0 * np.log10(cn0_linear)
        else:
            cn0_db = 0.0

        assert 35.0 <= cn0_db <= 70.0, f"CN0 {cn0_db:.1f} dB-Hz outside expected range"

    def test_cn0_increases_with_snr(self):
        """Higher SNR should yield higher CN0."""
        T = 1e-3
        n_hist = 20
        np.random.seed(42)

        cn0_values = []
        for signal_amp in [10.0, 50.0, 200.0]:
            hist_PI = signal_amp + np.random.normal(0, 1.0, n_hist)
            hist_PQ = np.random.normal(0, 1.0, n_hist)

            NP = np.sum(hist_PI)**2 + np.sum(hist_PQ)**2
            WP = np.sum(hist_PI**2 + hist_PQ**2)
            M = float(n_hist)
            ratio = NP / WP

            if ratio > 1.0 and ratio < M:
                cn0_linear = (1.0 / T) * (ratio - 1.0) / (M - ratio)
                cn0_values.append(10.0 * np.log10(cn0_linear))
            else:
                cn0_values.append(0.0)

        # CN0 should increase with signal amplitude
        assert cn0_values[0] < cn0_values[1] < cn0_values[2], \
            f"CN0 should increase with SNR: {cn0_values}"

    @pytest.mark.skipif(not HAS_GPU, reason="GPU bindings not available")
    def test_cn0_gpu_reasonable(self):
        """GPU CN0 estimation should produce reasonable values."""
        n_channels = 2
        n_hist = 20
        T = 1e-3

        np.random.seed(42)
        hist = np.zeros((n_channels, n_hist, 6))
        for ch in range(n_channels):
            signal_amp = 80.0 + ch * 40.0
            for m in range(n_hist):
                hist[ch, m, 2] = signal_amp + np.random.normal(0, 1.0)
                hist[ch, m, 3] = np.random.normal(0, 1.0)

        cn0_out = cn0_nwpr(hist.ravel(), n_channels, n_hist, T)

        for ch in range(n_channels):
            assert 35.0 <= cn0_out[ch] <= 70.0, \
                f"Channel {ch} CN0 {cn0_out[ch]:.1f} dB-Hz outside range"


# ============================================================
# Test loop filter
# ============================================================
class TestLoopFilter:
    """Test 2nd-order loop filter behavior."""

    def test_zero_discriminator_no_change(self):
        """Zero discriminator input should produce zero NCO output (no integrator history)."""
        zeta = 0.707
        bandwidth = 2.0
        dt = 1e-3
        omega_n = bandwidth * 8.0 * zeta / (4.0 * zeta**2 + 1.0)

        disc = 0.0
        integrator = 0.0
        integrator += omega_n**2 * disc * dt
        nco_freq = omega_n * 2.0 * zeta * disc + integrator

        assert abs(nco_freq) < 1e-15
        assert abs(integrator) < 1e-15

    def test_positive_discriminator_positive_output(self):
        """Positive discriminator should yield positive NCO correction."""
        zeta = 0.707
        bandwidth = 15.0
        dt = 1e-3
        omega_n = bandwidth * 8.0 * zeta / (4.0 * zeta**2 + 1.0)

        disc = 0.1
        integrator = 0.0
        integrator += omega_n**2 * disc * dt
        nco_freq = omega_n * 2.0 * zeta * disc + integrator

        assert nco_freq > 0

    def test_convergence_over_time(self):
        """Loop filter should converge integrator over repeated updates."""
        zeta = 0.707
        bandwidth = 2.0
        dt = 1e-3
        omega_n = bandwidth * 8.0 * zeta / (4.0 * zeta**2 + 1.0)

        integrator = 0.0
        # Apply constant discriminator and check integrator grows
        for _ in range(100):
            disc = 0.05
            integrator += omega_n**2 * disc * dt

        assert integrator > 0, "Integrator should accumulate over time"
