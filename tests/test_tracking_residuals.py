import numpy as np
import pytest

try:
    from gnss_gpu._gnss_gpu_tracking import batch_correlate  # noqa: F401

    _HAS_GPU = True
except Exception:
    _HAS_GPU = False

from gnss_gpu.validation.tracking_residuals import (
    CA_CHIP_M,
    CA_CODE_LENGTH,
    cn0_from_prompt,
    code_phase_to_residual_m,
    convergence_residual,
    correlations_to_discriminator,
    discriminator_zero_crossing,
    eml_discriminator,
    generate_ca_code,
    generate_ca_signal,
    generate_multipath_signal,
    settled_value,
    wrap_code_phase_chips,
)


def test_wrap_code_phase_chips_scalar_and_vector():
    assert wrap_code_phase_chips(0.0) == 0.0
    assert wrap_code_phase_chips(1022.0) == -1.0
    assert wrap_code_phase_chips(-1.0) == -1.0
    assert wrap_code_phase_chips(512.0) == -511.0

    values = np.array([0.0, 1022.0, -1.0, 512.0])
    expected = np.array([0.0, -1.0, -1.0, -511.0])
    np.testing.assert_allclose(wrap_code_phase_chips(values), expected)


def test_code_phase_to_residual_m():
    reference = 100.0

    assert code_phase_to_residual_m(reference, reference) == 0.0
    assert code_phase_to_residual_m(reference + 1.0, reference) == pytest.approx(CA_CHIP_M)

    residual = code_phase_to_residual_m(0.2, 1022.9)
    assert residual == pytest.approx(0.3 * CA_CHIP_M)


def test_eml_discriminator_sign_and_zero_cases():
    assert eml_discriminator(1.0, 0.0, 0.0, 0.0, 1.0, 0.0) == pytest.approx(0.0)

    positive = eml_discriminator(2.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    negative = eml_discriminator(1.0, 0.0, 0.0, 0.0, 2.0, 0.0)
    assert positive > 0.0
    assert negative < 0.0

    positive_raw = eml_discriminator(2.0, 0.0, 0.0, 0.0, 1.0, 0.0, normalize=False)
    negative_raw = eml_discriminator(1.0, 0.0, 0.0, 0.0, 2.0, 0.0, normalize=False)
    assert np.sign(positive_raw) == np.sign(positive)
    assert np.sign(negative_raw) == np.sign(negative)

    assert eml_discriminator(0.0, 0.0, 0.0, 0.0, 0.0, 0.0) == 0.0


def test_correlations_to_discriminator_matches_channel_calls():
    correlations = np.array(
        [
            [2.0, 0.0, 10.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 10.0, 0.0, 3.0, 0.0],
        ]
    )

    result = correlations_to_discriminator(correlations)
    expected = np.array(
        [
            eml_discriminator(2.0, 0.0, 10.0, 0.0, 1.0, 0.0),
            eml_discriminator(1.0, 0.0, 10.0, 0.0, 3.0, 0.0),
        ]
    )

    assert result.shape == (2,)
    np.testing.assert_allclose(result, expected)


def test_cn0_from_prompt_monotonic_power_and_noise():
    low_power = cn0_from_prompt(1.0, 0.0, noise_power=1.0, integration_time=0.001)
    high_power = cn0_from_prompt(2.0, 0.0, noise_power=1.0, integration_time=0.001)
    high_noise = cn0_from_prompt(2.0, 0.0, noise_power=2.0, integration_time=0.001)

    assert high_power > low_power
    assert high_noise < high_power


def test_settled_value():
    assert settled_value([3.0, 3.0, 3.0, 3.0]) == pytest.approx(3.0)
    assert settled_value([0.0, 1.0, 2.0, 3.0], settle_fraction=0.5) == pytest.approx(2.5)
    assert settled_value([42.0]) == pytest.approx(42.0)

    with pytest.raises(ValueError):
        settled_value([])


def test_convergence_residual():
    reference = 200.0
    history = np.array([198.0, 199.0, 200.45, 200.55])

    result = convergence_residual(history, reference, settle_fraction=0.5)

    assert set(result) == {"steady_code_phase", "residual_m", "residual_chips"}
    assert result["steady_code_phase"] == pytest.approx(reference + 0.5)
    assert result["residual_chips"] == pytest.approx(0.5)
    assert result["residual_m"] == pytest.approx(0.5 * CA_CHIP_M)


def test_discriminator_zero_crossing_falling_edge():
    offsets = np.linspace(-1.0, 1.0, 201)
    # Falling S-curve crossing zero at +0.2 chip.
    disc = -(offsets - 0.2)
    crossing = discriminator_zero_crossing(offsets, disc)
    assert crossing == pytest.approx(0.2, abs=1e-6)

    # No falling crossing -> None.
    assert discriminator_zero_crossing(offsets, np.ones_like(offsets)) is None

    with pytest.raises(ValueError):
        discriminator_zero_crossing([0.0], [0.0])


def test_discriminator_zero_crossing_picks_nearest_to_zero():
    offsets = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    # Falling crossings near -1.5 and +1.5; nearest-to-zero is... both equal,
    # use an asymmetric S-curve with a single on-time crossing at +0.5.
    disc = np.array([1.0, 1.0, 1.0, -1.0, -1.0])
    crossing = discriminator_zero_crossing(offsets, disc)
    assert crossing == pytest.approx(0.5, abs=1e-9)


def test_generate_ca_code_properties():
    code = generate_ca_code(5)
    assert code.shape == (CA_CODE_LENGTH,)
    assert set(np.unique(code)).issubset({-1.0, 1.0})
    # C/A codes are nearly balanced (exactly 512 ones, 511 minus-ones).
    assert int(np.sum(code > 0)) == 512


def test_generate_multipath_signal_is_superposition():
    prn = 5
    n = 4092
    sf, iff = 4.092e6, 4.092e6
    direct = {"code_phase": 200.0, "amplitude": 1.0}
    replica = {"code_phase": 200.5, "amplitude": 0.5}

    s_direct = generate_ca_signal(prn, 200.0, iff, n, sf, iff, amplitude=1.0)
    s_replica = generate_ca_signal(prn, 200.5, iff, n, sf, iff, amplitude=0.5)
    composite = generate_multipath_signal([direct, replica], prn, n, sf, iff)

    np.testing.assert_allclose(composite, s_direct + s_replica, rtol=0.0, atol=1e-9)


@pytest.mark.skipif(not _HAS_GPU, reason="GPU tracking extension not available")
def test_measure_multipath_bias_envelope_gpu():
    from gnss_gpu.tracking import TrackingConfig
    from gnss_gpu.validation.tracking_residuals import measure_multipath_bias

    prn = 5
    cfg = TrackingConfig(
        sampling_freq=16.368e6, intermediate_freq=4.092e6,
        integration_time=1e-3, correlator_spacing=0.5,
    )

    # LOS only -> zero bias.
    los = measure_multipath_bias([{"code_phase": 200.0, "amplitude": 1.0}], prn,
                                 config=cfg, n_points=401)
    assert los["bias_chips"] is not None
    assert abs(los["residual_m"]) < 5.0

    # In-phase multipath at half a chip -> positive (delayed) bias of tens of m.
    mp = measure_multipath_bias(
        [{"code_phase": 200.0, "amplitude": 1.0},
         {"code_phase": 200.5, "amplitude": 0.5}],
        prn, config=cfg, n_points=401)
    assert mp["residual_m"] > 10.0

    # Replica well beyond the correlator reach -> back to ~zero bias.
    far = measure_multipath_bias(
        [{"code_phase": 200.0, "amplitude": 1.0},
         {"code_phase": 201.6, "amplitude": 0.5}],
        prn, config=cfg, n_points=401)
    assert abs(far["residual_m"]) < 5.0
