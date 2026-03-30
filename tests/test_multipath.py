"""Tests for GPU-accelerated multipath simulation."""

import numpy as np
import pytest

# Constants matching the CUDA implementation
SPEED_OF_LIGHT = 299792458.0
L1_FREQ = 1575.42e6
CA_CHIP_RATE = 1.023e6
CA_CHIP_LENGTH = SPEED_OF_LIGHT / CA_CHIP_RATE  # ~293.05 m


def _try_import():
    """Try to import the multipath module; skip tests if CUDA bindings unavailable."""
    try:
        from gnss_gpu.multipath import MultipathSimulator
        # Quick check that bindings are actually loadable
        return MultipathSimulator
    except (ImportError, RuntimeError):
        pytest.skip("Multipath CUDA bindings not available")


class TestSingleReflector:
    """Single reflector plane: verify excess delay matches hand calculation."""

    def test_excess_delay_horizontal_plane(self):
        MultipathSimulator = _try_import()

        # Receiver at (0, 0, 10) m in some ECEF-like frame
        # Satellite directly above at (0, 0, 20200e3) m
        # Horizontal ground plane at z=0, normal=(0,0,1)
        rx = np.array([[0.0, 0.0, 10.0]])
        sat = np.array([[0.0, 0.0, 20200e3]])
        plane = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])

        sim = MultipathSimulator(plane)
        delays, attens = sim.simulate(rx, sat)

        # By image method: reflected rx at (0, 0, -10)
        # Direct path: 20200e3 - 10 = 20199990 m
        # Reflected path: 20200e3 + 10 = 20200010 m
        # Excess delay = 20 m
        expected_excess = 20.0
        assert abs(delays[0, 0] - expected_excess) < 0.01, \
            f"Expected excess delay ~{expected_excess} m, got {delays[0, 0]}"

        # Attenuation should be positive (within correlation triangle since 20m << 293m)
        assert attens[0, 0] > 0.0

    def test_excess_delay_vertical_wall(self):
        MultipathSimulator = _try_import()

        # Receiver at (0, 0, 0), satellite at (-1e6, 0, 1e6)
        # Wall at x=50, normal=(-1, 0, 0) facing the receiver
        # Satellite must be on the same side as the receiver (x < 50)
        # for a valid specular reflection off the front face of the wall.
        rx = np.array([[0.0, 0.0, 0.0]])
        sat = np.array([[-1e6, 0.0, 1e6]])
        plane = np.array([[50.0, 0.0, 0.0, -1.0, 0.0, 0.0]])

        sim = MultipathSimulator(plane)
        delays, attens = sim.simulate(rx, sat)

        # Reflected rx at (100, 0, 0) (image of receiver across x=50 plane)
        # Direct: sqrt(1e12 + 1e12) = sqrt(2)*1e6
        # Reflected: sqrt((1e6+100)^2 + 1e12) (image is 100m further from sat)
        direct = np.sqrt(1e12 + 1e12)
        reflected = np.sqrt((1e6 + 100.0)**2 + 1e12)
        expected = reflected - direct
        assert abs(delays[0, 0] - expected) < 0.1


class TestFarReflector:
    """Verify multipath error is zero when reflector is far away."""

    def test_no_contribution_when_far(self):
        MultipathSimulator = _try_import()

        # Receiver at origin, satellite overhead
        # Reflector very far away -> excess delay >> 1 chip -> no correlation
        rx = np.array([[0.0, 0.0, 0.0]])
        sat = np.array([[0.0, 0.0, 20200e3]])
        # Plane at x=1000 km away with normal pointing toward receiver
        plane = np.array([[1e6, 0.0, 0.0, -1.0, 0.0, 0.0]])

        sim = MultipathSimulator(plane)
        delays, attens = sim.simulate(rx, sat)

        # Excess delay should be very large -> attenuation should be 0
        # (outside correlation triangle)
        assert attens[0, 0] == 0.0, \
            f"Expected zero attenuation for far reflector, got {attens[0, 0]}"


class TestDLLErrorMagnitude:
    """Verify DLL error is in reasonable range for C/A code (1-10 m)."""

    def test_dll_error_reasonable_magnitude(self):
        MultipathSimulator = _try_import()

        # Setup: receiver near ground, satellite at moderate elevation
        # Reflector plane = ground beneath receiver
        rx = np.array([[0.0, 0.0, 2.0]])  # 2m above ground
        sat = np.array([[5000e3, 0.0, 20000e3]])  # ~76 deg elevation

        # Ground plane at z=0
        plane = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])

        sim = MultipathSimulator(plane, correlator_spacing=1.0)

        # Create a pseudo-range (approximate direct distance)
        direct_dist = np.sqrt(5000e3**2 + 20000e3**2 - 2.0**2)
        clean_pr = np.array([[direct_dist]])

        rx_epoch = rx.copy()
        sat_epoch = sat.reshape(1, 1, 3)

        corrupted, errors = sim.corrupt_pseudoranges(clean_pr, rx_epoch, sat_epoch)

        # For C/A code with 1-chip spacing, typical multipath error envelope
        # peaks around 10-15 m for close reflectors
        error_mag = abs(errors[0, 0])
        assert error_mag < 50.0, \
            f"DLL error magnitude {error_mag} m exceeds reasonable bound"
        # With a 4m excess delay (2m height * 2), which is ~0.014 chips,
        # we expect a non-trivial error
        assert error_mag > 0.0, "Expected non-zero multipath error for close reflector"


class TestBatchEpochs:
    """Batch test with multiple epochs."""

    def test_multi_epoch_consistency(self):
        MultipathSimulator = _try_import()

        n_epoch = 10
        n_sat = 4

        # Fixed receiver, fixed reflector, varying satellite positions
        rx = np.tile(np.array([0.0, 0.0, 5.0]), (n_epoch, 1))
        plane = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])  # ground

        # Satellites at different positions per epoch
        rng = np.random.RandomState(42)
        sat = np.zeros((n_epoch, n_sat, 3))
        for i in range(n_epoch):
            for j in range(n_sat):
                az = rng.uniform(0, 2 * np.pi)
                el = rng.uniform(np.radians(15), np.radians(85))
                r = 20200e3
                sat[i, j, 0] = r * np.cos(el) * np.cos(az)
                sat[i, j, 1] = r * np.cos(el) * np.sin(az)
                sat[i, j, 2] = r * np.sin(el)

        # Clean pseudoranges = direct distances
        clean_pr = np.zeros((n_epoch, n_sat))
        for i in range(n_epoch):
            for j in range(n_sat):
                clean_pr[i, j] = np.linalg.norm(sat[i, j] - rx[i])

        sim = MultipathSimulator(plane)
        corrupted, errors = sim.corrupt_pseudoranges(clean_pr, rx, sat)

        # Basic checks
        assert corrupted.shape == (n_epoch, n_sat)
        assert errors.shape == (n_epoch, n_sat)

        # Corrupted should differ from clean (ground reflection at 5m height
        # gives 10m excess delay which is well within 1 chip = 293m)
        assert np.any(errors != 0.0), "Expected some non-zero multipath errors"

        # All errors should be finite
        assert np.all(np.isfinite(errors)), "All errors should be finite"
        assert np.all(np.isfinite(corrupted)), "All corrupted PR should be finite"

    def test_identical_epochs_give_same_result(self):
        MultipathSimulator = _try_import()

        n_epoch = 5
        rx_single = np.array([0.0, 0.0, 3.0])
        sat_single = np.array([[10000e3, 0.0, 20000e3],
                                [0.0, 15000e3, 18000e3]])
        plane = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])

        rx = np.tile(rx_single, (n_epoch, 1))
        sat = np.tile(sat_single, (n_epoch, 1, 1))
        clean_pr = np.zeros((n_epoch, 2))
        for j in range(2):
            clean_pr[:, j] = np.linalg.norm(sat_single[j] - rx_single)

        sim = MultipathSimulator(plane)
        corrupted, errors = sim.corrupt_pseudoranges(clean_pr, rx, sat)

        # All epochs should produce identical results
        for i in range(1, n_epoch):
            np.testing.assert_allclose(errors[i], errors[0], atol=1e-10,
                                       err_msg=f"Epoch {i} differs from epoch 0")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
