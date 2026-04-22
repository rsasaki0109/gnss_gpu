"""Tests for wide-lane integer ambiguity resolution."""

from __future__ import annotations

import numpy as np
import pytest

from gnss_gpu.wide_lane import (
    C_LIGHT,
    L1_FREQ,
    L2_FREQ,
    LAMBDA_1,
    LAMBDA_2,
    LAMBDA_WL,
    WidelaneResolver,
    compute_n_wl_float,
)


# ---------------------------------------------------------------------------
# Constants sanity
# ---------------------------------------------------------------------------


def test_constants() -> None:
    assert abs(LAMBDA_WL - C_LIGHT / (L1_FREQ - L2_FREQ)) < 1e-6
    assert abs(LAMBDA_1 - C_LIGHT / L1_FREQ) < 1e-6
    assert abs(LAMBDA_2 - C_LIGHT / L2_FREQ) < 1e-6
    # Wide-lane wavelength is approximately 0.862 m.
    assert 0.86 < LAMBDA_WL < 0.87


# ---------------------------------------------------------------------------
# compute_n_wl_float
# ---------------------------------------------------------------------------


class TestComputeNwlFloat:
    """Test single-epoch float N_wl computation."""

    def test_known_integer(self) -> None:
        """Construct observations that yield an exact integer N_wl."""
        true_n_wl = 7
        # Choose arbitrary carrier phase values.
        L1_cycles = 100_000.0
        L2_cycles = L1_cycles - true_n_wl  # so L1 - L2 = true_n_wl
        # Make P_nl / lambda_wl = 0, i.e. P_nl = 0.
        # P_nl = (f1*P1 + f2*P2) / (f1+f2) = 0  =>  f1*P1 = -f2*P2
        # Simplest: P1 = P2 = 0.
        P1_m = 0.0
        P2_m = 0.0
        n_wl = compute_n_wl_float(L1_cycles, L2_cycles, P1_m, P2_m)
        assert abs(n_wl - true_n_wl) < 1e-10

    def test_with_pseudorange(self) -> None:
        """Verify formula with non-zero pseudoranges."""
        L1 = 120_000_000.0
        L2 = 93_000_000.0
        P1 = 22_000_000.0
        P2 = 22_100_000.0
        P_nl = (L1_FREQ * P1 + L2_FREQ * P2) / (L1_FREQ + L2_FREQ)
        expected = L1 - L2 - P_nl / LAMBDA_WL
        got = compute_n_wl_float(L1, L2, P1, P2)
        assert abs(got - expected) < 1e-6

    def test_nan_input(self) -> None:
        """NaN input propagates to NaN output."""
        assert not np.isfinite(compute_n_wl_float(float("nan"), 0, 0, 0))


# ---------------------------------------------------------------------------
# WidelaneResolver — convergence
# ---------------------------------------------------------------------------


class TestWidelaneResolverConvergence:
    """Feed consistent data and verify integer fix."""

    @staticmethod
    def _make_obs(true_n_wl: int, rng: np.random.Generator, code_noise_m: float = 0.5):
        """Return (L1, L2, P1, P2) whose float N_wl ≈ true_n_wl + noise."""
        # True geometry: pick a "true range" and build observations.
        true_range = 22_000_000.0  # meters (arbitrary)
        L1_cycles = true_range / LAMBDA_1 + true_n_wl * 0  # carrier (no N1 needed here)
        L2_cycles = L1_cycles - true_n_wl  # exact wl ambiguity relationship
        # Code observations with noise so that P_nl / lambda_wl ≈ 0 + noise.
        P1 = rng.normal(0.0, code_noise_m)
        P2 = rng.normal(0.0, code_noise_m)
        return L1_cycles, L2_cycles, P1, P2

    def test_converges_after_enough_epochs(self) -> None:
        rng = np.random.default_rng(42)
        wr = WidelaneResolver(min_epochs=5, max_std=0.4)
        true_n_wl = 12
        prn = 7

        for _ in range(10):
            L1, L2, P1, P2 = self._make_obs(true_n_wl, rng, code_noise_m=0.3)
            wr.update(prn, L1, L2, P1, P2)

        fixed = wr.get_fixed_ambiguity(prn)
        assert fixed is not None
        assert fixed == true_n_wl

    def test_does_not_converge_before_min_epochs(self) -> None:
        rng = np.random.default_rng(99)
        wr = WidelaneResolver(min_epochs=5, max_std=0.4)
        true_n_wl = -3
        prn = 15

        for _ in range(4):
            L1, L2, P1, P2 = self._make_obs(true_n_wl, rng, code_noise_m=0.1)
            wr.update(prn, L1, L2, P1, P2)

        assert wr.get_fixed_ambiguity(prn) is None

    def test_noisy_data_does_not_converge(self) -> None:
        """High code noise should prevent convergence (std > threshold)."""
        rng = np.random.default_rng(7)
        wr = WidelaneResolver(min_epochs=5, max_std=0.4)
        prn = 22

        for _ in range(20):
            # Very large code noise → std(N_wl) >> 0.4
            L1, L2, P1, P2 = self._make_obs(5, rng, code_noise_m=50.0)
            wr.update(prn, L1, L2, P1, P2)

        assert wr.get_fixed_ambiguity(prn) is None

    def test_multiple_satellites_independent(self) -> None:
        rng = np.random.default_rng(0)
        wr = WidelaneResolver(min_epochs=5, max_std=0.4)

        for prn, true_n in [(1, 10), (5, -4), (31, 0)]:
            for _ in range(10):
                L1, L2, P1, P2 = self._make_obs(true_n, rng, code_noise_m=0.2)
                wr.update(prn, L1, L2, P1, P2)

            assert wr.get_fixed_ambiguity(prn) == true_n


# ---------------------------------------------------------------------------
# WidelaneResolver — pseudorange output
# ---------------------------------------------------------------------------


class TestWidelanePseudorange:
    """Test get_widelane_pseudorange after fix."""

    def test_returns_none_before_fix(self) -> None:
        wr = WidelaneResolver()
        assert wr.get_widelane_pseudorange(1, 100_000.0, 80_000.0) is None

    def test_pseudorange_after_fix(self) -> None:
        """Verify pseudorange formula: rho = (phi_wl - N_wl) * lambda_wl."""
        np.random.default_rng(123)
        wr = WidelaneResolver(min_epochs=5, max_std=0.4)
        prn = 3
        true_n_wl = 8

        # Feed clean data to fix.
        L1_base = 120_000_000.0
        L2_base = L1_base - true_n_wl
        for _ in range(10):
            wr.update(prn, L1_base, L2_base, 0.0, 0.0)

        assert wr.get_fixed_ambiguity(prn) == true_n_wl

        # Now get pseudorange with specific carrier phases.
        L1_test = 120_000_500.0
        L2_test = L1_test - true_n_wl + 0.3  # small fractional offset
        rho = wr.get_widelane_pseudorange(prn, L1_test, L2_test)
        assert rho is not None

        # Expected value.
        phi_wl = (L1_FREQ * L1_test - L2_FREQ * L2_test) / (L1_FREQ - L2_FREQ)
        expected = (phi_wl - true_n_wl) * LAMBDA_WL
        assert abs(rho - expected) < 1e-6


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------


class TestReset:
    def test_reset_single(self) -> None:
        wr = WidelaneResolver(min_epochs=1, max_std=999.0)
        wr.update(1, 100.0, 93.0, 0.0, 0.0)
        assert wr.get_fixed_ambiguity(1) is not None
        wr.reset(1)
        assert wr.get_fixed_ambiguity(1) is None

    def test_reset_all(self) -> None:
        wr = WidelaneResolver(min_epochs=1, max_std=999.0)
        wr.update(1, 100.0, 93.0, 0.0, 0.0)
        wr.update(2, 200.0, 193.0, 0.0, 0.0)
        wr.reset()
        assert wr.get_fixed_ambiguity(1) is None
        assert wr.get_fixed_ambiguity(2) is None
