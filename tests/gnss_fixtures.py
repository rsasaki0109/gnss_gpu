"""Shared test helpers for GNSS GPU wrapper and integration tests."""

import numpy as np


def generate_satellites(n_sat=4, seed=42):
    """Generate synthetic satellite ECEF positions on a spherical shell."""
    rng = np.random.RandomState(seed)
    R_orbit = 26_571_000.0
    theta = rng.uniform(0, 2 * np.pi, n_sat)
    phi = rng.uniform(-np.pi / 3, np.pi / 3, n_sat)
    sat = np.zeros((n_sat, 3))
    sat[:, 0] = R_orbit * np.cos(phi) * np.cos(theta)
    sat[:, 1] = R_orbit * np.cos(phi) * np.sin(theta)
    sat[:, 2] = R_orbit * np.sin(phi)
    return sat


def sample_position_ecef():
    """Return a simple finite ECEF position for initialization tests."""
    return np.array([1.0, 2.0, 3.0])
