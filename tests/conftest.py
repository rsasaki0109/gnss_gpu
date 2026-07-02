"""Shared pytest fixtures for GNSS GPU tests."""

import pytest

from gnss_fixtures import generate_satellites, sample_position_ecef


@pytest.fixture
def satellites():
    """Default synthetic satellite constellation (4 sats, seed=42)."""
    return generate_satellites()


@pytest.fixture
def sample_position():
    """Simple finite ECEF position for filter initialization."""
    return sample_position_ecef()
