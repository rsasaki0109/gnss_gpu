"""Unit tests for the raw-bridge DD-carrier anchor coverage helper.

Covers the thin adapter in ``experiments.gsdc2023_raw_bridge`` that bridges the
DD-carrier stats payload and the pure ratio helper in
``experiments.gsdc2023_chunk_selection``.
"""

from __future__ import annotations

from experiments.gsdc2023_bridge_config import BridgeConfig
from experiments.gsdc2023_raw_bridge import _dd_carrier_anchor_coverage


def _enabled_config(**overrides: float) -> BridgeConfig:
    """Build a BridgeConfig with DD-carrier FGO turned on for gate tests."""

    return BridgeConfig(dd_carrier_fgo_enabled=True, **overrides)


def test_dd_carrier_anchor_coverage_returns_none_when_disabled() -> None:
    config = BridgeConfig()  # default: dd_carrier_fgo_enabled=False
    assert _dd_carrier_anchor_coverage(
        config, {"accepted_anchor_epochs": 80}, n_epoch=100
    ) is None


def test_dd_carrier_anchor_coverage_returns_ratio_when_enabled() -> None:
    config = _enabled_config()
    coverage = _dd_carrier_anchor_coverage(
        config, {"accepted_anchor_epochs": 75}, n_epoch=150
    )
    assert coverage == 0.5


def test_dd_carrier_anchor_coverage_handles_missing_stat_key() -> None:
    config = _enabled_config()
    assert _dd_carrier_anchor_coverage(config, {}, n_epoch=100) == 0.0


def test_dd_carrier_anchor_coverage_handles_none_stat_value() -> None:
    config = _enabled_config()
    assert _dd_carrier_anchor_coverage(
        config, {"accepted_anchor_epochs": None}, n_epoch=100
    ) == 0.0


def test_dd_carrier_anchor_coverage_returns_none_for_empty_trip() -> None:
    config = _enabled_config()
    assert _dd_carrier_anchor_coverage(
        config, {"accepted_anchor_epochs": 0}, n_epoch=0
    ) is None
