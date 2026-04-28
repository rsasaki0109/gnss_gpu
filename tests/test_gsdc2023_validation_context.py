from __future__ import annotations

from pathlib import Path

import pytest

from experiments.gsdc2023_bridge_config import BridgeConfig
from experiments.gsdc2023_validation_context import (
    build_raw_trip_validation_context,
    effective_validation_config,
    max_epochs_for_build,
    outlier_refinement_config,
    require_base_correction_ready,
)


def test_effective_validation_config_applies_phone_family_overrides() -> None:
    cfg = BridgeConfig(position_source="auto", multi_gnss=True)

    effective = effective_validation_config("test/course/xiaomimi8", cfg)

    assert effective.position_source == "raw_wls"
    assert effective.multi_gnss is False
    assert cfg.position_source == "auto"
    assert cfg.multi_gnss is True


def test_effective_validation_config_returns_same_config_when_unchanged() -> None:
    cfg = BridgeConfig(position_source="gated", multi_gnss=True)

    assert effective_validation_config("test/course/pixel5", cfg) is cfg


def test_require_base_correction_ready_raises_with_status() -> None:
    cfg = BridgeConfig(apply_base_correction=True)

    with pytest.raises(RuntimeError, match="missing_base_obs"):
        require_base_correction_ready(cfg, {"base_correction_ready": False, "base_correction_status": "missing_base_obs"})


def test_require_base_correction_ready_ignores_preflight_when_not_requested() -> None:
    require_base_correction_ready(
        BridgeConfig(apply_base_correction=False),
        {"base_correction_ready": False, "base_correction_status": "missing_base_obs"},
    )


def test_build_raw_trip_validation_context_uses_injected_audit_and_effective_config(tmp_path) -> None:
    calls: list[tuple[Path, str]] = []

    def audit_fn(data_root: Path, trip: str) -> dict[str, object]:
        calls.append((data_root, trip))
        return {"base_correction_ready": True}

    context = build_raw_trip_validation_context(
        tmp_path,
        "test/course/mi8",
        BridgeConfig(multi_gnss=True, position_source="auto"),
        parity_audit_fn=audit_fn,
    )

    assert calls == [(tmp_path, "test/course/mi8")]
    assert context.parity_audit == {"base_correction_ready": True}
    assert context.trip_dir == tmp_path / "test/course/mi8"
    assert context.config.multi_gnss is False
    assert context.config.position_source == "raw_wls"


def test_build_raw_trip_validation_context_uses_default_config(tmp_path) -> None:
    context = build_raw_trip_validation_context(
        tmp_path,
        "test/course/pixel5",
        None,
        parity_audit_fn=lambda _root, _trip: {"base_correction_ready": True},
    )

    assert context.config == BridgeConfig()


def test_max_epochs_for_build_matches_raw_bridge_unbounded_sentinel() -> None:
    assert max_epochs_for_build(200) == 200
    assert max_epochs_for_build(0) == 1_000_000_000
    assert max_epochs_for_build(-1) == 1_000_000_000


def test_outlier_refinement_config_only_for_large_auto_or_gated_errors() -> None:
    cfg = BridgeConfig(position_source="gated", chunk_epochs=200)

    refined = outlier_refinement_config(cfg, 1200.0)

    assert refined is not None
    assert refined.chunk_epochs == 30
    assert cfg.chunk_epochs == 200
    assert outlier_refinement_config(cfg, 999.0) is None
    assert outlier_refinement_config(BridgeConfig(position_source="raw_wls", chunk_epochs=200), 5000.0) is None
    assert outlier_refinement_config(BridgeConfig(position_source="auto", chunk_epochs=30), 5000.0) is None
