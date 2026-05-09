from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from experiments.gsdc2023_bridge_config import BridgeConfig
from experiments.gsdc2023_observation_matrix import repair_baseline_wls
from experiments.gsdc2023_raw_bridge import (
    DEFAULT_ROOT,
    build_trip_arrays,
    fit_state_with_clock_bias,
    run_wls,
    validate_raw_gsdc2023_trip,
    weighted_mse,
)
from experiments.gsdc2023_validation_context import max_epochs_for_build


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _pixel6pro_cluster_split_trip() -> Path:
    return (
        _repo_root().parent
        / "ref/gsdc2023/kaggle_smartphone_decimeter_2023/sdc2023/test"
        / "2021-11-05-18-28-us-ca-mtv-m/pixel6pro"
    )


def _pixel6pro_2023_raw_proxy_trip() -> str:
    return "test/2023-05-23-22-16-us-ca-mtv-ie2/pixel6pro"


def _baseline_pr_mse(trip_dir: Path, *, apply_observation_mask: bool) -> tuple[float, object]:
    batch = build_trip_arrays(
        trip_dir,
        max_epochs=max_epochs_for_build(0),
        start_epoch=0,
        constellation_type=1,
        signal_type="GPS_L1_CA",
        weight_mode="cn0",
        multi_gnss=False,
        use_tdcp=False,
        apply_observation_mask=apply_observation_mask,
    )
    _state, sse, weight_sum, _per_epoch = fit_state_with_clock_bias(
        batch.sat_ecef,
        batch.pseudorange,
        batch.weights,
        batch.kaggle_wls,
        sys_kind=batch.sys_kind,
        n_clock=batch.n_clock,
    )
    return weighted_mse(sse, weight_sum), batch


def _step_max_m(xyz: np.ndarray) -> float:
    step = np.linalg.norm(np.diff(np.asarray(xyz, dtype=np.float64).reshape(-1, 3), axis=0), axis=1)
    return float(np.max(step))


@pytest.mark.slow
def test_real_pixel6pro_observation_mask_suppresses_cluster_split_pr_outliers() -> None:
    trip_dir = _pixel6pro_cluster_split_trip()
    if not (trip_dir / "device_gnss.csv").is_file():
        pytest.skip(f"GSDC2023 raw bridge fixture is not available: {trip_dir}")

    unmasked_mse, unmasked_batch = _baseline_pr_mse(trip_dir, apply_observation_mask=False)
    masked_mse, masked_batch = _baseline_pr_mse(trip_dir, apply_observation_mask=True)

    assert unmasked_batch.times_ms.size == 1446
    assert masked_batch.times_ms.size == unmasked_batch.times_ms.size
    assert np.count_nonzero(unmasked_batch.weights > 0.0) == 10467
    assert masked_batch.observation_mask_count >= 700
    assert masked_batch.residual_mask_count > 0
    assert masked_batch.pseudorange_doppler_mask_count > 0
    assert unmasked_mse > 1.0e6
    assert masked_mse < 100.0
    assert masked_mse < unmasked_mse * 1.0e-4


@pytest.mark.slow
def test_real_pixel6pro_raw_wls_repair_removes_single_epoch_position_spike() -> None:
    trip_dir = _pixel6pro_cluster_split_trip()
    if not (trip_dir / "device_gnss.csv").is_file():
        pytest.skip(f"GSDC2023 raw bridge fixture is not available: {trip_dir}")

    batch = build_trip_arrays(
        trip_dir,
        max_epochs=max_epochs_for_build(0),
        start_epoch=0,
        constellation_type=1,
        signal_type="GPS_L1_CA",
        weight_mode="sin2el",
        multi_gnss=True,
        use_tdcp=True,
        apply_observation_mask=True,
    )
    raw_wls = run_wls(
        batch.sat_ecef,
        batch.pseudorange,
        batch.weights,
        sys_kind=batch.sys_kind,
        n_clock=batch.n_clock,
        fallback_xyz=batch.kaggle_wls,
    )
    repaired_xyz = repair_baseline_wls(batch.times_ms, raw_wls[:, :3])

    assert _step_max_m(raw_wls[:, :3]) > 50_000.0
    assert _step_max_m(repaired_xyz) < 100.0


@pytest.mark.slow
def test_real_pixel6pro_2023_gated_rejects_raw_backed_fgo_despite_lower_pr_mse() -> None:
    trip = _pixel6pro_2023_raw_proxy_trip()
    trip_dir = DEFAULT_ROOT / trip
    if not (trip_dir / "device_gnss.csv").is_file():
        pytest.skip(f"GSDC2023 raw bridge fixture is not available: {trip_dir}")

    result = validate_raw_gsdc2023_trip(
        DEFAULT_ROOT,
        trip,
        max_epochs=400,
        config=BridgeConfig(
            position_source="gated",
            chunk_epochs=200,
            apply_observation_mask=True,
            multi_gnss=True,
            fgo_iters=8,
        ),
    )

    assert result.n_epochs == 400
    assert result.fgo_iters == 0
    assert result.vd_seed_guard_skipped_segments == 2
    assert result.vd_seed_guard_skipped_epochs == 400
    assert result.fgo_mse_pr == pytest.approx(result.raw_wls_mse_pr)
    assert result.raw_wls_mse_pr < result.baseline_mse_pr * 0.6
    assert result.selected_mse_pr == pytest.approx(result.baseline_mse_pr)
    assert result.selected_source_counts["baseline"] == 400

    records = result.chunk_selection_records or []
    assert len(records) == 2
    assert {record["gated_source"] for record in records} == {"baseline"}
    assert all(
        record["candidates"]["fgo"]["mse_pr"] < record["candidates"]["baseline"]["mse_pr"]
        for record in records
    )
