from __future__ import annotations

import numpy as np
import pytest

from experiments.build_wp29_imu_heading_route_seed_trace import (
    build_endpoint_closed_route,
    local_enu_basis,
    select_pre_jump_anchor,
    select_pre_jump_seed_support_anchor,
)


def test_select_pre_jump_anchor_uses_guard_step() -> None:
    rows = []
    for i, epoch in enumerate(range(10, 40, 5)):
        residual = 3.0 if epoch == 35 else 0.1
        rows.extend(
            [
                {
                    "epoch": str(epoch),
                    "selected": "1",
                    "max_marginal_rank": "1",
                    "max_marginal_relative": "0",
                    "next_selected_transition_residual_m": str(residual),
                },
                {
                    "epoch": str(epoch),
                    "selected": "0",
                    "max_marginal_rank": "2",
                    "max_marginal_relative": "-8",
                    "next_selected_transition_residual_m": "0",
                },
            ]
        )
    anchor, metrics = select_pre_jump_anchor(rows, min_epoch=10, max_epoch=40)
    assert anchor == 30
    assert metrics["jump_epoch"] == 35


def test_select_pre_jump_anchor_rejects_weak_history() -> None:
    rows = []
    for epoch in range(10, 40, 5):
        rows.extend(
            [
                {
                    "epoch": str(epoch), "selected": "1", "max_marginal_rank": "1",
                    "max_marginal_relative": "0", "next_selected_transition_residual_m": "3",
                },
                {
                    "epoch": str(epoch), "selected": "0", "max_marginal_rank": "2",
                    "max_marginal_relative": "-1", "next_selected_transition_residual_m": "0",
                },
            ]
        )
    with pytest.raises(RuntimeError):
        select_pre_jump_anchor(rows, min_epoch=10, max_epoch=40)


def test_select_pre_jump_seed_support_anchor_uses_guard() -> None:
    rows = []
    for epoch, support in [(10, 1), (15, 2), (20, 2), (25, 3)]:
        rows.append(
            {
                "epoch": str(epoch),
                "selected": "1",
                "current_seed_support": str(support),
                "next_selected_transition_residual_m": "3" if epoch == 25 else "0.1",
            }
        )
    anchor, metrics = select_pre_jump_seed_support_anchor(
        rows, min_epoch=10, max_epoch=30
    )
    assert anchor == 20
    assert metrics["history_min_seed_support"] == 2


def test_endpoint_closed_route_closes_known_turn() -> None:
    n = 31
    center = np.array([-3.96e6, 3.35e6, 3.70e6])
    basis = local_enu_basis(center)
    increments = np.full(n - 1, np.deg2rad(2.0))
    cumulative = np.r_[0.0, np.cumsum(increments)]
    heading = -cumulative[1:] + 0.4
    speed = np.ones(n)
    enu_steps = np.column_stack([np.sin(heading), np.cos(heading), np.zeros(n - 1)])
    positions = np.tile(center, (n, 1))
    positions[1:] = center + np.cumsum(enu_steps @ basis, axis=0)
    doppler = np.zeros((n, 3))
    doppler[1:] = enu_steps @ basis
    route, metrics = build_endpoint_closed_route(
        positions, doppler, speed, increments, start=0, end=n - 1,
        gyro_bias_radps=0.0, epoch_dt_s=np.ones(n - 1),
    )
    assert np.linalg.norm(route[-1] - positions[-1]) < 1e-8
    assert metrics["gyro_sign"] == -1.0
    assert np.max(np.linalg.norm(route - positions, axis=1)) < 1e-8


def test_endpoint_closed_route_uses_doppler_speed_for_tdcp_gap() -> None:
    n = 31
    center = np.array([-3.96e6, 3.35e6, 3.70e6])
    basis = local_enu_basis(center)
    increments = np.full(n - 1, np.deg2rad(2.0))
    cumulative = np.r_[0.0, np.cumsum(increments)]
    heading = -cumulative[1:] + 0.4
    enu_steps = np.column_stack([np.sin(heading), np.cos(heading), np.zeros(n - 1)])
    positions = np.tile(center, (n, 1))
    positions[1:] = center + np.cumsum(enu_steps @ basis, axis=0)
    doppler = np.zeros((n, 3))
    doppler[1:] = enu_steps @ basis
    tdcp_speed = np.ones(n)
    tdcp_speed[12:18] = np.nan
    route, metrics = build_endpoint_closed_route(
        positions,
        doppler,
        tdcp_speed,
        increments,
        start=0,
        end=n - 1,
        gyro_bias_radps=0.0,
        epoch_dt_s=np.ones(n - 1),
    )
    assert np.isfinite(route).all()
    assert metrics["endpoint_error_m"] < 1e-8


def test_endpoint_closed_route_uses_tdcp_speed_for_doppler_gap() -> None:
    n = 31
    center = np.array([-3.96e6, 3.35e6, 3.70e6])
    basis = local_enu_basis(center)
    increments = np.full(n - 1, np.deg2rad(2.0))
    cumulative = np.r_[0.0, np.cumsum(increments)]
    heading = -cumulative[1:] + 0.4
    enu_steps = np.column_stack([np.sin(heading), np.cos(heading), np.zeros(n - 1)])
    positions = np.tile(center, (n, 1))
    positions[1:] = center + np.cumsum(enu_steps @ basis, axis=0)
    doppler = np.zeros((n, 3))
    doppler[1:] = enu_steps @ basis
    doppler[12:18] = np.nan
    route, metrics = build_endpoint_closed_route(
        positions,
        doppler,
        np.ones(n),
        increments,
        start=0,
        end=n - 1,
        gyro_bias_radps=0.0,
        epoch_dt_s=np.ones(n - 1),
    )
    assert np.isfinite(route).all()
    assert metrics["tdcp_only_speed_intervals"] == 6
    assert metrics["endpoint_error_m"] < 1e-8


def test_endpoint_closed_route_accepts_interval_gyro_bias_profile() -> None:
    n = 31
    center = np.array([-3.96e6, 3.35e6, 3.70e6])
    basis = local_enu_basis(center)
    true_increments = np.full(n - 1, np.deg2rad(2.0))
    bias = np.linspace(np.deg2rad(-0.1), np.deg2rad(0.1), n - 1)
    measured_increments = true_increments + bias
    cumulative = np.r_[0.0, np.cumsum(true_increments)]
    heading = -cumulative[1:] + 0.4
    enu_steps = np.column_stack([np.sin(heading), np.cos(heading), np.zeros(n - 1)])
    positions = np.tile(center, (n, 1))
    positions[1:] = center + np.cumsum(enu_steps @ basis, axis=0)
    doppler = np.zeros((n, 3))
    doppler[1:] = enu_steps @ basis
    route, metrics = build_endpoint_closed_route(
        positions,
        doppler,
        np.ones(n),
        measured_increments,
        start=0,
        end=n - 1,
        gyro_bias_radps=bias,
        epoch_dt_s=np.ones(n - 1),
    )
    assert metrics["doppler_heading_p95_deg"] < 1e-8
    assert np.max(np.linalg.norm(route - positions, axis=1)) < 1e-8


def test_endpoint_closed_route_piecewise_doppler_corrects_heading_drift() -> None:
    n = 101
    center = np.array([-3.96e6, 3.35e6, 3.70e6])
    basis = local_enu_basis(center)
    true_increments = np.full(n - 1, np.deg2rad(1.0))
    drifting_bias = np.linspace(0.0, np.deg2rad(0.2), n - 1)
    measured = true_increments + drifting_bias
    cumulative = np.r_[0.0, np.cumsum(true_increments)]
    heading = -cumulative[1:] + 0.4
    enu_steps = np.column_stack([np.sin(heading), np.cos(heading), np.zeros(n - 1)])
    positions = np.tile(center, (n, 1))
    positions[1:] = center + np.cumsum(enu_steps @ basis, axis=0)
    doppler = np.zeros((n, 3))
    doppler[1:] = enu_steps @ basis
    route, metrics = build_endpoint_closed_route(
        positions,
        doppler,
        np.ones(n),
        measured,
        start=0,
        end=n - 1,
        gyro_bias_radps=0.0,
        epoch_dt_s=np.ones(n - 1),
        heading_correction_stride_epochs=20,
    )
    assert metrics["heading_correction_knots"] == 5
    assert metrics["doppler_heading_p95_deg"] < 2.0
    assert np.max(np.linalg.norm(route - positions, axis=1)) < 1.0
