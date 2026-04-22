import numpy as np

from gnss_gpu.smoother_postprocess import (
    _apply_smoother_tail_guard,
    _apply_smoother_widelane_forward_guard,
    _apply_stop_segment_constant_position,
)
from gnss_gpu.local_fgo import UndiffPseudorangeEpoch
from gnss_gpu.stop_segment_static import (
    StaticStopSegmentConfig,
    apply_static_stop_segment_gnss,
)


def test_smoother_widelane_forward_guard_only_replaces_wl_epochs():
    smoothed = np.array(
        [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [20.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    forward = np.array(
        [[1.0, 0.0, 0.0], [11.0, 0.0, 0.0], [22.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    rows = [
        {"used_widelane": True},
        {"used_widelane": False},
        {"used_widelane": True},
    ]

    guarded, n_applied = _apply_smoother_widelane_forward_guard(
        smoothed,
        forward,
        rows,
        min_shift_m=1.5,
    )

    assert n_applied == 1
    np.testing.assert_allclose(guarded[0], smoothed[0])
    np.testing.assert_allclose(guarded[1], smoothed[1])
    np.testing.assert_allclose(guarded[2], forward[2])
    assert rows[0]["widelane_forward_guard_applied"] is False
    assert rows[1]["widelane_forward_guard_applied"] is False
    assert rows[2]["widelane_forward_guard_applied"] is True


def test_smoother_tail_guard_expands_from_moving_anchor_only():
    smoothed = np.array(
        [
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
            [30.0, 0.0, 0.0],
            [40.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    forward = np.array(
        [
            [0.0, 0.0, 0.0],
            [11.0, 0.0, 0.0],
            [23.0, 0.0, 0.0],
            [31.0, 0.0, 0.0],
            [42.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    rows = [
        {"imu_stop_detected": False, "gate_ess_ratio": 1.0, "dd_pr_kept_pairs": 0},
        {"imu_stop_detected": False, "gate_ess_ratio": 1.0, "dd_pr_kept_pairs": 0},
        {"imu_stop_detected": False, "gate_ess_ratio": 1e-5, "dd_pr_kept_pairs": 0},
        {"imu_stop_detected": True, "gate_ess_ratio": 1.0, "dd_pr_kept_pairs": 0},
        {"imu_stop_detected": False, "gate_ess_ratio": 1.0, "dd_pr_kept_pairs": 1},
    ]

    guarded, n_applied = _apply_smoother_tail_guard(
        smoothed,
        forward,
        rows,
        ess_max_ratio=1e-4,
        min_shift_m=2.0,
        expand_epochs=1,
        expand_min_shift_m=0.5,
        expand_dd_pseudorange_max_pairs=0,
    )

    assert n_applied == 2
    np.testing.assert_allclose(guarded[0], smoothed[0])
    np.testing.assert_allclose(guarded[1], forward[1])
    np.testing.assert_allclose(guarded[2], forward[2])
    np.testing.assert_allclose(guarded[3], smoothed[3])
    np.testing.assert_allclose(guarded[4], smoothed[4])
    assert rows[1]["tail_guard_applied"] is True
    assert rows[1]["tail_guard_expanded"] is True
    assert rows[2]["tail_guard_applied"] is True
    assert rows[2]["tail_guard_expanded"] is False
    assert rows[3]["tail_guard_applied"] is False



def test_stop_segment_constant_position_replaces_contiguous_stop_window():
    smoothed = np.array(
        [
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [11.0, 1.0, 0.0],
            [9.0, -1.0, 0.0],
            [30.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    forward = smoothed.copy()
    diagnostics = [{} for _ in range(len(smoothed))]

    updated, info = _apply_stop_segment_constant_position(
        smoothed,
        forward,
        [False, True, True, True, False],
        diagnostics,
        min_epochs=3,
        source="smoothed",
    )

    np.testing.assert_allclose(updated[0], smoothed[0])
    np.testing.assert_allclose(updated[4], smoothed[4])
    np.testing.assert_allclose(updated[1:4], np.array([[10.0, 0.0, 0.0]] * 3))
    assert info["segments"] == 1
    assert info["segments_applied"] == 1
    assert info["epochs_applied"] == 3
    assert diagnostics[1]["stop_segment_constant_applied"] is True
    assert diagnostics[0]["stop_segment_constant_applied"] is False


def test_stop_segment_density_source_uses_dense_cluster_center():
    smoothed = np.array(
        [
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [10.1, 0.0, 0.0],
            [10.2, 0.0, 0.0],
            [20.0, 0.0, 0.0],
            [20.1, 0.0, 0.0],
            [50.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    forward = smoothed.copy()
    diagnostics = [{} for _ in range(len(smoothed))]

    updated, info = _apply_stop_segment_constant_position(
        smoothed,
        forward,
        [False, True, True, True, True, True, False],
        diagnostics,
        min_epochs=5,
        source="smoothed_density",
        density_neighbors=3,
    )

    np.testing.assert_allclose(updated[1:6], np.array([[10.1, 0.0, 0.0]] * 5))
    assert info["segments_applied"] == 1
    assert info["density_neighbors"] == 3
    assert diagnostics[1]["stop_segment_density_neighbors"] == 3
    assert diagnostics[1]["stop_segment_density_radius_m"] > 0.0


def test_stop_segment_auto_source_uses_density_only_for_compact_cluster():
    smoothed = np.array(
        [
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [10.1, 0.0, 0.0],
            [10.2, 0.0, 0.0],
            [20.0, 0.0, 0.0],
            [20.1, 0.0, 0.0],
            [50.0, 0.0, 0.0],
            [90.0, 0.0, 0.0],
            [100.0, 0.0, 0.0],
            [100.1, 0.0, 0.0],
            [100.2, 0.0, 0.0],
            [140.0, 0.0, 0.0],
            [140.1, 0.0, 0.0],
            [200.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    forward = smoothed.copy()
    diagnostics = [{} for _ in range(len(smoothed))]

    updated, info = _apply_stop_segment_constant_position(
        smoothed,
        forward,
        [
            False,
            True,
            True,
            True,
            True,
            True,
            False,
            True,
            True,
            True,
            True,
            True,
            True,
            False,
        ],
        diagnostics,
        min_epochs=5,
        source="smoothed_auto",
        density_neighbors=3,
    )

    np.testing.assert_allclose(updated[1:6], np.array([[10.1, 0.0, 0.0]] * 5))
    np.testing.assert_allclose(updated[7:13], np.array([[100.15, 0.0, 0.0]] * 6))
    assert info["segments_applied"] == 2
    assert info["density_segments_selected"] == 1
    assert diagnostics[1]["stop_segment_center_mode"] == "density"
    assert diagnostics[7]["stop_segment_center_mode"] == "median"


def test_stop_segment_auto_tail_source_uses_principal_tail_for_extreme_segment():
    smoothed = np.array(
        [
            [-10.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            [40.0, 0.0, 0.0],
            [80.0, 0.0, 0.0],
            [120.0, 0.0, 0.0],
            [160.0, 0.0, 0.0],
            [200.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    diagnostics = [{} for _ in range(len(smoothed))]

    updated, info = _apply_stop_segment_constant_position(
        smoothed,
        smoothed.copy(),
        [False, True, True, True, True, True, True, True, False],
        diagnostics,
        min_epochs=7,
        source="smoothed_auto_tail",
        density_neighbors=3,
    )

    np.testing.assert_allclose(updated[1:8], np.array([[120.0, 0.0, 0.0]] * 7))
    assert info["principal_segments_selected"] == 1
    assert diagnostics[1]["stop_segment_center_mode"] == "principal"
    assert diagnostics[1]["stop_segment_principal_percentile"] == 20.0


def test_static_stop_segment_gnss_solves_common_position_with_epoch_clocks():
    true_pos = np.array([10.0, 20.0, 30.0], dtype=np.float64)
    initial_pos = true_pos + np.array([8.0, -3.0, 2.0], dtype=np.float64)
    directions = np.array(
        [
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [-1.0, 0.0, 1.0],
            [0.0, -1.0, 1.0],
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    directions /= np.linalg.norm(directions, axis=1).reshape(-1, 1)
    sat = true_pos.reshape(1, 3) + directions * 20_200_000.0
    epochs = []
    for i in range(6):
        clock_bias = 100_000.0 + i * 10.0
        pr = np.linalg.norm(sat - true_pos.reshape(1, 3), axis=1) + clock_bias
        epochs.append(
            UndiffPseudorangeEpoch(
                sat_ecef=sat,
                pseudoranges_m=pr,
                clock_bias_m=clock_bias,
                weights=np.ones(len(pr), dtype=np.float64),
            )
        )
    smoothed = np.vstack([initial_pos + [0.1 * i, 0.0, 0.0] for i in range(6)])
    diagnostics = [{} for _ in range(6)]

    updated, info = apply_static_stop_segment_gnss(
        smoothed,
        [True] * 6,
        [None] * 6,
        [None] * 6,
        epochs,
        diagnostics,
        config=StaticStopSegmentConfig(
            min_epochs=5,
            min_observations=20,
            prior_sigma_m=100.0,
            undiff_pr_sigma_m=1.0,
            dd_cp_sigma_cycles=0.0,
            max_update_m=20.0,
        ),
    )

    assert info["segments_applied"] == 1
    assert info["epochs_applied"] == 6
    np.testing.assert_allclose(updated, np.vstack([true_pos] * 6), atol=0.2)
    assert diagnostics[0]["stop_segment_static_applied"] is True
    assert diagnostics[0]["stop_segment_static_reason"] in {"converged", "ok"}
