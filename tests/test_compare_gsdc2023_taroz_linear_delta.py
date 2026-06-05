from __future__ import annotations

import numpy as np

from experiments.compare_gsdc2023_taroz_linear_delta import (
    NATIVE_DELTA_COLUMNS,
    delta_comparison_frame,
    summarize_delta_comparison,
    taroz_gtsam_delta_to_native_delta,
)


def test_taroz_gtsam_delta_to_native_delta_reorders_pose_and_rotates_enu_vectors() -> None:
    taroz = np.zeros((1, 26), dtype=np.float64)
    taroz[0, 0:3] = [1.0, 2.0, 3.0]
    taroz[0, 3:6] = [4.0, 5.0, 6.0]
    taroz[0, 6:14] = np.arange(10.0, 18.0)
    taroz[0, 14:17] = [0.1, 0.2, 0.3]
    taroz[0, 17:20] = [7.0, 8.0, 9.0]
    taroz[0, 20:26] = np.arange(20.0, 26.0)

    native = taroz_gtsam_delta_to_native_delta(
        taroz,
        origin_ecef=np.array([6378137.0, 0.0, 0.0], dtype=np.float64),
    )

    np.testing.assert_allclose(native[0, 0:3], [3.0, 1.0, 2.0], atol=1e-9)
    np.testing.assert_allclose(native[0, 3:6], [6.0, 4.0, 5.0], atol=1e-9)
    np.testing.assert_allclose(native[0, 6:14], np.arange(10.0, 18.0))
    np.testing.assert_allclose(native[0, 14:17], [7.0, 8.0, 9.0])
    np.testing.assert_allclose(native[0, 17:20], [0.1, 0.2, 0.3])
    np.testing.assert_allclose(native[0, 20:26], np.arange(20.0, 26.0))


def test_delta_comparison_summary_groups_component_and_norm_stats() -> None:
    native = np.zeros((2, len(NATIVE_DELTA_COLUMNS)), dtype=np.float64)
    taroz = np.zeros_like(native)
    native[:, 6] = [1.0, 3.0]
    taroz[:, 6] = [0.5, 1.0]
    native[:, 0:3] = [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]

    comparison = delta_comparison_frame(native, taroz)
    summary = summarize_delta_comparison(comparison)

    assert summary["matched_rows"] == 2
    assert summary["groups"]["clock_bias_m"]["component_max_abs"] == 2.0
    assert summary["groups"]["position_m"]["max_norm"] == 2.0
    assert summary["column_rank"][0]["column"] in {"position_y", "clock_bias_m_0"}
