import numpy as np

from gnss_gpu.local_fgo import LocalFgoConfig, UndiffPseudorangeEpoch
from gnss_gpu.local_fgo_bridge import (
    _apply_local_fgo_postprocess,
    _select_local_fgo_motion_deltas,
)


def test_local_fgo_postprocess_replaces_requested_window_only():
    sats = np.array(
        [
            [120.0, 20.0, 80.0],
            [-110.0, 35.0, 70.0],
            [15.0, 130.0, 90.0],
            [30.0, -125.0, 85.0],
            [-20.0, 10.0, 155.0],
        ],
        dtype=np.float64,
    )
    true_pos = np.array([[float(i), 0.2 * float(i), 0.05 * float(i)] for i in range(6)])
    smoothed = true_pos.copy()
    smoothed[1:5, 1] += 1.5
    smoothed[1:5, 2] -= 0.5
    clock_bias = 4.0
    stored_undiff = [
        UndiffPseudorangeEpoch(
            sat_ecef=sats,
            pseudoranges_m=np.linalg.norm(sats - pos, axis=1) + clock_bias,
            clock_bias_m=clock_bias,
            weights=np.ones(len(sats), dtype=np.float64),
        )
        for pos in true_pos
    ]

    updated, info = _apply_local_fgo_postprocess(
        smoothed,
        aligned_indices=list(range(len(smoothed))),
        stored_motion_deltas=list(np.diff(true_pos, axis=0)),
        stored_tdcp_motion_deltas=None,
        stored_dd_carrier=[None] * len(smoothed),
        stored_dd_pseudorange=[None] * len(smoothed),
        stored_undiff_pr=stored_undiff,
        epoch_diagnostics=None,
        window_spec="1:4",
        min_epochs=2,
        dd_max_pairs=4,
        config=LocalFgoConfig(
            prior_sigma_m=0.05,
            motion_sigma_m=0.2,
            undiff_pr_sigma_m=0.2,
            max_iterations=20,
        ),
        two_step=True,
    )

    assert info["applied"] is True
    assert info["window"] == "1:4"
    assert info["two_step"] is True
    assert info["stage1"]["applied"] is True
    np.testing.assert_allclose(updated[0], smoothed[0])
    np.testing.assert_allclose(updated[5], smoothed[5])
    before = np.linalg.norm(smoothed[1:5] - true_pos[1:5], axis=1)
    after = np.linalg.norm(updated[1:5] - true_pos[1:5], axis=1)
    assert float(np.median(after)) < 0.25 * float(np.median(before))



def test_select_local_fgo_motion_deltas_prefers_tdcp_when_finite():
    predict = [
        np.array([1.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
    ]
    tdcp = [
        np.array([10.0, 0.0, 0.0]),
        np.array([np.nan, np.nan, np.nan]),
        np.array([30.0, 0.0, 0.0]),
    ]

    selected, info = _select_local_fgo_motion_deltas(
        [0, 1, 2, 3],
        predict,
        tdcp,
        motion_source="prefer_tdcp",
    )

    np.testing.assert_allclose(
        selected,
        np.array(
            [
                [10.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [30.0, 0.0, 0.0],
            ]
        ),
    )
    assert info["motion_source"] == "prefer_tdcp"
    assert info["motion_predict_edges"] == 3
    assert info["motion_tdcp_edges"] == 2
    assert info["motion_tdcp_selected_edges"] == 2
