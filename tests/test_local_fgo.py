import numpy as np

from gnss_gpu.local_fgo import (
    DDCarrierEpoch,
    DDPseudorangeEpoch,
    LocalFgoConfig,
    LocalFgoProblem,
    LocalFgoWindow,
    UndiffPseudorangeEpoch,
    detect_weak_dd_window,
    inject_into_pf,
    parse_window_spec,
    solve_local_fgo,
)


WAVELENGTH_M = 299792458.0 / 1575.42e6


def _synthetic_satellites():
    return np.array(
        [
            [120.0, 20.0, 80.0],
            [-110.0, 35.0, 70.0],
            [15.0, 130.0, 90.0],
            [30.0, -125.0, 85.0],
            [-20.0, 10.0, 155.0],
        ],
        dtype=np.float64,
    )


def _dd_epoch_for_position(x, sats, base_pos):
    ref = sats[0]
    sat_k = sats[1:4]
    sat_ref = np.repeat(ref.reshape(1, 3), len(sat_k), axis=0)
    base_k = np.linalg.norm(sat_k - base_pos, axis=1)
    base_ref = np.repeat(np.linalg.norm(ref - base_pos), len(sat_k))
    dd_m = (
        np.linalg.norm(sat_k - x, axis=1)
        - np.linalg.norm(sat_ref - x, axis=1)
        - base_k
        + base_ref
    )
    return DDPseudorangeEpoch(
        dd_pseudorange_m=dd_m,
        sat_ecef_k=sat_k,
        sat_ecef_ref=sat_ref,
        base_range_k=base_k,
        base_range_ref=base_ref,
        weights=np.ones(len(sat_k), dtype=np.float64),
    ), DDCarrierEpoch(
        dd_carrier_cycles=dd_m / WAVELENGTH_M + 17.0,
        sat_ecef_k=sat_k,
        sat_ecef_ref=sat_ref,
        base_range_k=base_k,
        base_range_ref=base_ref,
        wavelengths_m=np.repeat(WAVELENGTH_M, len(sat_k)),
        weights=np.ones(len(sat_k), dtype=np.float64),
    )


def test_local_fgo_reduces_synthetic_window_error():
    sats = _synthetic_satellites()
    base_pos = np.array([1_200.0, -80.0, 20.0], dtype=np.float64)
    true_pos = np.array([[float(i), 0.25 * float(i), 0.05 * float(i)] for i in range(8)])
    initial = true_pos.copy()
    initial[2:6, 1] += 2.0
    initial[2:6, 2] -= 1.0

    undiff = []
    dd_pr = []
    dd_cp = []
    for x in true_pos:
        clock_bias = 12.5
        undiff.append(
            UndiffPseudorangeEpoch(
                sat_ecef=sats,
                pseudoranges_m=np.linalg.norm(sats - x, axis=1) + clock_bias,
                clock_bias_m=clock_bias,
                weights=np.ones(len(sats), dtype=np.float64),
            )
        )
        pr_epoch, cp_epoch = _dd_epoch_for_position(x, sats, base_pos)
        dd_pr.append(pr_epoch)
        dd_cp.append(cp_epoch)

    result = solve_local_fgo(
        LocalFgoProblem(
            initial_positions_ecef=initial,
            prior_positions_ecef=true_pos,
            window=LocalFgoWindow(1, 6),
            motion_deltas_ecef=np.diff(true_pos, axis=0),
            undiff_pseudorange=undiff,
            dd_pseudorange=dd_pr,
            dd_carrier=dd_cp,
        ),
        LocalFgoConfig(
            prior_sigma_m=0.05,
            motion_sigma_m=0.2,
            dd_sigma_cycles=20.0,
            dd_pr_sigma_m=0.5,
            undiff_pr_sigma_m=0.5,
            max_iterations=30,
        ),
    )

    before = np.linalg.norm(initial[1:7] - true_pos[1:7], axis=1)
    after = np.linalg.norm(result.positions_ecef - true_pos[1:7], axis=1)
    assert float(np.median(after)) < 0.25 * float(np.median(before))
    assert result.factor_counts["dd_carrier"] > 0
    assert result.factor_counts["dd_pseudorange"] > 0
    assert result.final_error < result.initial_error


def test_window_helpers_and_injection():
    win = parse_window_spec("2:4", 10)
    assert win == LocalFgoWindow(2, 4)

    traj = np.zeros((6, 3), dtype=np.float64)
    replacement = np.ones((3, 3), dtype=np.float64)
    updated = inject_into_pf(traj, replacement, win)
    np.testing.assert_allclose(updated[2:5], replacement)
    np.testing.assert_allclose(traj, 0.0)


def test_detect_weak_dd_window_uses_longest_run():
    rows = [{"dd_cp_kept_pairs": 7, "dd_cp_input_pairs": 7, "used_dd_carrier": True} for _ in range(8)]
    for i in range(2, 5):
        rows[i] = {"dd_cp_kept_pairs": 3, "dd_cp_input_pairs": 5, "used_dd_carrier": True}
    for i in range(6, 8):
        rows[i] = {"dd_cp_kept_pairs": 8, "dd_cp_input_pairs": 8, "used_dd_carrier": False}

    assert detect_weak_dd_window(rows, min_epochs=3, dd_max_pairs=4) == LocalFgoWindow(2, 4)
