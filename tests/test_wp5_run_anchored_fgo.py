"""Unit tests for experiments/wp5_run_anchored_fgo.py (synthetic, no dataset)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))
sys.path.insert(0, str(PROJECT_ROOT / "python"))

from wp5_run_anchored_fgo import (  # noqa: E402
    RtkPosRecord,
    anchor_sigma_m,
    build_hybrid_coverage_seed,
    build_hybrid_seed,
    classify_anchor_status,
    compute_extension_stats,
    load_rtk_pos_with_status,
    nearest_anchor_distance_epochs,
    nearest_fix_distance_epochs,
)


_RTK_POS_HEADER = (
    "% LibGNSS++ Position Solution\n"
    "% GPS_Week GPS_TOW X(m) Y(m) Z(m) Lat(deg) Lon(deg) Height(m) Status NumSat\n"
)


def _write_rtk_pos(path: Path, rows: list[tuple[float, float, float, float, int]]) -> None:
    lines = [_RTK_POS_HEADER]
    for tow, x, y, z, status in rows:
        lines.append(f"2324 {tow:14.4f} {x:16.4f} {y:16.4f} {z:16.4f}  0.0 0.0 0.0 {status}   10\n")
    path.write_text("".join(lines), encoding="ascii")


def test_load_rtk_pos_with_status_parses_status_column(tmp_path: Path) -> None:
    path = tmp_path / "rtk.pos"
    _write_rtk_pos(
        path,
        [
            (100.0, 1.0, 2.0, 3.0, 4),
            (100.2, 4.0, 5.0, 6.0, 3),
            (100.4, 7.0, 8.0, 9.0, 0),
        ],
    )
    out = load_rtk_pos_with_status(path)
    assert set(out.keys()) == {100.0, 100.2, 100.4}
    ecef, status = out[100.0]
    np.testing.assert_allclose(ecef, [1.0, 2.0, 3.0])
    assert status == 4
    assert out[100.4][1] == 0


def test_build_hybrid_seed_prefers_rtk_over_backbone_when_status_nonzero() -> None:
    tows = np.array([100.0, 100.2, 100.4, 100.6])
    backbone = np.array(
        [[10.0, 0.0, 0.0], [11.0, 0.0, 0.0], [12.0, 0.0, 0.0], [13.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    rtk_by_tow = {
        100.0: (np.array([1.0, 1.0, 1.0]), 4),  # FIX -> use RTK
        100.2: (np.array([2.0, 2.0, 2.0]), 3),  # FLOAT -> use RTK
        100.4: (np.array([99.0, 99.0, 99.0]), 0),  # Status 0 -> fall back to backbone
        # 100.6 missing entirely -> backbone
    }
    positions, is_rtk = build_hybrid_seed(tows, rtk_by_tow, backbone)
    np.testing.assert_allclose(positions[0], [1.0, 1.0, 1.0])
    np.testing.assert_allclose(positions[1], [2.0, 2.0, 2.0])
    np.testing.assert_allclose(positions[2], backbone[2])
    np.testing.assert_allclose(positions[3], backbone[3])
    assert list(is_rtk) == [True, True, False, False]


def test_build_hybrid_seed_rejects_shape_mismatch() -> None:
    tows = np.array([100.0, 100.2])
    backbone = np.zeros((3, 3))
    with pytest.raises(ValueError):
        build_hybrid_seed(tows, {}, backbone)


def test_classify_anchor_status_maps_fix_float_none() -> None:
    tows = np.array([1.0, 2.0, 3.0, 4.0])
    rtk_by_tow = {
        1.0: (np.zeros(3), 4),
        2.0: (np.zeros(3), 3),
        3.0: (np.zeros(3), 1),
        # 4.0 missing
    }
    out = classify_anchor_status(tows, rtk_by_tow, fix_statuses=(4,), float_statuses=(1, 3))
    assert list(out) == [2, 1, 1, 0]


def test_nearest_fix_distance_epochs_symmetric_and_zero_at_fix() -> None:
    anchor_class = np.array([0, 0, 2, 0, 0, 0, 2, 0])
    dist = nearest_fix_distance_epochs(anchor_class)
    np.testing.assert_allclose(dist, [2, 1, 0, 1, 2, 1, 0, 1])


def test_nearest_fix_distance_epochs_all_inf_when_no_fix() -> None:
    anchor_class = np.array([0, 1, 1, 0])
    dist = nearest_fix_distance_epochs(anchor_class)
    assert np.all(np.isinf(dist))


def test_compute_extension_stats_counts_passing_non_fix_epochs_by_distance() -> None:
    # Epochs 0..4; epoch 0 is FIX (exact), 1-4 are non-fix but all within 0.5 m
    # of the reference except epoch 4 (2 m off, should not count as a pass).
    tows = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    ecef = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            [0.3, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ]
    )
    anchor_class = np.array([2, 0, 0, 0, 0], dtype=np.int8)
    reference = {t: np.zeros(3) for t in tows}
    out = compute_extension_stats(tows, ecef, anchor_class, reference, pass_threshold_m=0.5)
    assert out["n_fix_epochs"] == 1
    assert out["n_ref_covered"] == 5
    assert out["n_pass_lt_50cm"] == 4  # epochs 0-3
    assert out["n_extension_pass_lt_50cm"] == 3  # epochs 1-3 (non-fix passes)
    assert out["extension_epochs_max"] == 3.0  # epoch 3 is 3 epochs from the anchor


def test_build_hybrid_coverage_seed_end_to_end(tmp_path: Path) -> None:
    rover_obs = tmp_path / "rover.obs"
    rover_obs.write_text(
        "     3.04           OBSERVATION DATA    M                   RINEX VERSION / TYPE\n"
        "G    4 C1C L1C D1C S1C                                    SYS / # / OBS TYPES\n"
        "                                                            END OF HEADER\n"
        "> 2024 07 23 04 04 30.0000000  0  1\n"
        "G01  20000000.000 8  105000000.000 8       -100.000 8     45.000\n"
        "> 2024 07 23 04 04 30.2000000  0  1\n"
        "G01  20000000.000 8  105000000.000 8       -100.000 8     45.000\n"
        "> 2024 07 23 04 04 30.4000000  0  1\n"
        "G01  20000000.000 8  105000000.000 8       -100.000 8     45.000\n",
        encoding="ascii",
    )
    backbone_csv = tmp_path / "backbone.csv"
    # Only epoch 0 has a backbone row; epochs 1-2 must be gap-filled.
    tows_written = None
    import wp5_run_anchored_fgo as mod

    tows_all = mod.parse_rover_tows_from_obs(rover_obs)
    tow0 = float(tows_all[0])
    backbone_csv.write_text(
        "tow,lat_deg,lon_deg,height_m,ecef_x,ecef_y,ecef_z,fix\n"
        f"{tow0:.1f},0,0,0,10.0,0.0,0.0,0\n",
        encoding="utf-8",
    )
    rtk_pos = tmp_path / "rtk.pos"
    tow1 = float(tows_all[1])
    _write_rtk_pos(rtk_pos, [(tow1, 5.0, 5.0, 5.0, 4)])

    out_pos = tmp_path / "hybrid_seed.pos"
    tows, stats = build_hybrid_coverage_seed(
        rover_obs_path=rover_obs,
        backbone_csv_path=backbone_csv,
        rtk_pos_path=rtk_pos,
        out_pos_path=out_pos,
    )
    assert stats["n_epochs"] == 3
    assert stats["n_rtk"] == 1
    assert stats["n_backbone"] == 2
    assert out_pos.exists()
    seed = mod.read_pos_ecef(out_pos)
    np.testing.assert_allclose(seed[round(tow1, 1)], [5.0, 5.0, 5.0])
    np.testing.assert_allclose(seed[round(tow0, 1)], [10.0, 0.0, 0.0])


def test_anchor_sigma_m_float_uses_pdop_floor_and_quality_weight() -> None:
    rec = RtkPosRecord(ecef=np.zeros(3), status=3, nsats=4, pdop=6.0, ratio=2.0)
    sigma = anchor_sigma_m(rec, 1, fix_sigma_m=0.15, float_sigma_m=3.0, quality_weight=True)
    assert sigma > 3.0
    assert sigma <= 5.0 * 1.5 * np.sqrt(8.0 / 4.0)


def test_nearest_anchor_distance_epochs_includes_float_when_enabled() -> None:
    anchor_class = np.array([1, 0, 0, 2, 0])
    dist_fix = nearest_anchor_distance_epochs(anchor_class, include_fix=True, include_float=False)
    dist_dense = nearest_anchor_distance_epochs(anchor_class, include_fix=True, include_float=True)
    assert dist_fix[1] == pytest.approx(2.0)
    assert dist_dense[1] == pytest.approx(1.0)
