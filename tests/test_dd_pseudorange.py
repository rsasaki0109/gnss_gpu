"""Tests for double-differenced pseudorange formation."""

from __future__ import annotations

import numpy as np

from gnss_gpu.dd_pseudorange import DDPseudorangeComputer


class _Meas:
    __slots__ = (
        "system_id",
        "prn",
        "satellite_ecef",
        "corrected_pseudorange",
        "elevation",
        "weight",
        "snr",
    )

    def __init__(
        self,
        system_id: int,
        prn: int,
        satellite_ecef: np.ndarray,
        corrected_pseudorange: float,
        *,
        elevation: float = 0.6,
        weight: float = 1.0,
        snr: float = 45.0,
    ):
        self.system_id = int(system_id)
        self.prn = int(prn)
        self.satellite_ecef = np.asarray(satellite_ecef, dtype=np.float64)
        self.corrected_pseudorange = float(corrected_pseudorange)
        self.elevation = float(elevation)
        self.weight = float(weight)
        self.snr = float(snr)


def _hdr(content: str, label: str) -> str:
    return f"{content:<60}{label}\n"


def _obs_line(sat_id: str, c1c: float) -> str:
    return f"{sat_id:<3}{c1c:14.3f}  \n"


def _obs_line_multi(sat_id: str, obs_codes: list[str], obs_values: dict[str, float]) -> str:
    line = f"{sat_id:<3}"
    for code in obs_codes:
        value = float(obs_values.get(code, 0.0))
        line += f"{value:14.3f}  "
    return line.rstrip() + "\n"


def _write_base_rinex(
    tmp_path,
    base_ecef: np.ndarray,
    sat_ids: list[str],
    base_pr: np.ndarray,
    filename: str = "base.obs",
):
    path = tmp_path / filename
    systems = sorted({sat_id[0] for sat_id in sat_ids})
    lines = [
        _hdr("     3.04           O                   G", "RINEX VERSION / TYPE"),
        _hdr("BASE", "MARKER NAME"),
        _hdr(f"{base_ecef[0]:14.4f}{base_ecef[1]:14.4f}{base_ecef[2]:14.4f}", "APPROX POSITION XYZ"),
    ]
    lines.extend(_hdr(f"{sys_char}    1 C1C", "SYS / # / OBS TYPES") for sys_char in systems)
    lines.append(_hdr("", "END OF HEADER"))
    lines.append(f"> 2024 01 01 00 00 00.0000000  0 {len(sat_ids):2d}\n")
    for sat_id, pr in zip(sat_ids, base_pr, strict=True):
        lines.append(_obs_line(sat_id, pr))
    path.write_text("".join(lines))
    return path


def _write_rinex_epochs(
    path,
    approx_ecef: np.ndarray,
    sat_ids: list[str],
    epochs: list[tuple[str, np.ndarray]],
):
    systems = sorted({sat_id[0] for sat_id in sat_ids})
    lines = [
        _hdr("     3.04           O                   G", "RINEX VERSION / TYPE"),
        _hdr("BASE", "MARKER NAME"),
        _hdr(
            f"{approx_ecef[0]:14.4f}{approx_ecef[1]:14.4f}{approx_ecef[2]:14.4f}",
            "APPROX POSITION XYZ",
        ),
    ]
    lines.extend(_hdr(f"{sys_char}    1 C1C", "SYS / # / OBS TYPES") for sys_char in systems)
    lines.append(_hdr("", "END OF HEADER"))
    for epoch_header, prs in epochs:
        lines.append(f"> {epoch_header}  0 {len(sat_ids):2d}\n")
        for sat_id, pr in zip(sat_ids, prs, strict=True):
            lines.append(_obs_line(sat_id, pr))
    path.write_text("".join(lines))
    return path


def _write_rinex_epochs_multi(
    path,
    approx_ecef: np.ndarray,
    obs_types_by_system: dict[str, list[str]],
    epochs: list[tuple[str, dict[str, dict[str, float]]]],
):
    lines = [
        _hdr("     3.04           O                   G", "RINEX VERSION / TYPE"),
        _hdr("BASE", "MARKER NAME"),
        _hdr(
            f"{approx_ecef[0]:14.4f}{approx_ecef[1]:14.4f}{approx_ecef[2]:14.4f}",
            "APPROX POSITION XYZ",
        ),
    ]
    for sys_char in sorted(obs_types_by_system):
        obs_codes = obs_types_by_system[sys_char]
        lines.append(_hdr(f"{sys_char}{len(obs_codes):5d} {' '.join(obs_codes)}", "SYS / # / OBS TYPES"))
    lines.append(_hdr("", "END OF HEADER"))
    for epoch_header, observations in epochs:
        sat_ids = sorted(observations)
        lines.append(f"> {epoch_header}  0 {len(sat_ids):2d}\n")
        for sat_id in sat_ids:
            lines.append(
                _obs_line_multi(
                    sat_id,
                    obs_types_by_system[sat_id[0]],
                    observations[sat_id],
                )
            )
    path.write_text("".join(lines))
    return path


def _make_geometry():
    base_ecef = np.array([-3957199.0, 3310205.0, 3737911.0])
    rover_ecef = base_ecef + np.array([120.0, -35.0, 18.0])
    sat_ecef = np.array(
        [
            [-14985000.0, -3988000.0, 21474000.0],
            [-9575000.0, 15498000.0, 19457000.0],
            [7624000.0, -16218000.0, 19843000.0],
            [16305000.0, 12037000.0, 17183000.0],
        ],
        dtype=np.float64,
    )
    sat_ids = [f"G{i:02d}" for i in range(1, len(sat_ecef) + 1)]
    base_ranges = np.linalg.norm(sat_ecef - base_ecef, axis=1)
    rover_ranges = np.linalg.norm(sat_ecef - rover_ecef, axis=1)
    return base_ecef, rover_ecef, sat_ecef, sat_ids, base_ranges, rover_ranges


def test_dd_pseudorange_matches_geometric_double_difference(tmp_path):
    base_ecef, rover_ecef, sat_ecef, sat_ids, base_ranges, rover_ranges = _make_geometry()
    del rover_ecef

    base_bias = 1200.0
    rover_bias = 5300.0
    base_pr = base_ranges + base_bias
    rover_pr = rover_ranges + rover_bias

    base_obs = _write_base_rinex(tmp_path, base_ecef, sat_ids, base_pr)
    computer = DDPseudorangeComputer(base_obs)

    elevations = [1.2, 0.8, 0.6, 0.4]
    meas = [
        _Meas(0, i + 1, sat_ecef[i], rover_pr[i], elevation=elevations[i], weight=1.0, snr=45.0)
        for i in range(len(sat_ecef))
    ]
    result = computer.compute_dd(86400.0, meas)

    assert result is not None
    assert result.ref_sat_ids == ("G01", "G01", "G01")
    assert result.n_dd == 3

    ref = 0
    expected_dd = (rover_ranges[1:] - rover_ranges[ref]) - (base_ranges[1:] - base_ranges[ref])
    np.testing.assert_allclose(result.dd_pseudorange_m, expected_dd, atol=2e-3)
    np.testing.assert_allclose(result.sat_ecef_ref, np.repeat(sat_ecef[[0]], 3, axis=0))
    np.testing.assert_allclose(result.base_range_ref, np.repeat(base_ranges[0], 3))


def test_dd_pseudorange_prefers_unique_highest_snr_row(tmp_path):
    base_ecef, _rover_ecef, sat_ecef, sat_ids, base_ranges, rover_ranges = _make_geometry()
    base_pr = base_ranges + 2000.0
    rover_pr = rover_ranges + 6000.0

    base_obs = _write_base_rinex(tmp_path, base_ecef, sat_ids, base_pr)
    computer = DDPseudorangeComputer(base_obs)

    meas = [
        _Meas(0, 1, sat_ecef[0], rover_pr[0], elevation=1.2, snr=50.0, weight=4.0),
        _Meas(0, 1, sat_ecef[0], rover_pr[0] + 100.0, elevation=1.2, snr=30.0, weight=0.1),
        _Meas(0, 2, sat_ecef[1], rover_pr[1], elevation=0.8, snr=45.0, weight=1.0),
        _Meas(0, 3, sat_ecef[2], rover_pr[2], elevation=0.6, snr=44.0, weight=1.0),
        _Meas(0, 4, sat_ecef[3], rover_pr[3], elevation=0.4, snr=43.0, weight=1.0),
    ]
    result = computer.compute_dd(86400.0, meas)

    assert result is not None
    ref = 0
    expected_dd = (rover_ranges[1:] - rover_ranges[ref]) - (base_ranges[1:] - base_ranges[ref])
    np.testing.assert_allclose(result.dd_pseudorange_m, expected_dd, atol=2e-3)


def test_dd_pseudorange_returns_none_without_enough_common_sats(tmp_path):
    base_ecef, _rover_ecef, sat_ecef, sat_ids, base_ranges, rover_ranges = _make_geometry()
    base_obs = _write_base_rinex(tmp_path, base_ecef, sat_ids[:3], base_ranges[:3] + 1000.0)
    computer = DDPseudorangeComputer(base_obs)

    meas = [
        _Meas(0, i + 1, sat_ecef[i], rover_ranges[i] + 5000.0, elevation=0.5, snr=45.0)
        for i in range(3)
    ]

    assert computer.compute_dd(86400.0, meas) is None


def test_dd_pseudorange_forms_per_system_references(tmp_path):
    base_ecef = np.array([-3957199.0, 3310205.0, 3737911.0])
    rover_ecef = base_ecef + np.array([80.0, -15.0, 12.0])
    sat_ids = ["G01", "G02", "G03", "E11", "E12"]
    sat_ecef = np.array(
        [
            [-14985000.0, -3988000.0, 21474000.0],
            [-9575000.0, 15498000.0, 19457000.0],
            [7624000.0, -16218000.0, 19843000.0],
            [16305000.0, 12037000.0, 17183000.0],
            [-20889000.0, 13759000.0, 8291000.0],
        ],
        dtype=np.float64,
    )
    base_ranges = np.linalg.norm(sat_ecef - base_ecef, axis=1)
    rover_ranges = np.linalg.norm(sat_ecef - rover_ecef, axis=1)
    base_obs = _write_base_rinex(tmp_path, base_ecef, sat_ids, base_ranges + 1200.0)
    computer = DDPseudorangeComputer(base_obs)

    elevations = [1.3, 0.9, 0.6, 1.1, 0.7]
    meas = [
        _Meas(
            0 if sat_ids[i].startswith("G") else 2,
            int(sat_ids[i][1:]),
            sat_ecef[i],
            rover_ranges[i] + 5300.0,
            elevation=elevations[i],
            snr=45.0,
        )
        for i in range(len(sat_ids))
    ]
    result = computer.compute_dd(86400.0, meas)

    assert result is not None
    assert result.n_dd == 3
    assert result.ref_sat_ids == ("E11", "G01", "G01")

    expected = np.array(
        [
            (rover_ranges[4] - rover_ranges[3]) - (base_ranges[4] - base_ranges[3]),
            (rover_ranges[1] - rover_ranges[0]) - (base_ranges[1] - base_ranges[0]),
            (rover_ranges[2] - rover_ranges[0]) - (base_ranges[2] - base_ranges[0]),
        ],
        dtype=np.float64,
    )
    np.testing.assert_allclose(result.dd_pseudorange_m, expected, atol=1e-3)
    np.testing.assert_allclose(result.sat_ecef_ref[0], sat_ecef[3], atol=1e-9)
    np.testing.assert_allclose(result.sat_ecef_ref[1:], np.repeat(sat_ecef[[0]], 2, axis=0), atol=1e-9)


def test_dd_pseudorange_prefers_raw_rover_rinex_when_available(tmp_path):
    base_ecef, _rover_ecef, sat_ecef, sat_ids, base_ranges, rover_ranges = _make_geometry()
    base_obs = _write_base_rinex(tmp_path, base_ecef, sat_ids, base_ranges + 1200.0, "base.obs")
    rover_obs = _write_base_rinex(tmp_path, base_ecef, sat_ids, rover_ranges + 5300.0, "rover.obs")

    computer = DDPseudorangeComputer(base_obs, rover_obs_path=rover_obs, interpolate_base_epochs=True)
    meas = [
        _Meas(
            0,
            i + 1,
            sat_ecef[i],
            rover_ranges[i] + 5300.0 + 10_000.0,  # intentionally wrong if raw rover RINEX is ignored
            elevation=1.2 - 0.2 * i,
            snr=45.0,
        )
        for i in range(len(sat_ecef))
    ]
    result = computer.compute_dd(86400.0, meas)

    assert result is not None
    expected_dd = (rover_ranges[1:] - rover_ranges[0]) - (base_ranges[1:] - base_ranges[0])
    np.testing.assert_allclose(result.dd_pseudorange_m, expected_dd, atol=2e-3)


def test_dd_pseudorange_interpolates_base_rinex_to_rover_epoch(tmp_path):
    base_ecef, _rover_ecef, sat_ecef, sat_ids, _base_ranges, _rover_ranges = _make_geometry()
    base_pr_0 = np.array([20_000_000.0, 20_000_010.0, 20_000_020.0, 20_000_030.0], dtype=np.float64)
    base_pr_1 = base_pr_0 + np.array([10.0, 12.0, 14.0, 16.0], dtype=np.float64)
    rover_pr = np.array([21_000_000.0, 21_000_011.0, 21_000_021.5, 21_000_032.0], dtype=np.float64)
    interp_base = 0.9 * base_pr_0 + 0.1 * base_pr_1

    base_obs = _write_rinex_epochs(
        tmp_path / "base_interp.obs",
        base_ecef,
        sat_ids,
        [
            ("2024 01 01 00 00 00.0000000", base_pr_0),
            ("2024 01 01 00 00 01.0000000", base_pr_1),
        ],
    )
    rover_obs = _write_rinex_epochs(
        tmp_path / "rover_interp.obs",
        base_ecef,
        sat_ids,
        [("2024 01 01 00 00 00.1000000", rover_pr)],
    )
    computer = DDPseudorangeComputer(
        base_obs,
        rover_obs_path=rover_obs,
        interpolate_base_epochs=True,
    )
    meas = [
        _Meas(0, i + 1, sat_ecef[i], rover_pr[i] + 50_000.0, elevation=1.2 - 0.2 * i, snr=45.0)
        for i in range(len(sat_ids))
    ]

    result = computer.compute_dd(86400.1, meas)

    assert result is not None
    expected_dd = (rover_pr[1:] - rover_pr[0]) - (interp_base[1:] - interp_base[0])
    np.testing.assert_allclose(result.dd_pseudorange_m, expected_dd, atol=2e-3)


def test_dd_pseudorange_uses_one_common_code_per_system(tmp_path):
    base_ecef, rover_ecef, sat_ecef, sat_ids, base_ranges, rover_ranges = _make_geometry()
    del rover_ecef

    obs_types = {"G": ["C1C", "C1W"]}
    base_obs = _write_rinex_epochs_multi(
        tmp_path / "base_multi.obs",
        base_ecef,
        obs_types,
        [
            (
                "2024 01 01 00 00 00.0000000",
                {
                    "G01": {"C1W": base_ranges[0] + 1200.0},
                    "G02": {"C1W": base_ranges[1] + 1200.0},
                    "G03": {"C1C": base_ranges[2] + 1200.0},
                    "G04": {"C1W": base_ranges[3] + 1200.0},
                },
            )
        ],
    )
    rover_obs = _write_rinex_epochs_multi(
        tmp_path / "rover_multi.obs",
        base_ecef,
        obs_types,
        [
            (
                "2024 01 01 00 00 00.0000000",
                {
                    "G01": {"C1W": rover_ranges[0] + 5300.0},
                    "G02": {"C1W": rover_ranges[1] + 5300.0},
                    "G03": {"C1C": rover_ranges[2] + 5300.0},
                    "G04": {"C1W": rover_ranges[3] + 5300.0},
                },
            )
        ],
    )
    computer = DDPseudorangeComputer(base_obs, rover_obs_path=rover_obs)
    meas = [
        _Meas(0, i + 1, sat_ecef[i], rover_ranges[i] + 9999.0, elevation=1.2 - 0.2 * i, snr=45.0)
        for i in range(len(sat_ids))
    ]

    result = computer.compute_dd(86400.0, meas, min_common_sats=3)

    assert result is not None
    assert result.n_dd == 2
    assert result.ref_sat_ids == ("G01", "G01")
    expected_dd = np.array(
        [
            (rover_ranges[1] - rover_ranges[0]) - (base_ranges[1] - base_ranges[0]),
            (rover_ranges[3] - rover_ranges[0]) - (base_ranges[3] - base_ranges[0]),
        ],
        dtype=np.float64,
    )
    np.testing.assert_allclose(result.dd_pseudorange_m, expected_dd, atol=2e-3)
