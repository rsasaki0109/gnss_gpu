"""Tests for double-differenced carrier phase formation."""

from __future__ import annotations

import numpy as np

from gnss_gpu.dd_carrier import (
    BEIDOU_B1I_WAVELENGTH,
    DDCarrierComputer,
    GALILEO_E1_WAVELENGTH,
    GPS_L1_WAVELENGTH,
)


class _Meas:
    __slots__ = (
        "system_id",
        "prn",
        "satellite_ecef",
        "carrier_phase",
        "elevation",
        "snr",
    )

    def __init__(
        self,
        system_id: int,
        prn: int,
        satellite_ecef: np.ndarray,
        carrier_phase: float,
        *,
        elevation: float = 0.6,
        snr: float = 45.0,
    ):
        self.system_id = int(system_id)
        self.prn = int(prn)
        self.satellite_ecef = np.asarray(satellite_ecef, dtype=np.float64)
        self.carrier_phase = float(carrier_phase)
        self.elevation = float(elevation)
        self.snr = float(snr)


def _hdr(content: str, label: str) -> str:
    return f"{content:<60}{label}\n"


def _obs_line(sat_id: str, value: float) -> str:
    return f"{sat_id:<3}{value:14.3f}  \n"


def _obs_line_multi(sat_id: str, obs_codes: list[str], obs_values: dict[str, float]) -> str:
    line = f"{sat_id:<3}"
    for code in obs_codes:
        value = float(obs_values.get(code, 0.0))
        line += f"{value:14.3f}  "
    return line.rstrip() + "\n"


def _write_base_rinex(tmp_path, base_ecef: np.ndarray, sat_ids: list[str], base_cp: np.ndarray):
    path = tmp_path / "base.obs"
    systems = sorted({sat_id.strip()[0] for sat_id in sat_ids})
    code_map = {"G": "L1C", "E": "L1X", "J": "L1C", "C": "L1I", "R": "L1C"}
    lines = [
        _hdr("     3.04           O                   G", "RINEX VERSION / TYPE"),
        _hdr("BASE", "MARKER NAME"),
        _hdr(f"{base_ecef[0]:14.4f}{base_ecef[1]:14.4f}{base_ecef[2]:14.4f}", "APPROX POSITION XYZ"),
    ]
    for sys_char in systems:
        lines.append(_hdr(f"{sys_char}    1 {code_map[sys_char]}", "SYS / # / OBS TYPES"))
    lines.append(_hdr("", "END OF HEADER"))
    lines.append(f"> 2024 01 01 00 00 00.0000000  0 {len(sat_ids):2d}\n")
    for sat_id, cp in zip(sat_ids, base_cp, strict=True):
        lines.append(_obs_line(sat_id, cp))
    path.write_text("".join(lines))
    return path


def _write_rinex_epochs(
    path,
    approx_ecef: np.ndarray,
    sat_ids: list[str],
    epochs: list[tuple[str, np.ndarray]],
):
    systems = sorted({sat_id.strip()[0] for sat_id in sat_ids})
    code_map = {"G": "L1C", "E": "L1X", "J": "L1C", "C": "L1I", "R": "L1C"}
    lines = [
        _hdr("     3.04           O                   G", "RINEX VERSION / TYPE"),
        _hdr("BASE", "MARKER NAME"),
        _hdr(
            f"{approx_ecef[0]:14.4f}{approx_ecef[1]:14.4f}{approx_ecef[2]:14.4f}",
            "APPROX POSITION XYZ",
        ),
    ]
    for sys_char in systems:
        lines.append(_hdr(f"{sys_char}    1 {code_map[sys_char]}", "SYS / # / OBS TYPES"))
    lines.append(_hdr("", "END OF HEADER"))
    for epoch_header, cps in epochs:
        lines.append(f"> {epoch_header}  0 {len(sat_ids):2d}\n")
        for sat_id, cp in zip(sat_ids, cps, strict=True):
            lines.append(_obs_line(sat_id, cp))
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


def test_dd_carrier_forms_per_system_references_and_wavelengths(tmp_path):
    base_ecef = np.array([-3957199.0, 3310205.0, 3737911.0], dtype=np.float64)
    rover_ecef = base_ecef + np.array([80.0, -15.0, 12.0], dtype=np.float64)

    sat_ids = ["G01", "G02", "G03", "E 1", "E11", "C 3", "C24"]
    sat_ecef = np.array(
        [
            [-14985000.0, -3988000.0, 21474000.0],
            [-9575000.0, 15498000.0, 19457000.0],
            [7624000.0, -16218000.0, 19843000.0],
            [16305000.0, 12037000.0, 17183000.0],
            [-20889000.0, 13759000.0, 8291000.0],
            [5463000.0, 24413000.0, 8934000.0],
            [22169000.0, 3975000.0, 13781000.0],
        ],
        dtype=np.float64,
    )
    wavelengths = np.array(
        [
            GPS_L1_WAVELENGTH,
            GPS_L1_WAVELENGTH,
            GPS_L1_WAVELENGTH,
            GALILEO_E1_WAVELENGTH,
            GALILEO_E1_WAVELENGTH,
            BEIDOU_B1I_WAVELENGTH,
            BEIDOU_B1I_WAVELENGTH,
        ],
        dtype=np.float64,
    )

    base_ranges = np.linalg.norm(sat_ecef - base_ecef, axis=1)
    rover_ranges = np.linalg.norm(sat_ecef - rover_ecef, axis=1)
    sys_offsets_base = {"G": 100.0, "E": 200.0, "C": -50.0}
    sys_offsets_rover = {"G": 150.0, "E": 260.0, "C": -10.0}
    base_cp = np.array(
        [
            base_ranges[i] / wavelengths[i] + sys_offsets_base[sat_ids[i].strip()[0]]
            for i in range(len(sat_ids))
        ],
        dtype=np.float64,
    )
    rover_cp = np.array(
        [
            rover_ranges[i] / wavelengths[i] + sys_offsets_rover[sat_ids[i].strip()[0]]
            for i in range(len(sat_ids))
        ],
        dtype=np.float64,
    )

    base_obs = _write_base_rinex(tmp_path, base_ecef, sat_ids, base_cp)
    computer = DDCarrierComputer(base_obs)

    elevations = [1.2, 0.9, 0.6, 1.1, 0.7, 1.0, 0.8]
    system_ids = [0, 0, 0, 2, 2, 3, 3]
    prns = [1, 2, 3, 1, 11, 3, 24]
    meas = [
        _Meas(system_ids[i], prns[i], sat_ecef[i], rover_cp[i], elevation=elevations[i], snr=45.0)
        for i in range(len(sat_ids))
    ]
    result = computer.compute_dd(86400.0, meas)

    assert result is not None
    assert result.n_dd == 4
    assert result.ref_sat_ids == ("C03", "E01", "G01", "G01")
    np.testing.assert_allclose(
        result.wavelengths_m,
        np.array(
            [
                BEIDOU_B1I_WAVELENGTH,
                GALILEO_E1_WAVELENGTH,
                GPS_L1_WAVELENGTH,
                GPS_L1_WAVELENGTH,
            ],
            dtype=np.float64,
        ),
    )

    expected = np.array(
        [
            (rover_ranges[6] - rover_ranges[5]) / BEIDOU_B1I_WAVELENGTH
            - (base_ranges[6] - base_ranges[5]) / BEIDOU_B1I_WAVELENGTH,
            (rover_ranges[4] - rover_ranges[3]) / GALILEO_E1_WAVELENGTH
            - (base_ranges[4] - base_ranges[3]) / GALILEO_E1_WAVELENGTH,
            (rover_ranges[1] - rover_ranges[0]) / GPS_L1_WAVELENGTH
            - (base_ranges[1] - base_ranges[0]) / GPS_L1_WAVELENGTH,
            (rover_ranges[2] - rover_ranges[0]) / GPS_L1_WAVELENGTH
            - (base_ranges[2] - base_ranges[0]) / GPS_L1_WAVELENGTH,
        ],
        dtype=np.float64,
    )
    np.testing.assert_allclose(result.dd_carrier_cycles, expected, atol=1e-3)


def test_dd_carrier_ignores_glonass_without_fdma_channel_support(tmp_path):
    base_ecef = np.array([-3957199.0, 3310205.0, 3737911.0], dtype=np.float64)
    rover_ecef = base_ecef + np.array([80.0, -15.0, 12.0], dtype=np.float64)
    sat_ids = ["R 6", "R16", "G01", "G02"]
    sat_ecef = np.array(
        [
            [-14985000.0, -3988000.0, 21474000.0],
            [-9575000.0, 15498000.0, 19457000.0],
            [7624000.0, -16218000.0, 19843000.0],
            [16305000.0, 12037000.0, 17183000.0],
        ],
        dtype=np.float64,
    )
    base_ranges = np.linalg.norm(sat_ecef - base_ecef, axis=1)
    rover_ranges = np.linalg.norm(sat_ecef - rover_ecef, axis=1)
    base_cp = np.array(
        [
            1000.0,
            1005.0,
            base_ranges[2] / GPS_L1_WAVELENGTH + 100.0,
            base_ranges[3] / GPS_L1_WAVELENGTH + 100.0,
        ],
        dtype=np.float64,
    )
    rover_cp = np.array(
        [
            1100.0,
            1105.0,
            rover_ranges[2] / GPS_L1_WAVELENGTH + 150.0,
            rover_ranges[3] / GPS_L1_WAVELENGTH + 150.0,
        ],
        dtype=np.float64,
    )

    base_obs = _write_base_rinex(tmp_path, base_ecef, sat_ids, base_cp)
    computer = DDCarrierComputer(base_obs)
    meas = [
        _Meas(1, 6, sat_ecef[0], rover_cp[0], elevation=0.9),
        _Meas(1, 16, sat_ecef[1], rover_cp[1], elevation=0.8),
        _Meas(0, 1, sat_ecef[2], rover_cp[2], elevation=1.2),
        _Meas(0, 2, sat_ecef[3], rover_cp[3], elevation=0.7),
    ]

    assert computer.compute_dd(86400.0, meas) is None


def test_dd_carrier_prefers_raw_rover_rinex_when_available(tmp_path):
    base_ecef = np.array([-3957199.0, 3310205.0, 3737911.0], dtype=np.float64)
    rover_ecef = base_ecef + np.array([80.0, -15.0, 12.0], dtype=np.float64)
    sat_ids = ["G01", "G02", "G03", "E 1", "E11", "C24"]
    sat_ecef = np.array(
        [
            [-14985000.0, -3988000.0, 21474000.0],
            [-9575000.0, 15498000.0, 19457000.0],
            [7624000.0, -16218000.0, 19843000.0],
            [16305000.0, 12037000.0, 17183000.0],
            [-20889000.0, 13759000.0, 8291000.0],
            [22169000.0, 3975000.0, 13781000.0],
        ],
        dtype=np.float64,
    )
    wavelengths = np.array(
        [
            GPS_L1_WAVELENGTH,
            GPS_L1_WAVELENGTH,
            GPS_L1_WAVELENGTH,
            GALILEO_E1_WAVELENGTH,
            GALILEO_E1_WAVELENGTH,
            BEIDOU_B1I_WAVELENGTH,
        ],
        dtype=np.float64,
    )
    base_ranges = np.linalg.norm(sat_ecef - base_ecef, axis=1)
    rover_ranges = np.linalg.norm(sat_ecef - rover_ecef, axis=1)
    base_cp = np.array(
        [base_ranges[i] / wavelengths[i] + 100.0 for i in range(len(sat_ids))],
        dtype=np.float64,
    )
    rover_cp = np.array(
        [rover_ranges[i] / wavelengths[i] + 150.0 for i in range(len(sat_ids))],
        dtype=np.float64,
    )
    base_obs = _write_rinex_epochs(
        tmp_path / "base.obs",
        base_ecef,
        sat_ids,
        [("2024 01 01 00 00 00.0000000", base_cp)],
    )
    rover_obs = _write_rinex_epochs(
        tmp_path / "rover.obs",
        base_ecef,
        sat_ids,
        [("2024 01 01 00 00 00.0000000", rover_cp)],
    )

    computer = DDCarrierComputer(base_obs, rover_obs_path=rover_obs)
    elevations = [1.2, 0.9, 0.6, 1.1, 0.7, 0.8]
    system_ids = [0, 0, 0, 2, 2, 3]
    prns = [1, 2, 3, 1, 11, 24]
    meas = [
        _Meas(
            system_ids[i],
            prns[i],
            sat_ecef[i],
            rover_cp[i] + 1.0e6,  # intentionally wrong if raw rover RINEX is ignored
            elevation=elevations[i],
            snr=45.0,
        )
        for i in range(len(sat_ids))
    ]

    result = computer.compute_dd(86400.0, meas)

    assert result is not None
    expected = np.array(
        [
            (rover_ranges[4] - rover_ranges[3]) / GALILEO_E1_WAVELENGTH
            - (base_ranges[4] - base_ranges[3]) / GALILEO_E1_WAVELENGTH,
            (rover_ranges[1] - rover_ranges[0]) / GPS_L1_WAVELENGTH
            - (base_ranges[1] - base_ranges[0]) / GPS_L1_WAVELENGTH,
            (rover_ranges[2] - rover_ranges[0]) / GPS_L1_WAVELENGTH
            - (base_ranges[2] - base_ranges[0]) / GPS_L1_WAVELENGTH,
        ],
        dtype=np.float64,
    )
    np.testing.assert_allclose(result.dd_carrier_cycles[:3], expected, atol=2e-3)


def test_dd_carrier_interpolates_base_rinex_to_rover_epoch(tmp_path):
    base_ecef = np.array([-3957199.0, 3310205.0, 3737911.0], dtype=np.float64)
    sat_ids = ["G01", "G02", "G03", "G04"]
    sat_ecef = np.array(
        [
            [-14985000.0, -3988000.0, 21474000.0],
            [-9575000.0, 15498000.0, 19457000.0],
            [7624000.0, -16218000.0, 19843000.0],
            [16305000.0, 12037000.0, 17183000.0],
        ],
        dtype=np.float64,
    )
    base_cp_0 = np.array([5000.0, 5010.0, 5020.0, 5030.0], dtype=np.float64)
    base_cp_1 = base_cp_0 + np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    rover_cp = np.array([6000.0, 6011.0, 6021.5, 6032.0], dtype=np.float64)
    interp_base = 0.9 * base_cp_0 + 0.1 * base_cp_1

    base_obs = _write_rinex_epochs(
        tmp_path / "base_interp.obs",
        base_ecef,
        sat_ids,
        [
            ("2024 01 01 00 00 00.0000000", base_cp_0),
            ("2024 01 01 00 00 01.0000000", base_cp_1),
        ],
    )
    rover_obs = _write_rinex_epochs(
        tmp_path / "rover_interp.obs",
        base_ecef,
        sat_ids,
        [("2024 01 01 00 00 00.1000000", rover_cp)],
    )

    computer = DDCarrierComputer(
        base_obs,
        rover_obs_path=rover_obs,
        interpolate_base_epochs=True,
    )
    meas = [
        _Meas(0, i + 1, sat_ecef[i], rover_cp[i] + 10_000.0, elevation=1.2 - 0.1 * i, snr=45.0)
        for i in range(len(sat_ids))
    ]

    result = computer.compute_dd(86400.1, meas)

    assert result is not None
    expected = (rover_cp[1:] - rover_cp[0]) - (interp_base[1:] - interp_base[0])
    np.testing.assert_allclose(result.dd_carrier_cycles, expected, atol=2e-3)


def test_dd_carrier_uses_one_common_code_per_system(tmp_path):
    base_ecef = np.array([-3957199.0, 3310205.0, 3737911.0], dtype=np.float64)
    rover_ecef = base_ecef + np.array([80.0, -15.0, 12.0], dtype=np.float64)
    sat_ids = ["G01", "G02", "G03", "G04"]
    sat_ecef = np.array(
        [
            [-14985000.0, -3988000.0, 21474000.0],
            [-9575000.0, 15498000.0, 19457000.0],
            [7624000.0, -16218000.0, 19843000.0],
            [16305000.0, 12037000.0, 17183000.0],
        ],
        dtype=np.float64,
    )
    base_ranges = np.linalg.norm(sat_ecef - base_ecef, axis=1)
    rover_ranges = np.linalg.norm(sat_ecef - rover_ecef, axis=1)
    base_cp = base_ranges / GPS_L1_WAVELENGTH + 100.0
    rover_cp = rover_ranges / GPS_L1_WAVELENGTH + 150.0

    obs_types = {"G": ["L1C", "L1W"]}
    base_obs = _write_rinex_epochs_multi(
        tmp_path / "base_multi.obs",
        base_ecef,
        obs_types,
        [
            (
                "2024 01 01 00 00 00.0000000",
                {
                    "G01": {"L1W": base_cp[0]},
                    "G02": {"L1W": base_cp[1]},
                    "G03": {"L1C": base_cp[2]},
                    "G04": {"L1W": base_cp[3]},
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
                    "G01": {"L1W": rover_cp[0]},
                    "G02": {"L1W": rover_cp[1]},
                    "G03": {"L1C": rover_cp[2]},
                    "G04": {"L1W": rover_cp[3]},
                },
            )
        ],
    )

    computer = DDCarrierComputer(base_obs, rover_obs_path=rover_obs)
    meas = [
        _Meas(0, i + 1, sat_ecef[i], rover_cp[i] + 1.0e6, elevation=1.2 - 0.1 * i, snr=45.0)
        for i in range(len(sat_ids))
    ]

    result = computer.compute_dd(86400.0, meas, min_common_sats=3)

    assert result is not None
    assert result.n_dd == 2
    assert result.ref_sat_ids == ("G01", "G01")
    expected = np.array(
        [
            (rover_ranges[1] - rover_ranges[0]) / GPS_L1_WAVELENGTH
            - (base_ranges[1] - base_ranges[0]) / GPS_L1_WAVELENGTH,
            (rover_ranges[3] - rover_ranges[0]) / GPS_L1_WAVELENGTH
            - (base_ranges[3] - base_ranges[0]) / GPS_L1_WAVELENGTH,
        ],
        dtype=np.float64,
    )
    np.testing.assert_allclose(result.dd_carrier_cycles, expected, atol=2e-3)
