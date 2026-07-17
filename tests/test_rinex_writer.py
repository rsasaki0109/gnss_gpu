"""Tests for the RINEX 3.04 observation writer.

Covers a round-trip write -> read through the repo's own
``read_rinex_obs`` parser (RINEX 3.x branch), blank/NaN field handling,
header field emission (including a >13 obs-types continuation line),
and, when a real fixture is available, a read -> write -> read
idempotence check.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from gnss_gpu.io.rinex import read_rinex_obs
from gnss_gpu.io.rinex_writer import (
    EpochRecord,
    RinexObsHeader,
    write_rinex_obs,
    write_rinex_obs_from_arrays,
)

FIXTURE_CANDIDATES = [
    Path("experiments/data/urbannav/Odaiba/rover_ublox.obs"),
    Path("experiments/data/urbannav/Odaiba/base_trimble.obs"),
    Path("experiments/data/urbannav/Odaiba/rover_trimble.obs"),
    Path("experiments/data/urbannav/Shinjuku/rover_ublox.obs"),
]


def _find_fixture() -> Path | None:
    for candidate in FIXTURE_CANDIDATES:
        if candidate.exists():
            return candidate
    return None


def _make_header(**overrides) -> RinexObsHeader:
    kwargs = dict(
        marker_name="TEST",
        receiver_type="test-rx",
        antenna_type="test-ant",
        approx_position_ecef=np.array([-3957196.1328, 3310204.8922, 3737910.8017]),
        obs_types={
            "G": ["C1C", "L1C", "D1C", "S1C"],
            "E": ["C1C", "L1C", "D1C", "S1C"],
        },
        interval_s=1.0,
        time_first_obs=datetime(2024, 7, 20, 9, 52, 30),
    )
    kwargs.update(overrides)
    return RinexObsHeader(**kwargs)


def _make_epochs() -> list[EpochRecord]:
    t0 = datetime(2024, 7, 20, 9, 52, 30)
    epochs = []
    for k in range(3):
        t = t0.replace(second=30 + k)
        sat_ids = ["G05", "G12", "E09"]
        pr = np.array([20200002.0 + k, 20300003.0 + k, 21000000.0 + k])
        ph = np.array([106_000_000.1 + k, 107_000_000.2 + k, np.nan])
        dp = np.array([45.0 + k, 46.0 + k, 47.0 + k])
        cn0 = np.array([40.0, 41.0, np.nan])
        epochs.append(
            EpochRecord(
                time=t,
                sat_ids=sat_ids,
                obs={"C1C": pr, "L1C": ph, "D1C": dp, "S1C": cn0},
            )
        )
    return epochs


def test_round_trip_multi_constellation(tmp_path):
    header = _make_header()
    epochs = _make_epochs()
    out_path = tmp_path / "roundtrip.obs"

    write_rinex_obs(out_path, header, epochs)
    obs = read_rinex_obs(out_path)

    assert len(obs.epochs) == 3
    for expected, actual in zip(epochs, obs.epochs):
        assert actual.time == expected.time
        assert actual.satellites == expected.sat_ids
        for i, sat_id in enumerate(expected.sat_ids):
            for code, arr in expected.obs.items():
                expected_val = arr[i]
                actual_val = actual.observations[sat_id][code]
                if np.isnan(expected_val):
                    # The repo's reader treats blank fields as 0.0
                    # (its own "missing" sentinel -- see
                    # RinexObs.pseudoranges), so a blank round-trips
                    # to 0.0 rather than NaN.
                    assert actual_val == 0.0
                else:
                    assert abs(actual_val - expected_val) < 1e-3


def test_blank_field_round_trips_as_missing(tmp_path):
    header = _make_header(obs_types={"G": ["C1C", "L1C", "D1C", "S1C"]})
    t = datetime(2024, 7, 20, 9, 52, 30)
    epoch = EpochRecord(
        time=t,
        sat_ids=["G05"],
        obs={
            "C1C": np.array([20200002.0]),
            "L1C": np.array([np.nan]),
            "D1C": np.array([45.0]),
            "S1C": np.array([None]),
        },
    )
    out_path = tmp_path / "blank.obs"
    write_rinex_obs(out_path, header, [epoch])

    text = out_path.read_text()
    # locate the satellite data line and confirm the L1C/S1C fields are blank
    sat_line = [ln for ln in text.splitlines() if ln.startswith("G05")][0]
    l1c_field = sat_line[3 + 16 : 3 + 16 + 14]
    s1c_field = sat_line[3 + 16 * 3 : 3 + 16 * 3 + 14]
    assert l1c_field.strip() == ""
    assert s1c_field.strip() == ""

    obs = read_rinex_obs(out_path)
    sat_obs = obs.epochs[0].observations["G05"]
    assert sat_obs["C1C"] == pytest.approx(20200002.0, abs=1e-3)
    assert sat_obs["D1C"] == pytest.approx(45.0, abs=1e-3)
    # blank fields parse back as 0.0 (repo convention for "missing")
    assert sat_obs["L1C"] == 0.0
    assert sat_obs["S1C"] == 0.0


def test_header_fields_and_long_obs_type_continuation(tmp_path):
    codes = (
        [f"C{i}C" for i in range(1, 6)]
        + [f"L{i}C" for i in range(1, 6)]
        + [f"D{i}C" for i in range(1, 6)]
        + [f"S{i}C" for i in range(1, 6)]
    )
    assert len(codes) == 20  # > 13, forces a SYS / # / OBS TYPES continuation line

    header = _make_header(marker_name="LONGHDR", obs_types={"G": codes})
    t = datetime(2024, 1, 1, 0, 0, 0)
    epoch = EpochRecord(
        time=t,
        sat_ids=["G01"],
        obs={code: np.array([float(i)]) for i, code in enumerate(codes)},
    )
    out_path = tmp_path / "longheader.obs"
    write_rinex_obs(out_path, header, [epoch])

    text = out_path.read_text()
    obs_type_lines = [ln for ln in text.splitlines() if ln.endswith("SYS / # / OBS TYPES ")]
    assert len(obs_type_lines) >= 2, "expected a continuation line for >13 obs types"

    obs = read_rinex_obs(out_path)
    assert obs.header.marker_name == "LONGHDR"
    assert obs.header.obs_types["G"] == codes
    assert obs.epochs[0].observations["G01"]["S5C"] == 19.0


def test_write_rinex_obs_from_arrays_groups_by_epoch(tmp_path):
    t0 = datetime(2024, 7, 20, 9, 52, 30)
    t1 = datetime(2024, 7, 20, 9, 52, 31)
    epoch_times = [t0, t0, t1, t1]
    sat_ids = ["G05", "E09", "G05", "E09"]
    pr = np.array([20200002.0, 21000000.0, 20200003.0, 21000001.0])
    ph = np.array([106_000_000.1, np.nan, 106_000_001.1, np.nan])
    dp = np.array([45.0, 46.0, 45.0, 46.0])
    cn0 = np.array([40.0, 41.0, 40.0, 41.0])

    out_path = tmp_path / "from_arrays.obs"
    write_rinex_obs_from_arrays(
        out_path,
        epoch_times,
        sat_ids,
        pseudorange_m=pr,
        carrier_cycles=ph,
        doppler_hz=dp,
        cn0_dbhz=cn0,
    )

    obs = read_rinex_obs(out_path)
    assert len(obs.epochs) == 2
    assert obs.epochs[0].time == t0
    assert obs.epochs[0].satellites == ["G05", "E09"]
    assert obs.epochs[0].observations["G05"]["C1C"] == pytest.approx(20200002.0, abs=1e-3)
    assert obs.epochs[1].satellites == ["G05", "E09"]
    assert obs.epochs[1].observations["E09"]["C1C"] == pytest.approx(21000001.0, abs=1e-3)


_FIXTURE = _find_fixture()


@pytest.mark.skipif(_FIXTURE is None, reason="no real RINEX OBS fixture found under experiments/")
def test_read_write_read_idempotence_on_real_fixture(tmp_path):
    obs = read_rinex_obs(_FIXTURE)
    first_two = obs.epochs[:2]
    assert len(first_two) == 2

    header = RinexObsHeader(
        marker_name=obs.header.marker_name,
        approx_position_ecef=np.asarray(obs.header.approx_position, dtype=float),
        obs_types=obs.header.obs_types,
        interval_s=obs.header.interval or 1.0,
        time_first_obs=first_two[0].time,
    )

    epochs = []
    for ep in first_two:
        sat_ids = list(ep.satellites)
        codes: set[str] = set()
        for sat_obs in ep.observations.values():
            codes.update(sat_obs.keys())
        obs_arrays = {
            code: np.array([ep.observations[sat].get(code, 0.0) for sat in sat_ids])
            for code in codes
        }
        epochs.append(EpochRecord(time=ep.time, sat_ids=sat_ids, obs=obs_arrays))

    out_path = tmp_path / "roundtrip_fixture.obs"
    write_rinex_obs(out_path, header, epochs)
    reread = read_rinex_obs(out_path)

    assert len(reread.epochs) == 2
    for original, again in zip(first_two, reread.epochs):
        assert again.time == original.time
        assert again.satellites == original.satellites
        for sat in original.satellites:
            for code, value in original.observations[sat].items():
                assert again.observations[sat][code] == pytest.approx(value, abs=1e-3)
