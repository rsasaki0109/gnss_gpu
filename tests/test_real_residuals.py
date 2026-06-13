from types import SimpleNamespace

import numpy as np
import pytest

from gnss_gpu.validation.real_residuals import (
    collect_residual_samples,
    elevation_azimuth,
    epoch_residuals,
    estimate_clock_bias,
    prn_to_int,
    residual_array,
    residual_samples_from_epoch,
    iono_delays,
    residual_samples_from_experiment_data,
    tropo_delays,
)


def test_prn_to_int():
    assert prn_to_int("G01") == 1
    assert prn_to_int("E12") == 12
    assert prn_to_int("R5") == 5
    assert prn_to_int(7) == 7
    assert prn_to_int("G 5") == 5

    with pytest.raises(ValueError):
        prn_to_int("G")


def test_estimate_clock_bias():
    assert estimate_clock_bias([1.0, 2.0, 3.0], "median") == 2.0
    assert estimate_clock_bias([1.0, 2.0, 3.0], "mean") == 2.0
    assert estimate_clock_bias([1.0, np.nan, 3.0], "median") == 2.0
    assert estimate_clock_bias([1.0, np.nan, 3.0], "mean") == 2.0

    with pytest.raises(ValueError):
        estimate_clock_bias([])

    with pytest.raises(ValueError):
        estimate_clock_bias([np.nan, np.nan])


def test_epoch_residuals_removes_median_clock_bias():
    rx = np.array([6378137.0, 0.0, 0.0])
    sat = np.array(
        [
            [6378137.0 + 20200000.0, 0.0, 0.0],
            [6378137.0, 20200000.0, 0.0],
            [6378137.0, 0.0, 20200000.0],
        ]
    )
    geom = np.linalg.norm(sat - rx[None, :], axis=1)
    sat_clock = np.array([5.0, -12.0, 20.0])
    per_sat = np.array([0.0, 30.0, -10.0])
    common_clock = 1000.0

    pseudorange = geom + common_clock + per_sat - sat_clock
    residual, bias = epoch_residuals(
        pseudorange,
        sat,
        rx,
        sat_clock_m=sat_clock,
        clock_bias=None,
    )

    assert bias == pytest.approx(common_clock + np.median(per_sat))
    np.testing.assert_allclose(residual, per_sat - np.median(per_sat))


def test_elevation_azimuth_zenith():
    rx = np.array([6378137.0, 0.0, 0.0])
    sat = np.array([[6378137.0 + 20200000.0, 0.0, 0.0]])

    elevation, azimuth = elevation_azimuth(rx, sat)

    assert elevation[0] == pytest.approx(np.pi / 2.0, abs=1e-12)
    assert np.isfinite(azimuth[0])


def test_residual_samples_from_epoch_and_elevation_mask():
    rx = np.array([6378137.0, 0.0, 0.0])
    sat = np.array(
        [
            [6378137.0 + 20200000.0, 0.0, 0.0],
            [6378137.0, 20200000.0, 0.0],
            [6378137.0, 0.0, 20200000.0],
        ]
    )
    geom = np.linalg.norm(sat - rx[None, :], axis=1)
    pseudorange = geom + 1000.0 + np.array([0.0, 30.0, -10.0])

    samples = residual_samples_from_epoch(
        123.0,
        ["G01", "E12", "R5"],
        pseudorange,
        sat,
        rx,
        cn0=np.array([40.0, 41.0, 42.0]),
    )

    assert len(samples) == 3
    assert [sample.prn for sample in samples] == [1, 12, 5]
    np.testing.assert_allclose(
        [sample.residual_m for sample in samples],
        np.array([0.0, 30.0, -10.0]) - np.median([0.0, 30.0, -10.0]),
    )

    masked = residual_samples_from_epoch(
        123.0,
        ["G01", "E12", "R5"],
        pseudorange,
        sat,
        rx,
        elevation_mask_rad=0.1,
    )
    assert len(masked) == 1
    assert masked[0].prn == 1


def test_collect_residual_samples_skips_missing_lookup():
    rx = np.array([6378137.0, 0.0, 0.0])
    sat_by_prn = {
        "G01": np.array([6378137.0 + 20200000.0, 0.0, 0.0]),
        "G02": np.array([6378137.0, 20200000.0, 0.0]),
        "G03": np.array([6378137.0, 0.0, 20200000.0]),
    }

    def make_obs(time, prns):
        sat = np.vstack([sat_by_prn[p] for p in prns])
        geom = np.linalg.norm(sat - rx[None, :], axis=1)
        return SimpleNamespace(
            time=time,
            prn=prns,
            pseudorange=geom + 1000.0,
            cn0=np.full(len(prns), 45.0),
        )

    epochs = [
        (make_obs(1.0, ["G01", "G02"]), rx),
        (make_obs(2.0, ["G02", "G03"]), rx),
    ]

    def sat_lookup(time, prn):
        if time == 2.0 and prn == "G03":
            return None
        return sat_by_prn[prn], 0.0

    samples = collect_residual_samples(epochs, sat_lookup)

    assert len(samples) == 3
    assert [sample.prn for sample in samples] == [1, 2, 2]


def test_residual_array():
    rx = np.array([6378137.0, 0.0, 0.0])
    sat = np.array(
        [
            [6378137.0 + 20200000.0, 0.0, 0.0],
            [6378137.0, 20200000.0, 0.0],
        ]
    )
    geom = np.linalg.norm(sat - rx[None, :], axis=1)
    pseudorange = geom + 1000.0 + np.array([10.0, -10.0])

    samples = residual_samples_from_epoch(
        1.0,
        ["G01", "G02"],
        pseudorange,
        sat,
        rx,
    )
    arr = residual_array(samples)

    assert arr.shape == (2,)
    np.testing.assert_allclose(arr, [10.0, -10.0])


def test_residual_samples_from_experiment_data():
    rx = np.array([6378137.0, 0.0, 0.0])
    sat0 = np.array([
        [6378137.0 + 20200000.0, 0.0, 0.0],
        [6378137.0, 20200000.0, 0.0],
        [6378137.0, 0.0, 20200000.0],
    ])
    sat1 = sat0 + np.array([10.0, -5.0, 3.0])
    geom0 = np.linalg.norm(sat0 - rx[None, :], axis=1)
    geom1 = np.linalg.norm(sat1 - rx[None, :], axis=1)
    # pseudoranges already include sat clock (adapter passes sat_clock_m=None).
    pr0 = geom0 + 500.0 + np.array([0.0, 40.0, -20.0])
    pr1 = geom1 + 500.0 + np.array([0.0, 15.0, -5.0])

    data = {
        "sat_ecef": [sat0, sat1],
        "pseudoranges": [pr0, pr1],
        "ground_truth": np.vstack([rx, rx]),
        "times": np.array([100.0, 101.0]),
        "used_prns": [["G01", "G 5", "G24"], ["G01", "G 5", "G24"]],
        "weights": [np.array([45.0, 30.0, 38.0]), np.array([44.0, 31.0, 37.0])],
    }

    samples = residual_samples_from_experiment_data(data)
    assert len(samples) == 6
    assert [s.prn for s in samples[:3]] == [1, 5, 24]
    # Epoch 0 residuals = per_sat - median(per_sat), median([0,40,-20]) = 0.
    np.testing.assert_allclose(
        [s.residual_m for s in samples[:3]], [0.0, 40.0, -20.0])
    # Elevation mask drops low-elevation satellites.
    masked = residual_samples_from_experiment_data(data, elevation_mask_rad=0.1)
    assert all(s.prn == 1 for s in masked)


def test_tropo_delays_positive_and_elevation_dependent():
    rx = np.array([-3963426.8, 3350882.2, 3694865.5])  # near Tokyo surface
    # Build satellites at decreasing elevation by tilting away from local up.
    up = rx / np.linalg.norm(rx)
    east = np.cross([0.0, 0.0, 1.0], up); east /= np.linalg.norm(east)
    sats = []
    for el_deg in (85.0, 45.0, 15.0, 7.0):
        el = np.radians(el_deg)
        d = np.sin(el) * up + np.cos(el) * east
        sats.append(rx + d * 2.2e7)
    sats = np.vstack(sats)

    tropo = tropo_delays(rx, sats)
    assert np.all(tropo > 0.0)
    # Zenith-ish delay is a few metres; low elevation is much larger.
    assert tropo[0] < 5.0
    assert tropo[-1] > tropo[0]
    assert np.all(np.diff(tropo) > 0.0)  # monotonic increase toward horizon


def test_epoch_residuals_subtracts_atmo_before_clock():
    rx = np.array([6378137.0, 0.0, 0.0])
    sat = np.array([
        [6378137.0 + 20200000.0, 0.0, 0.0],
        [6378137.0, 20200000.0, 0.0],
        [6378137.0, 0.0, 20200000.0],
    ])
    geom = np.linalg.norm(sat - rx[None, :], axis=1)
    atmo = np.array([3.0, 8.0, 12.0])
    per_sat = np.array([0.0, 25.0, -10.0])
    common = 1000.0
    pr = geom + common + atmo + per_sat

    res, bias = epoch_residuals(pr, sat, rx, atmo_delay_m=atmo)
    # atmo removed, then median(per_sat)=0 removed.
    np.testing.assert_allclose(res, per_sat - np.median(per_sat), atol=1e-9)


def test_apply_tropo_purifies_low_elevation_bias():
    rx = np.array([-3963426.8, 3350882.2, 3694865.5])
    up = rx / np.linalg.norm(rx)
    east = np.cross([0.0, 0.0, 1.0], up); east /= np.linalg.norm(east)
    north = np.cross(up, east)
    sats = []
    for el_deg, az_deg in [(80, 0), (50, 90), (20, 180), (10, 270), (35, 45)]:
        el = np.radians(el_deg); az = np.radians(az_deg)
        d = np.sin(el) * up + np.cos(el) * (np.cos(az) * north + np.sin(az) * east)
        sats.append(rx + d * 2.2e7)
    sats = np.vstack(sats)
    geom = np.linalg.norm(sats - rx[None, :], axis=1)

    tropo = tropo_delays(rx, sats)
    multipath = np.array([0.0, 2.0, 0.0, 5.0, 1.0])
    common = 1.0e5
    pr = geom + common + tropo + multipath  # sat clock folded in (=0 here)

    data = {
        "sat_ecef": [sats],
        "pseudoranges": [pr],
        "ground_truth": np.vstack([rx]),
        "times": np.array([0.0]),
        "used_prns": [["G01", "G02", "G03", "G04", "G05"]],
    }

    raw = residual_array(residual_samples_from_experiment_data(data))
    purified = residual_array(
        residual_samples_from_experiment_data(data, apply_tropo=True))

    # With tropo removed, residuals collapse toward the injected multipath
    # (minus its median) and the spread shrinks dramatically.
    assert np.std(purified) < np.std(raw)
    np.testing.assert_allclose(
        purified, multipath - np.median(multipath), atol=1e-6)


def _tokyo_sats(rx, el_az_list):
    up = rx / np.linalg.norm(rx)
    east = np.cross([0.0, 0.0, 1.0], up); east /= np.linalg.norm(east)
    north = np.cross(up, east)
    sats = []
    for el_deg, az_deg in el_az_list:
        el = np.radians(el_deg); az = np.radians(az_deg)
        d = np.sin(el) * up + np.cos(el) * (np.cos(az) * north + np.sin(az) * east)
        sats.append(rx + d * 2.2e7)
    return np.vstack(sats)


def test_iono_delays_positive_and_elevation_dependent():
    rx = np.array([-3963426.8, 3350882.2, 3694865.5])
    sats = _tokyo_sats(rx, [(85, 0), (45, 90), (15, 180), (7, 270)])
    iono = iono_delays(rx, sats, gps_time=259200.0)
    assert np.all(iono > 0.0)
    # Obliquity factor grows toward the horizon -> larger slant iono.
    assert iono[-1] > iono[0]


def test_iono_delays_default_vs_custom_params_differ():
    rx = np.array([-3963426.8, 3350882.2, 3694865.5])
    sats = _tokyo_sats(rx, [(60, 30), (30, 200)])
    default = iono_delays(rx, sats, gps_time=259200.0)
    custom = iono_delays(
        rx, sats, gps_time=259200.0,
        alpha=[2.0e-8, 0.0, 0.0, 0.0], beta=[1.2e5, 0.0, 0.0, 0.0])
    assert not np.allclose(default, custom)
    assert np.all(custom > 0.0)


def test_apply_iono_shifts_residual_distribution():
    rx = np.array([-3963426.8, 3350882.2, 3694865.5])
    sats = _tokyo_sats(rx, [(80, 0), (50, 90), (20, 180), (10, 270), (35, 45)])
    geom = np.linalg.norm(sats - rx[None, :], axis=1)
    iono = iono_delays(rx, sats, gps_time=259200.0)
    multipath = np.array([0.0, 3.0, 0.0, 6.0, 1.0])
    pr = geom + 1.0e5 + iono + multipath

    data = {
        "sat_ecef": [sats],
        "pseudoranges": [pr],
        "ground_truth": np.vstack([rx]),
        "times": np.array([259200.0]),
        "used_prns": [["G01", "G02", "G03", "G04", "G05"]],
    }

    raw = residual_array(residual_samples_from_experiment_data(data))
    purified = residual_array(
        residual_samples_from_experiment_data(data, apply_iono=True))
    # Iono removed -> residuals collapse to the injected multipath (minus median).
    np.testing.assert_allclose(
        purified, multipath - np.median(multipath), atol=1e-6)
    assert np.std(purified) < np.std(raw)
