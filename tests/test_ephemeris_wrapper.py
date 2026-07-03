"""CPU-side validation tests for the Ephemeris wrapper."""

from datetime import datetime

import pytest

from gnss_gpu.ephemeris import Ephemeris
from gnss_gpu.io.nav_rinex import NavMessage


def _make_reference_nav() -> NavMessage:
    return NavMessage(
        prn=1,
        toc=datetime(2024, 1, 15, 0, 0, 0),
        af0=-3.930553793907e-05,
        af1=-1.023181539495e-12,
        af2=0.0,
        sqrt_a=5153.637939453125,
        e=0.005765914916992,
        i0=0.973496578994,
        omega0=-0.249523028508,
        omega=0.685940414073,
        M0=1.245932843990,
        delta_n=4.623016997497e-09,
        omega_dot=-8.120689826012e-09,
        idot=1.132502065007e-10,
        cuc=-3.464519977570e-06,
        cus=7.525086402893e-06,
        crc=224.15625,
        crs=-14.03125,
        cic=-1.303851604462e-07,
        cis=5.587935447693e-08,
        toe=518400.0,
        week=2295,
        tgd=-1.117587089539e-08,
        toc_seconds=518400.0,
    )


def test_compute_rejects_nonfinite_gps_time():
    nav = _make_reference_nav()
    eph = Ephemeris({1: [nav]})

    with pytest.raises(ValueError, match="gps_time must be finite"):
        eph.compute(float("nan"))


def test_compute_batch_rejects_nonfinite_gps_times():
    nav = _make_reference_nav()
    eph = Ephemeris({1: [nav]})

    with pytest.raises(ValueError, match="gps_times must be finite"):
        eph.compute_batch([518400.0, float("inf")])
