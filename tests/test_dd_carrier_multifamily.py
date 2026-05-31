import numpy as np
import pytest

from gnss_gpu.dd_carrier import (
    DDCarrierComputer,
    GPS_L1_WAVELENGTH,
    GPS_L5_WAVELENGTH,
)


class _Meas:
    def __init__(self, system_id, prn, sat, elevation=0.7, snr=45.0):
        self.system_id = system_id
        self.prn = prn
        self.satellite_ecef = np.asarray(sat, dtype=np.float64)
        self.elevation = elevation
        self.snr = snr


def _computer_with_obs():
    comp = DDCarrierComputer.__new__(DDCarrierComputer)
    comp._allowed_systems = ("G",)
    comp._interpolate_base_epochs = False
    comp._base_pos = np.zeros(3, dtype=np.float64)
    comp._base_tow_keys = np.asarray([10.0], dtype=np.float64)
    comp._base_by_tow = {
        10.0: {
            "G01": {"L1C": 1000.0, "L5Q": 2000.0},
            "G02": {"L1C": 1100.0, "L5Q": 2110.0},
            "G03": {"L1C": 1200.0, "L5Q": 2230.0},
        }
    }
    comp._rover_by_tow = {
        10.0: {
            "G01": {"L1C": 3000.0, "L5Q": 4000.0},
            "G02": {"L1C": 3110.0, "L5Q": 4120.0},
            "G03": {"L1C": 3220.0, "L5Q": 4250.0},
        }
    }
    return comp


def _measurements():
    return [
        _Meas(0, 1, [20_200_000.0, 0.0, 0.0], elevation=0.9),
        _Meas(0, 2, [20_300_000.0, 1000.0, 0.0], elevation=0.6),
        _Meas(0, 3, [20_400_000.0, 2000.0, 0.0], elevation=0.5),
    ]


def test_compute_dd_families_emits_l1_and_l5_rows():
    result = _computer_with_obs().compute_dd_families(
        10.0,
        _measurements(),
        min_common_sats=2,
        carrier_families=("L1_E1_B1", "L5_E5A_B2A"),
    )

    assert result is not None
    assert result.n_dd == 4
    np.testing.assert_allclose(
        sorted(set(np.round(result.wavelengths_m, 12))),
        sorted(
            {
                round(GPS_L1_WAVELENGTH, 12),
                round(GPS_L5_WAVELENGTH, 12),
            }
        ),
    )
    assert any(s.endswith("@L1_E1_B1") for s in result.sat_ids)
    assert any(s.endswith("@L5_E5A_B2A") for s in result.sat_ids)


def test_compute_dd_families_requires_rover_rinex_cache():
    comp = _computer_with_obs()
    comp._rover_by_tow = None

    assert comp.compute_dd_families(10.0, _measurements()) is None


def test_compute_dd_families_rejects_unknown_family():
    with pytest.raises(ValueError, match="unknown carrier family"):
        _computer_with_obs().compute_dd_families(
            10.0,
            _measurements(),
            carrier_families=("NOPE",),
        )
