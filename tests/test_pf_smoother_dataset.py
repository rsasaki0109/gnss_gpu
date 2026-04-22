from types import SimpleNamespace

import numpy as np
import pytest

from gnss_gpu.pf_smoother_dataset import load_pf_smoother_dataset


class _Solution:
    def __init__(self, records):
        self._records = records

    def records(self):
        return self._records


def _record(tow, position, valid=True):
    return SimpleNamespace(
        time=SimpleNamespace(tow=tow),
        position_ecef_m=position,
        is_valid=lambda: valid,
    )


def _epoch(tow, position, valid=True):
    return SimpleNamespace(
        time=SimpleNamespace(tow=tow),
        position_ecef_m=position,
        is_valid=lambda: valid,
    )


def _measurement(sat_ecef, corrected_pseudorange):
    return SimpleNamespace(
        satellite_ecef=np.asarray(sat_ecef, dtype=np.float64),
        corrected_pseudorange=float(corrected_pseudorange),
    )


def test_load_pf_smoother_dataset_builds_lookup_init_clock_and_imu(tmp_path):
    (tmp_path / "imu.csv").write_text("placeholder", encoding="utf-8")
    captured = {}
    init_pos = np.array([1000.0, 2000.0, 3000.0])
    measurements = [
        _measurement([1001.0, 2000.0, 3000.0], 11.0),
        _measurement([1000.0, 2002.0, 3000.0], 22.0),
        _measurement([1000.0, 2000.0, 3003.0], 33.0),
        _measurement([1004.0, 2000.0, 3000.0], 44.0),
    ]
    epochs = [(_epoch(12.3, init_pos), measurements)]

    def preprocess(obs_path, nav_path):
        captured["preprocess"] = (obs_path, nav_path)
        return epochs

    def solve(obs_path, nav_path):
        captured["solve"] = (obs_path, nav_path)
        return _Solution(
            [
                _record(12.34, [1.0, 2.0, 3.0]),
                _record(13.0, [4.0, 5.0, 6.0], valid=False),
            ]
        )

    def urban_loader(run_dir, *, systems, urban_rover):
        captured["urban"] = (run_dir, systems, urban_rover)
        return {
            "ground_truth": np.array([[7.0, 8.0, 9.0]]),
            "times": np.array([12.3]),
        }

    def imu_loader(path):
        captured["imu"] = path
        return {"tow": np.array([1.0, 2.0])}

    dataset = load_pf_smoother_dataset(
        tmp_path,
        rover_source="ublox",
        urban_data_loader=urban_loader,
        preprocess_spp_file_func=preprocess,
        solve_spp_file_func=solve,
        imu_loader=imu_loader,
    )

    assert captured["preprocess"] == (
        str(tmp_path / "rover_ublox.obs"),
        str(tmp_path / "base.nav"),
    )
    assert captured["solve"] == captured["preprocess"]
    assert captured["urban"] == (tmp_path, ("G", "E", "J"), "ublox")
    assert captured["imu"] == tmp_path / "imu.csv"
    assert dataset["epochs"] is epochs
    assert list(dataset["spp_lookup"]) == [12.3]
    np.testing.assert_allclose(dataset["spp_lookup"][12.3], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(dataset["first_pos"], init_pos)
    assert dataset["init_cb"] == pytest.approx(np.median([10.0, 20.0, 30.0, 40.0]))
    assert dataset["imu_data"]["tow"].tolist() == [1.0, 2.0]


def test_load_pf_smoother_dataset_rejects_missing_valid_init_epoch(tmp_path):
    def urban_loader(run_dir, *, systems, urban_rover):
        return {"ground_truth": np.zeros((0, 3)), "times": np.zeros(0)}

    with pytest.raises(RuntimeError, match="No valid epoch"):
        load_pf_smoother_dataset(
            tmp_path,
            urban_data_loader=urban_loader,
            preprocess_spp_file_func=lambda _obs, _nav: [(_epoch(1.0, [1, 2, 3]), [])],
            solve_spp_file_func=lambda _obs, _nav: _Solution(
                [_record(1.0, [1.0, 2.0, 3.0])]
            ),
        )
