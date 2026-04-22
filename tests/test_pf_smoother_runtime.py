import math
from pathlib import Path

import numpy as np

from gnss_gpu.pf_smoother_config import PfSmootherConfig
from gnss_gpu.pf_smoother_runtime import (
    ForwardRunBuffers,
    build_observation_computers,
    find_base_obs_path,
    initialize_imu_filter,
    initialize_particle_filter,
    resolve_run_dataset,
    spp_heading_from_velocity,
)


def _minimal_config(**overrides):
    values = {
        "n_particles": 100,
        "sigma_pos": 1.2,
        "sigma_pr": 3.0,
        "position_update_sigma": 1.9,
        "predict_guide": "imu",
        "use_smoother": True,
    }
    values.update(overrides)
    return PfSmootherConfig(**values)


def test_find_base_obs_path_prefers_trimble(tmp_path):
    base = tmp_path / "base.obs"
    base_trimble = tmp_path / "base_trimble.obs"
    base.write_text("", encoding="utf-8")
    base_trimble.write_text("", encoding="utf-8")

    assert find_base_obs_path(tmp_path) == base_trimble


def test_resolve_run_dataset_uses_loader_and_normalizes_arrays(tmp_path):
    captured = {}

    def loader(run_dir: Path, rover_source: str):
        captured["run_dir"] = run_dir
        captured["rover_source"] = rover_source
        return {
            "epochs": [("epoch", [])],
            "spp_lookup": {1: [1.0, 2.0, 3.0]},
            "gt": [[4.0, 5.0, 6.0]],
            "our_times": [1.0],
            "first_pos": [7.0, 8.0, 9.0],
            "init_cb": 10,
            "imu_data": None,
        }

    dataset = resolve_run_dataset(tmp_path, "trimble", None, loader)

    assert captured == {"run_dir": tmp_path, "rover_source": "trimble"}
    assert dataset.epochs == [("epoch", [])]
    assert list(dataset.spp_lookup) == [1.0]
    np.testing.assert_allclose(dataset.first_pos, [7.0, 8.0, 9.0])
    assert dataset.init_cb == 10.0


def test_spp_heading_from_velocity_returns_east_heading_at_equator():
    heading = spp_heading_from_velocity(
        np.array([0.0, 2.0, 0.0]),
        lat=0.0,
        lon=0.0,
    )

    assert heading == math.pi / 2
    assert spp_heading_from_velocity(np.array([0.0, 0.1, 0.0]), 0.0, 0.0) is None


def test_initialize_imu_filter_uses_dataset_imu_and_initial_spp_heading(tmp_path):
    config = _minimal_config()
    dataset = resolve_run_dataset(
        tmp_path,
        "trimble",
        {
            "epochs": [],
            "spp_lookup": {
                0.0: [6378137.0, 0.0, 0.0],
                1.0: [6378137.0, 1.0, 0.0],
            },
            "gt": [],
            "our_times": [],
            "first_pos": [6378137.0, 0.0, 0.0],
            "init_cb": 0.0,
            "imu_data": {
                "tow": np.array([0.0, 0.02]),
                "gyro": np.zeros((2, 3), dtype=np.float64),
                "wheel_vel": np.zeros(2, dtype=np.float64),
            },
        },
        lambda _run_dir, _rover_source: {},
    )

    imu_filter = initialize_imu_filter(
        tmp_path,
        config.predict_guide,
        dataset,
        lambda _x, _y, _z: (0.0, 0.0, 0.0),
    )

    assert imu_filter is not None
    assert imu_filter.heading == math.pi / 2


def test_build_observation_computers_disabled_does_not_require_base(tmp_path):
    observations = _minimal_config(
        dd_pseudorange=False,
        widelane=False,
        mupf_dd=False,
    ).observations

    computers = build_observation_computers(tmp_path, "trimble", observations)

    assert computers.base_obs_path is None
    assert computers.dd_pr_computer is None
    assert computers.wl_computer is None
    assert computers.dd_computer is None
    assert computers.carrier_bias_tracker == {}


def test_forward_run_buffers_tracks_storage_and_alignment():
    buffers = ForwardRunBuffers()

    buffers.append_smoother_motion(
        np.array([1.0, 2.0, 3.0]),
        np.array([4.0, 5.0, 6.0]),
        dt=0.5,
        need_tdcp_motion=True,
    )
    buffers.append_smoother_observations("dd0", "pr0", "undiff0")
    buffers.append_smoother_motion(
        np.array([1.0, 2.0, 3.0]),
        None,
        dt=0.5,
        need_tdcp_motion=True,
    )
    buffers.append_smoother_observations("dd1", "pr1", "undiff1")
    aligned_index = buffers.append_aligned_epoch(
        np.array([10.0, 11.0, 12.0]),
        np.array([13.0, 14.0, 15.0]),
        stop_flag=True,
        use_smoother=True,
    )

    assert buffers.n_stored == 2
    assert aligned_index == 0
    assert buffers.aligned_indices == [1]
    assert buffers.aligned_stop_flags == [True]
    assert len(buffers.stored_motion_deltas) == 1
    assert len(buffers.stored_tdcp_motion_deltas) == 1
    np.testing.assert_allclose(buffers.stored_motion_deltas[0], [0.5, 1.0, 1.5])
    assert np.isnan(buffers.stored_tdcp_motion_deltas[0]).all()


def test_initialize_particle_filter_uses_grouped_runtime_config(monkeypatch):
    captured = {}

    class FakeParticleFilter:
        def __init__(self, **kwargs):
            captured["ctor"] = kwargs

        def initialize(self, first_pos, **kwargs):
            captured["first_pos"] = np.asarray(first_pos, dtype=np.float64)
            captured["initialize"] = kwargs

        def enable_smoothing(self):
            captured["enable_smoothing"] = True

    monkeypatch.setattr(
        "gnss_gpu.pf_smoother_runtime.ParticleFilterDevice",
        FakeParticleFilter,
    )
    parts = _minimal_config(
        rbpf_velocity_kf=True,
        rbpf_velocity_init_sigma=2.5,
        rbpf_velocity_process_noise=0.75,
        pf_init_spread_vel=3.0,
        per_particle_huber=True,
    ).parts()

    pf = initialize_particle_filter(
        np.array([1.0, 2.0, 3.0]),
        4.0,
        parts.particle_filter,
        parts.observations.robust,
        parts.doppler,
        sigma_cb=5.0,
        seed=6,
    )

    assert isinstance(pf, FakeParticleFilter)
    assert captured["ctor"]["n_particles"] == 100
    assert captured["ctor"]["sigma_cb"] == 5.0
    assert captured["ctor"]["seed"] == 6
    assert captured["ctor"]["rbpf_velocity_kf"] is True
    assert captured["ctor"]["velocity_process_noise"] == 0.75
    assert captured["ctor"]["per_particle_huber"] is True
    np.testing.assert_allclose(captured["first_pos"], [1.0, 2.0, 3.0])
    assert captured["initialize"]["clock_bias"] == 4.0
    assert captured["initialize"]["spread_vel"] == 0.0
    assert captured["initialize"]["velocity_init_sigma"] == 2.5
    assert captured["enable_smoothing"] is True
