import numpy as np
import pytest

from gnss_gpu.pf_smoother_config import PfSmootherConfig
from gnss_gpu.pf_smoother_run_context import (
    PfSmootherRunDependencies,
    build_pf_smoother_run_options,
)


def _base_config(**overrides):
    values = {
        "n_particles": 10,
        "sigma_pos": 1.0,
        "sigma_pr": 3.0,
        "position_update_sigma": None,
        "predict_guide": "spp",
        "use_smoother": False,
    }
    values.update(overrides)
    return PfSmootherConfig(**values)


def test_build_pf_smoother_run_options_normalizes_fgo_motion_and_builds_predict_options():
    options = build_pf_smoother_run_options(
        _base_config(
            fgo_local_window="auto",
            fgo_local_motion_source=" Prefer_TDCP ",
            tdcp_position_update=True,
            tdcp_elevation_weight=True,
            tdcp_el_sin_floor=0.2,
            tdcp_rms_threshold=2.5,
            tdcp_pu_rms_max=1.5,
            tdcp_pu_spp_max_diff_mps=4.0,
            fgo_local_tdcp_rms_max_m=2.0,
            fgo_local_tdcp_spp_max_diff_mps=3.0,
        )
    )

    assert options.fgo_motion_source == "prefer_tdcp"
    assert options.need_fgo_tdcp_motion is True
    assert options.predict_motion_options.need_fgo_tdcp_motion is True
    assert options.predict_motion_options.tdcp_position_update is True
    assert options.predict_motion_options.tdcp_elevation_weight is True
    assert options.predict_motion_options.tdcp_el_sin_floor == 0.2
    assert options.predict_motion_options.tdcp_rms_threshold == 2.5
    assert options.predict_motion_options.tdcp_pu_rms_max == 1.5
    assert options.predict_motion_options.tdcp_pu_spp_max_diff_mps == 4.0
    assert options.predict_motion_options.fgo_local_tdcp_rms_max_m == 2.0
    assert options.predict_motion_options.fgo_local_tdcp_spp_max_diff_mps == 3.0


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"dd_pseudorange": True, "use_gmm": True}, "dd_pseudorange"),
        (
            {"rbpf_velocity_kf": True, "doppler_per_particle": True},
            "mutually exclusive",
        ),
        ({"fgo_local_motion_source": "bad"}, "fgo_local_motion_source"),
        ({"tdcp_pu_gate_logic": "bad"}, "tdcp_pu_gate_logic"),
        ({"tdcp_pu_gate_stop_mode": "bad"}, "tdcp_pu_gate_stop_mode"),
    ],
)
def test_build_pf_smoother_run_options_rejects_invalid_modes(overrides, match):
    with pytest.raises(ValueError, match=match):
        build_pf_smoother_run_options(_base_config(**overrides))


def test_pf_smoother_run_dependencies_carries_external_functions():
    deps = PfSmootherRunDependencies(
        load_dataset_func=lambda run_dir, rover_source: {},
        ecef_to_lla_func=lambda x, y, z: (1.0, 2.0, 3.0),
        compute_metrics_func=lambda *args, **kwargs: {"ok": True},
        ecef_errors_func=lambda *args, **kwargs: (
            np.array([1.0]),
            np.array([2.0]),
        ),
        sigma_cb=9.0,
        seed=11,
    )

    assert deps.ecef_to_lla_func(0.0, 0.0, 0.0) == (1.0, 2.0, 3.0)
    assert deps.compute_metrics_func()["ok"] is True
    assert deps.sigma_cb == 9.0
    assert deps.seed == 11
