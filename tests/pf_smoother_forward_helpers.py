from pathlib import Path

import numpy as np

from gnss_gpu.epoch_gate_state import EpochGateState
from gnss_gpu.epoch_observation_inputs import EpochObservationInputs
from gnss_gpu.pf_smoother_config import PfSmootherConfig
from gnss_gpu.pf_smoother_epoch_history import ForwardEpochHistory
from gnss_gpu.pf_smoother_epoch_measurements import ForwardEpochMeasurementInputs
from gnss_gpu.pf_smoother_forward_context import PfSmootherForwardPassContext
from gnss_gpu.pf_smoother_forward_stats import ForwardRunStats
from gnss_gpu.pf_smoother_run_context import (
    PfSmootherRunDependencies,
    build_pf_smoother_run_options,
)
from gnss_gpu.pf_smoother_runtime import (
    ForwardRunBuffers,
    ObservationComputers,
    RunDataset,
)


class FakePf:
    def __init__(self, calls):
        self.calls = calls

    def estimate(self):
        return np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)

    def predict(self, **kwargs):
        self.calls.append(("pf_predict", kwargs))


def base_config(**overrides):
    values = {
        "n_particles": 10,
        "sigma_pos": 1.0,
        "sigma_pr": 3.0,
        "position_update_sigma": 2.0,
        "predict_guide": "spp",
        "use_smoother": True,
    }
    values.update(overrides)
    return PfSmootherConfig(**values)


def make_forward_context(calls, *, config=None):
    if config is None:
        config = base_config()
    return PfSmootherForwardPassContext(
        run_name="Odaiba",
        run_config=config,
        config_parts=config.parts(),
        run_options=build_pf_smoother_run_options(config),
        dependencies=PfSmootherRunDependencies(
            load_dataset_func=lambda run_dir, rover_source: {},
            ecef_to_lla_func=lambda x, y, z: (0.0, 0.0, 0.0),
            compute_metrics_func=lambda *args, **kwargs: {},
            ecef_errors_func=lambda *args, **kwargs: (
                np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.float64),
            ),
            sigma_cb=1.0,
        ),
        dataset=RunDataset(
            epochs=[],
            spp_lookup={123.4: np.zeros(4, dtype=np.float64)},
            gt=np.empty((0, 3), dtype=np.float64),
            our_times=np.empty(0, dtype=np.float64),
            first_pos=np.zeros(3, dtype=np.float64),
            init_cb=0.0,
        ),
        imu_filter=None,
        pf=FakePf(calls),
        buffers=ForwardRunBuffers(),
        stats=ForwardRunStats(),
        history=ForwardEpochHistory(),
        observation_setup=ObservationComputers(base_obs_path=Path("/tmp/base.obs")),
        pr_history={},
    )


def make_measurement_inputs():
    sat_ecef = np.zeros((4, 3), dtype=np.float64)
    pseudoranges = np.ones(4, dtype=np.float64)
    weights = np.ones(4, dtype=np.float64)
    gate_pf_estimate = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    return ForwardEpochMeasurementInputs(
        observation_inputs=EpochObservationInputs(
            sat_ecef=sat_ecef,
            pseudoranges=pseudoranges,
            weights=weights,
            carrier_anchor_rows={},
        ),
        gate_state=EpochGateState(
            pf_estimate=gate_pf_estimate,
            ess_ratio=0.9,
            spread_m=1.5,
            dd_pr_gate_scale=1.0,
            dd_cp_gate_scale=1.0,
        ),
        sat_ecef=sat_ecef,
        pseudoranges=pseudoranges,
        weights=weights,
        gate_pf_estimate=gate_pf_estimate,
        gate_ess_ratio=0.9,
        gate_spread_m=1.5,
    )
