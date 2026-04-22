from pathlib import Path

import numpy as np

import gnss_gpu.pf_smoother_forward_loop as loop_mod
from gnss_gpu.pf_smoother_config import PfSmootherConfig
from gnss_gpu.pf_smoother_epoch_history import ForwardEpochHistory
from gnss_gpu.pf_smoother_forward_context import PfSmootherForwardPassContext
from gnss_gpu.pf_smoother_forward_loop import run_pf_smoother_forward_pass
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


class _InvalidEpoch:
    def is_valid(self):
        return False


class _ValidEpoch:
    def is_valid(self):
        return True


class _PfShouldNotBeUsed:
    def estimate(self):
        raise AssertionError("invalid epochs should be skipped before PF access")


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


def _context_for_epochs(epochs):
    config = _base_config()
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
            epochs=epochs,
            spp_lookup={},
            gt=np.empty((0, 3), dtype=np.float64),
            our_times=np.empty(0, dtype=np.float64),
            first_pos=np.zeros(3, dtype=np.float64),
            init_cb=0.0,
        ),
        imu_filter=None,
        pf=_PfShouldNotBeUsed(),
        buffers=ForwardRunBuffers(),
        stats=ForwardRunStats(),
        history=ForwardEpochHistory(),
        observation_setup=ObservationComputers(base_obs_path=Path("/tmp/base.obs")),
        pr_history={},
    )


def test_run_pf_smoother_forward_pass_handles_empty_epoch_sequence():
    context = _context_for_epochs([])

    elapsed_ms = run_pf_smoother_forward_pass(context)

    assert elapsed_ms >= 0.0
    assert context.buffers.forward_aligned == []
    assert context.stats.as_result_context()["n_dd_used"] == 0


def test_run_pf_smoother_forward_pass_skips_invalid_epochs_before_pf_access():
    context = _context_for_epochs([(_InvalidEpoch(), [object(), object(), object(), object()])])

    elapsed_ms = run_pf_smoother_forward_pass(context)

    assert elapsed_ms >= 0.0
    assert context.history.prev_tow is None


def test_run_pf_smoother_forward_pass_dispatches_valid_epochs(monkeypatch):
    epoch = _ValidEpoch()
    measurements = [object(), object(), object(), object()]
    context = _context_for_epochs([(epoch, measurements)])
    calls = []

    def fake_process(ctx, sol_epoch, epoch_measurements):
        calls.append((ctx, sol_epoch, epoch_measurements))

    monkeypatch.setattr(loop_mod, "process_pf_smoother_forward_epoch", fake_process)

    elapsed_ms = run_pf_smoother_forward_pass(context)

    assert elapsed_ms >= 0.0
    assert calls == [(context, epoch, measurements)]


def test_run_pf_smoother_forward_pass_respects_history_limit(monkeypatch):
    context = _context_for_epochs([(_ValidEpoch(), [object(), object(), object(), object()])])
    context.run_config = _base_config(max_epochs=1)
    context.config_parts = context.run_config.parts()
    context.history.epochs_done = 1

    def fake_process(*args, **kwargs):
        raise AssertionError("forward epoch should not process after max_epochs")

    monkeypatch.setattr(loop_mod, "process_pf_smoother_forward_epoch", fake_process)

    elapsed_ms = run_pf_smoother_forward_pass(context)

    assert elapsed_ms >= 0.0
