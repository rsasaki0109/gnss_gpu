from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import gnss_gpu.pf_smoother_run as run_mod
from gnss_gpu.pf_smoother_config import PfSmootherConfig
from gnss_gpu.pf_smoother_run import (
    PfSmootherRunDependencies,
    run_pf_smoother_evaluation,
)
from gnss_gpu.pf_smoother_runtime import RunDataset


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


def _empty_dataset():
    return RunDataset(
        epochs=[],
        spp_lookup={},
        gt=np.empty((0, 3), dtype=np.float64),
        our_times=np.empty(0, dtype=np.float64),
        first_pos=np.array([1.0, 2.0, 3.0], dtype=np.float64),
        init_cb=4.0,
    )


def _deps(loader):
    return PfSmootherRunDependencies(
        load_dataset_func=loader,
        ecef_to_lla_func=lambda x, y, z: (0.0, 0.0, 0.0),
        compute_metrics_func=lambda *args, **kwargs: {},
        ecef_errors_func=lambda *args, **kwargs: (
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
        ),
        sigma_cb=12.5,
        seed=7,
    )


def test_run_pf_smoother_evaluation_wires_runtime_dependencies(monkeypatch):
    captured = {}

    def fake_initialize_particle_filter(
        first_pos,
        init_cb,
        particle_filter_config,
        robust_config,
        doppler_config,
        *,
        sigma_cb,
        seed,
    ):
        captured["first_pos"] = np.asarray(first_pos)
        captured["init_cb"] = init_cb
        captured["n_particles"] = particle_filter_config.n_particles
        captured["sigma_cb"] = sigma_cb
        captured["seed"] = seed
        return object()

    def fake_finalize_postrun(result, pf, buffers, **kwargs):
        captured["result"] = result
        captured["postrun_kwargs"] = kwargs
        return SimpleNamespace(result={**result, "postrun": True})

    monkeypatch.setattr(run_mod, "initialize_particle_filter", fake_initialize_particle_filter)
    monkeypatch.setattr(run_mod, "finalize_pf_smoother_postrun", fake_finalize_postrun)
    monkeypatch.setattr(run_mod, "print_pf_smoother_run_summary", lambda result: None)

    out = run_pf_smoother_evaluation(
        Path("/tmp/UrbanNav-Tokyo/Odaiba"),
        "Odaiba",
        dataset=_empty_dataset(),
        run_config=_base_config(n_particles=25),
        dependencies=_deps(lambda *_args: pytest.fail("loader should not run")),
    )

    assert out["postrun"] is True
    assert out["run"] == "Odaiba"
    assert out["n_particles"] == 25
    assert out["n_dd_used"] == 0
    np.testing.assert_allclose(captured["first_pos"], [1.0, 2.0, 3.0])
    assert captured["init_cb"] == 4.0
    assert captured["sigma_cb"] == 12.5
    assert captured["seed"] == 7
    assert captured["postrun_kwargs"]["use_smoother"] is False
    assert captured["postrun_kwargs"]["position_update_sigma"] is None


def test_run_pf_smoother_evaluation_loads_dataset_when_not_supplied(monkeypatch):
    captured = {}

    monkeypatch.setattr(run_mod, "initialize_particle_filter", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        run_mod,
        "finalize_pf_smoother_postrun",
        lambda result, pf, buffers, **kwargs: SimpleNamespace(result=result),
    )
    monkeypatch.setattr(run_mod, "print_pf_smoother_run_summary", lambda result: None)

    def loader(run_dir, rover_source):
        captured["loader"] = (run_dir, rover_source)
        return _empty_dataset()

    run_pf_smoother_evaluation(
        Path("/tmp/UrbanNav-Tokyo/Odaiba"),
        "Odaiba",
        dataset=None,
        run_config=_base_config(rover_source="septentrio"),
        dependencies=_deps(loader),
    )

    assert captured["loader"] == (Path("/tmp/UrbanNav-Tokyo/Odaiba"), "septentrio")
