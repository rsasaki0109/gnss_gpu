"""End-to-end PF smoother run orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from gnss_gpu.pf_smoother_config import PfSmootherConfig
from gnss_gpu.pf_smoother_epoch_history import ForwardEpochHistory
from gnss_gpu.pf_smoother_forward_context import PfSmootherForwardPassContext
from gnss_gpu.pf_smoother_forward_loop import run_pf_smoother_forward_pass
from gnss_gpu.pf_smoother_forward_stats import ForwardRunStats
from gnss_gpu.pf_smoother_postrun import finalize_pf_smoother_postrun
from gnss_gpu.pf_smoother_run_context import (
    PfSmootherRunDependencies,
    build_pf_smoother_run_options,
)
from gnss_gpu.pf_smoother_run_result import build_initial_pf_smoother_run_result
from gnss_gpu.pf_smoother_runtime import (
    ForwardRunBuffers,
    RunDataset,
    build_observation_computers,
    initialize_imu_filter,
    initialize_particle_filter,
    resolve_run_dataset,
)
from gnss_gpu.pf_smoother_summary import print_pf_smoother_run_summary


def run_pf_smoother_evaluation(
    run_dir: Path,
    run_name: str,
    *,
    dataset: dict[str, object] | RunDataset | None,
    run_config: PfSmootherConfig,
    dependencies: PfSmootherRunDependencies,
) -> dict[str, object]:
    config_parts = run_config.parts()
    run_options = build_pf_smoother_run_options(run_config)
    ds = resolve_run_dataset(
        run_dir,
        run_config.rover_source,
        dataset,
        dependencies.load_dataset_func,
    )

    imu_filter = initialize_imu_filter(
        run_dir,
        run_config.predict_guide,
        ds,
        dependencies.ecef_to_lla_func,
    )
    stats = ForwardRunStats()
    observation_setup = build_observation_computers(
        run_dir,
        run_config.rover_source,
        config_parts.observations,
    )

    pf = initialize_particle_filter(
        ds.first_pos,
        ds.init_cb,
        config_parts.particle_filter,
        config_parts.observations.robust,
        config_parts.doppler,
        sigma_cb=dependencies.sigma_cb,
        seed=dependencies.seed,
    )

    buffers = ForwardRunBuffers()
    pr_history: dict[int, list[float]] = {}
    history = ForwardEpochHistory()
    elapsed_ms = run_pf_smoother_forward_pass(
        PfSmootherForwardPassContext(
            run_name=run_name,
            run_config=run_config,
            config_parts=config_parts,
            run_options=run_options,
            dependencies=dependencies,
            dataset=ds,
            imu_filter=imu_filter,
            pf=pf,
            buffers=buffers,
            stats=stats,
            history=history,
            observation_setup=observation_setup,
            pr_history=pr_history,
        )
    )

    result_context: dict[str, Any] = run_config.to_kwargs()
    result_context.update(
        {
            "run_name": run_name,
            "elapsed_ms": elapsed_ms,
            "fgo_motion_source": run_options.fgo_motion_source,
        }
    )
    result_context.update(stats.as_result_context())
    result = build_initial_pf_smoother_run_result(result_context)
    print_pf_smoother_run_summary(result)

    return finalize_pf_smoother_postrun(
        result,
        pf,
        buffers,
        position_update_sigma=run_config.position_update_sigma,
        use_smoother=run_config.use_smoother,
        smoother_config=config_parts.smoother,
        local_fgo_config=config_parts.local_fgo,
        fgo_motion_source=run_options.fgo_motion_source,
        collect_epoch_diagnostics=run_config.collect_epoch_diagnostics,
        compute_metrics_func=dependencies.compute_metrics_func,
        ecef_errors_func=dependencies.ecef_errors_func,
    ).result
