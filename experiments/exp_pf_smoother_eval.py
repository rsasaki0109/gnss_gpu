#!/usr/bin/env python3
"""Evaluate forward-backward particle smoothing on the gnssplusplus UrbanNav PF stack.

Pipeline per epoch (forward): predict (SPP or TDCP guide) → ``correct_clock_bias`` →
``update`` → optional ``position_update`` → ``store_epoch`` when smoothing is enabled.

After the forward pass, ``smooth()`` runs a backward PF and averages with forward
estimates. Metrics are computed on epochs aligned with UrbanNav ground-truth time tags
(same convention as ``exp_position_update_eval.py``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
for _p in (
    _PROJECT_ROOT / "python",
    _PROJECT_ROOT / "third_party" / "gnssplusplus" / "python",
    _PROJECT_ROOT / "third_party" / "gnssplusplus" / "build" / "python",
):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from evaluate import compute_metrics, ecef_errors_2d_3d, ecef_to_lla
from exp_urbannav_baseline import load_or_generate_data
from exp_urbannav_pf3d import PF_SIGMA_CB, PF_SIGMA_POS
from gnss_gpu.pf_smoother_dataset import (
    load_pf_smoother_dataset as _load_pf_smoother_dataset,
)
from gnss_gpu.pf_smoother_config import (
    PfSmootherConfig,
    coerce_pf_smoother_config,
    validate_pf_smoother_config,
)
from gnss_gpu.pf_smoother_cli_parser import build_pf_smoother_arg_parser
from gnss_gpu.pf_smoother_cli_config import (
    namespace_requests_epoch_diagnostics as _namespace_requests_epoch_diagnostics,
    namespace_to_run_config as _namespace_to_run_config,
    namespace_to_run_kwargs as _namespace_to_run_kwargs,
)
from gnss_gpu.pf_smoother_cli_presets import (
    CLI_PRESETS as _CLI_PRESETS,
    expand_cli_preset_argv as _expand_cli_preset_argv,
    print_cli_presets as _print_cli_presets,
)
from gnss_gpu.pf_smoother_cli_runner import execute_pf_smoother_cli
from gnss_gpu.pf_smoother_run import (
    PfSmootherRunDependencies,
    run_pf_smoother_evaluation,
)
from gnss_gpu.pf_smoother_runtime import RunDataset

RESULTS_DIR = _SCRIPT_DIR / "results"


def load_pf_smoother_dataset(run_dir: Path, rover_source: str = "trimble") -> dict[str, object]:
    """Load RINEX / UrbanNav ground-truth once for repeated PF runs (sweeps).

    Returns a dict with keys: ``epochs``, ``spp_lookup``, ``gt``, ``our_times``,
    ``first_pos``, ``init_cb``.  If ``imu.csv`` exists in *run_dir*, also
    includes ``imu_data`` (raw dict from :func:`load_imu_csv`).
    """
    return _load_pf_smoother_dataset(
        run_dir,
        rover_source,
        urban_data_loader=load_or_generate_data,
    )


def run_pf_with_optional_smoother(
    run_dir: Path,
    run_name: str,
    *,
    dataset: dict[str, object] | RunDataset | None = None,
    config: PfSmootherConfig | None = None,
    **overrides: object,
) -> dict[str, object]:
    """Run one PF smoother evaluation.

    New call sites should pass ``config=PfSmootherConfig(...)``. ``overrides``
    keeps existing sweep scripts working while they migrate away from giant
    keyword argument lists.
    """

    run_config = coerce_pf_smoother_config(config, dict(overrides))
    validate_pf_smoother_config(run_config)
    return _run_pf_with_optional_smoother_impl(
        run_dir,
        run_name,
        dataset=dataset,
        run_config=run_config,
        **run_config.to_kwargs(),
    )


def _run_pf_with_optional_smoother_impl(
    run_dir: Path,
    run_name: str,
    *,
    dataset: dict[str, object] | RunDataset | None = None,
    run_config: PfSmootherConfig,
    **_legacy_kwargs: object,
) -> dict[str, object]:
    return run_pf_smoother_evaluation(
        run_dir,
        run_name,
        dataset=dataset,
        run_config=run_config,
        dependencies=PfSmootherRunDependencies(
            load_dataset_func=load_pf_smoother_dataset,
            ecef_to_lla_func=ecef_to_lla,
            compute_metrics_func=compute_metrics,
            ecef_errors_func=ecef_errors_2d_3d,
            sigma_cb=PF_SIGMA_CB,
            seed=run_config.seed,
        ),
    )


def build_arg_parser() -> argparse.ArgumentParser:
    return build_pf_smoother_arg_parser(PF_SIGMA_POS)


def main(argv: list[str] | None = None) -> int:
    return execute_pf_smoother_cli(
        argv,
        parser_factory=build_arg_parser,
        results_dir=RESULTS_DIR,
        run_func=run_pf_with_optional_smoother,
    )


if __name__ == "__main__":
    raise SystemExit(main())
