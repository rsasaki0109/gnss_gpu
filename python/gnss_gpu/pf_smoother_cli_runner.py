"""Top-level CLI execution loop for PF smoother evaluations."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from gnss_gpu.epoch_diagnostics import (
    _diagnostics_output_path,
    _print_top_epoch_diagnostics,
    _write_epoch_diagnostics,
)
from gnss_gpu.pf_smoother_cli_config import namespace_to_run_config
from gnss_gpu.pf_smoother_cli_output import (
    pf_smoother_variant_metrics,
    print_pf_smoother_run_header,
    print_pf_smoother_variant_metrics,
    print_pf_smoother_variant_start,
    select_pf_smoother_variants,
    write_pf_smoother_result_csv,
)
from gnss_gpu.pf_smoother_cli_presets import (
    expand_cli_preset_argv,
    print_cli_presets,
)
from gnss_gpu.pf_smoother_results import build_pf_smoother_result_row

RunPfSmootherFunc = Callable[..., dict[str, Any]]


def execute_pf_smoother_cli(
    argv: list[str] | None,
    *,
    parser_factory: Callable[[], argparse.ArgumentParser],
    results_dir: Path,
    run_func: RunPfSmootherFunc,
) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if "--list-presets" in raw_argv:
        print_cli_presets()
        return 0

    parser = parser_factory()
    args = parser.parse_args(expand_cli_preset_argv(raw_argv))
    return run_pf_smoother_cli_args(args, results_dir=results_dir, run_func=run_func)


def run_pf_smoother_cli_args(
    args: argparse.Namespace,
    *,
    results_dir: Path,
    run_func: RunPfSmootherFunc,
) -> int:
    pos_sigma = args.position_update_sigma
    if pos_sigma < 0:
        pos_sigma = None

    results_dir.mkdir(parents=True, exist_ok=True)
    runs = [r.strip() for r in args.runs.split(",") if r.strip()]
    rows: list[dict[str, object]] = []

    for run_name in runs:
        run_dir = args.data_root / run_name
        print_pf_smoother_run_header(run_name)
        variants = select_pf_smoother_variants(
            compare_both=args.compare_both,
            use_smoother=args.smoother,
        )

        for label, use_sm in variants:
            print_pf_smoother_variant_start(
                label=label,
                predict_guide=args.predict_guide,
                position_update_sigma=pos_sigma,
                sigma_pos_tdcp=args.sigma_pos_tdcp,
                use_smoother=use_sm,
            )
            run_config = namespace_to_run_config(
                args,
                position_update_sigma=pos_sigma,
                use_smoother=use_sm,
            )
            out = run_func(
                run_dir,
                run_name,
                config=run_config,
            )
            fm, sm, ep_n, ms_ep = pf_smoother_variant_metrics(out)
            print_pf_smoother_variant_metrics(
                out,
                fm,
                sm,
                n_epochs=ep_n,
                ms_per_epoch=ms_ep,
            )
            epoch_diagnostics = out.get("epoch_diagnostics")
            if epoch_diagnostics:
                if args.epoch_diagnostics_top_k > 0:
                    _print_top_epoch_diagnostics(epoch_diagnostics, args.epoch_diagnostics_top_k)
                if args.epoch_diagnostics_out is not None:
                    diag_path = _diagnostics_output_path(
                        args.epoch_diagnostics_out,
                        run_name=run_name,
                        label=label,
                        multiple_outputs=(len(runs) * len(variants) > 1),
                    )
                    _write_epoch_diagnostics(epoch_diagnostics, diag_path)
                    print(f"       epoch diagnostics: {diag_path}")
            rows.append(
                build_pf_smoother_result_row(
                    args=args,
                    out=out,
                    run_name=run_name,
                    label=label,
                    use_smoother=use_sm,
                    position_update_sigma=pos_sigma,
                    forward_metrics=fm,
                    smoothed_metrics=sm,
                    n_epochs=ep_n,
                    ms_per_epoch=ms_ep,
                )
            )

    out_csv = results_dir / "pf_smoother_eval.csv"
    write_pf_smoother_result_csv(rows, out_csv)
    return 0
