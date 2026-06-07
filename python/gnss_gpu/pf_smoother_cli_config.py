"""Translate PF smoother CLI namespaces into runtime configuration."""

from __future__ import annotations

import argparse
from dataclasses import fields

from gnss_gpu.pf_smoother_config import PfSmootherConfig

_CONFIG_ARG_ALIASES = {
    "rover_source": "urban_rover",
    "tdcp_tight_rms_max_m": "tdcp_tight_rms_max",
    "use_gmm": "gmm",
}

_RUNTIME_CONFIG_FIELDS = {
    "collect_epoch_diagnostics",
    "position_update_sigma",
    "resampling",
    "use_smoother",
}


def namespace_requests_epoch_diagnostics(args: argparse.Namespace) -> bool:
    return (
        args.epoch_diagnostics_out is not None
        or args.epoch_diagnostics_top_k > 0
        or args.smoother_tail_guard_ess_max_ratio is not None
        or args.smoother_tail_guard_dd_carrier_max_pairs is not None
        or args.smoother_tail_guard_dd_pseudorange_max_pairs is not None
        or args.smoother_tail_guard_min_shift_m is not None
        or args.smoother_tail_guard_expand_epochs is not None
        or args.smoother_tail_guard_expand_min_shift_m is not None
        or args.smoother_tail_guard_expand_dd_pseudorange_max_pairs is not None
        or args.smoother_widelane_forward_guard
        or str(args.fgo_local_window).strip().lower() == "auto"
    )


def namespace_to_run_kwargs(
    args: argparse.Namespace,
    *,
    position_update_sigma: float | None,
    use_smoother: bool,
) -> dict[str, object]:
    run_kwargs = {
        field.name: getattr(args, _CONFIG_ARG_ALIASES.get(field.name, field.name))
        for field in fields(PfSmootherConfig)
        if field.name not in _RUNTIME_CONFIG_FIELDS
    }
    run_kwargs.update(
        collect_epoch_diagnostics=namespace_requests_epoch_diagnostics(args),
        position_update_sigma=position_update_sigma,
        resampling="megopolis",
        use_smoother=use_smoother,
    )
    return run_kwargs


def namespace_to_run_config(
    args: argparse.Namespace,
    *,
    position_update_sigma: float | None,
    use_smoother: bool,
) -> PfSmootherConfig:
    return PfSmootherConfig(
        **namespace_to_run_kwargs(
            args,
            position_update_sigma=position_update_sigma,
            use_smoother=use_smoother,
        )
    )
