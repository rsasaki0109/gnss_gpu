import pytest

from gnss_gpu.pf_smoother_cli_config import (
    namespace_requests_epoch_diagnostics,
    namespace_to_run_config,
    namespace_to_run_kwargs,
)
from gnss_gpu.pf_smoother_cli_parser import build_pf_smoother_arg_parser


def _parse_args(*argv: str):
    parser = build_pf_smoother_arg_parser(default_sigma_pos=1.25)
    return parser.parse_args(["--data-root", "/tmp/UrbanNav-Tokyo", *argv])


def test_namespace_to_run_config_wraps_cli_kwargs():
    args = _parse_args(
        "--n-particles",
        "123",
        "--sigma-pos",
        "1.7",
        "--sigma-pr",
        "4.2",
        "--predict-guide",
        "imu",
        "--urban-rover",
        "septentrio",
        "--max-epochs",
        "9",
        "--skip-valid-epochs",
        "2",
        "--smoother-tail-guard-min-shift-m",
        "5.5",
        "--smoother-tail-guard-expand-epochs",
        "3",
        "--smoother-tail-guard-expand-dd-pseudorange-max-pairs",
        "0",
    )

    kwargs = namespace_to_run_kwargs(
        args,
        position_update_sigma=None,
        use_smoother=True,
    )
    config = namespace_to_run_config(
        args,
        position_update_sigma=None,
        use_smoother=True,
    )

    assert kwargs["n_particles"] == 123
    assert kwargs["sigma_pos"] == 1.7
    assert kwargs["sigma_pr"] == 4.2
    assert kwargs["position_update_sigma"] is None
    assert kwargs["predict_guide"] == "imu"
    assert kwargs["use_smoother"] is True
    assert kwargs["rover_source"] == "septentrio"
    assert kwargs["max_epochs"] == 9
    assert kwargs["skip_valid_epochs"] == 2
    assert kwargs["collect_epoch_diagnostics"] is True
    assert kwargs["smoother_tail_guard_expand_epochs"] == 3
    assert kwargs["smoother_tail_guard_expand_dd_pseudorange_max_pairs"] == 0
    assert config.to_kwargs() == kwargs


@pytest.mark.parametrize(
    "flags",
    [
        ("--epoch-diagnostics-out", "diag.csv"),
        ("--epoch-diagnostics-top-k", "2"),
        ("--smoother-widelane-forward-guard",),
        ("--smoother-tail-guard-expand-epochs", "2"),
        ("--fgo-local-window", "auto"),
    ],
)
def test_namespace_requests_epoch_diagnostics_for_diagnostic_cli_modes(flags):
    assert namespace_requests_epoch_diagnostics(_parse_args(*flags)) is True


def test_namespace_requests_epoch_diagnostics_false_for_default_args():
    assert namespace_requests_epoch_diagnostics(_parse_args()) is False
    assert namespace_to_run_kwargs(
        _parse_args(),
        position_update_sigma=3.0,
        use_smoother=False,
    )["collect_epoch_diagnostics"] is False
