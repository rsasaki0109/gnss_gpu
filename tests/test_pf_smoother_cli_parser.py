from pathlib import Path

from gnss_gpu.pf_smoother_cli_parser import build_pf_smoother_arg_parser


def test_build_pf_smoother_arg_parser_uses_injected_sigma_default():
    parser = build_pf_smoother_arg_parser(default_sigma_pos=1.7)

    args = parser.parse_args(["--data-root", "/tmp/UrbanNav-Tokyo"])

    assert args.data_root == Path("/tmp/UrbanNav-Tokyo")
    assert args.sigma_pos == 1.7
    assert args.runs == "Odaiba"


def test_build_pf_smoother_arg_parser_supports_boolean_optional_doppler_flag():
    parser = build_pf_smoother_arg_parser(default_sigma_pos=1.7)

    enabled = parser.parse_args(
        ["--data-root", "/tmp/UrbanNav-Tokyo", "--doppler-per-particle"]
    )
    disabled = parser.parse_args(
        ["--data-root", "/tmp/UrbanNav-Tokyo", "--no-doppler-per-particle"]
    )

    assert enabled.doppler_per_particle is True
    assert disabled.doppler_per_particle is False


def test_build_pf_smoother_arg_parser_exposes_recent_fgo_flags():
    parser = build_pf_smoother_arg_parser(default_sigma_pos=1.7)

    args = parser.parse_args(
        [
            "--data-root",
            "/tmp/UrbanNav-Tokyo",
            "--fgo-local-window",
            "auto",
            "--fgo-local-two-step",
            "--fgo-local-stage1-pr-sigma-m",
            "8.0",
        ]
    )

    assert args.fgo_local_window == "auto"
    assert args.fgo_local_two_step is True
    assert args.fgo_local_stage1_pr_sigma_m == 8.0
