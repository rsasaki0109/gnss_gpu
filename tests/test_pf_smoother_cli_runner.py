import csv
from pathlib import Path

from gnss_gpu.pf_smoother_cli_parser import build_pf_smoother_arg_parser
from gnss_gpu.pf_smoother_cli_runner import (
    execute_pf_smoother_cli,
    run_pf_smoother_cli_args,
)


def _parse_args(*argv: str):
    parser = build_pf_smoother_arg_parser(default_sigma_pos=1.0)
    return parser.parse_args(["--data-root", "/tmp/UrbanNav-Tokyo", *argv])


def _fake_run_output(*, use_smoother: bool) -> dict[str, object]:
    smoothed_metrics = (
        {"n_epochs": 2, "p50": 0.8, "p95": 1.4, "rms_2d": 0.9}
        if use_smoother
        else None
    )
    return {
        "forward_metrics": {"n_epochs": 2, "p50": 1.0, "p95": 1.8, "rms_2d": 1.2},
        "smoothed_metrics": smoothed_metrics,
        "elapsed_ms": 10.0,
    }


def test_run_pf_smoother_cli_args_runs_variants_and_writes_result_csv(tmp_path, capsys):
    args = _parse_args(
        "--runs",
        "Odaiba,Shinjuku",
        "--compare-both",
        "--position-update-sigma",
        "-1",
    )
    calls = []

    def run_func(run_dir: Path, run_name: str, *, config):
        calls.append((run_dir, run_name, config.use_smoother, config.position_update_sigma))
        return _fake_run_output(use_smoother=config.use_smoother)

    assert run_pf_smoother_cli_args(args, results_dir=tmp_path, run_func=run_func) == 0

    assert calls == [
        (Path("/tmp/UrbanNav-Tokyo/Odaiba"), "Odaiba", False, None),
        (Path("/tmp/UrbanNav-Tokyo/Odaiba"), "Odaiba", True, None),
        (Path("/tmp/UrbanNav-Tokyo/Shinjuku"), "Shinjuku", False, None),
        (Path("/tmp/UrbanNav-Tokyo/Shinjuku"), "Shinjuku", True, None),
    ]

    rows = list(csv.DictReader((tmp_path / "pf_smoother_eval.csv").open()))
    assert [row["run"] for row in rows] == ["Odaiba", "Odaiba", "Shinjuku", "Shinjuku"]
    assert [row["variant"] for row in rows] == [
        "forward_only",
        "with_smoother",
        "forward_only",
        "with_smoother",
    ]
    assert {row["position_update_sigma"] for row in rows} == {"off"}
    assert capsys.readouterr().out


def test_execute_pf_smoother_cli_lists_presets_without_parsing_required_args(tmp_path, capsys):
    def parser_factory():
        raise AssertionError("parser should not be built for --list-presets")

    def run_func(*args, **kwargs):
        raise AssertionError("run should not start for --list-presets")

    assert execute_pf_smoother_cli(
        ["--list-presets"],
        parser_factory=parser_factory,
        results_dir=tmp_path,
        run_func=run_func,
    ) == 0
    assert "odaiba_reference" in capsys.readouterr().out
