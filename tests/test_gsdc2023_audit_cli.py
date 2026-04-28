from __future__ import annotations

import argparse
from pathlib import Path

from experiments.gsdc2023_audit_cli import (
    add_data_root_arg,
    add_data_root_trip_args,
    add_max_epochs_arg,
    add_multi_gnss_arg,
    add_output_dir_arg,
    add_trip_arg,
    nonnegative_max_epochs,
    resolve_trip_dir,
    resolved_output_root,
)


def test_audit_cli_helpers_add_common_trip_window_and_output_args(tmp_path: Path) -> None:
    parser = argparse.ArgumentParser()
    add_data_root_trip_args(parser, default_root=tmp_path / "root")
    add_max_epochs_arg(parser)
    add_multi_gnss_arg(parser, help_text="multi")
    add_output_dir_arg(parser, default=tmp_path / "out")

    args = parser.parse_args(
        [
            "--trip",
            "train/course/phone",
            "--max-epochs",
            "-5",
            "--no-multi-gnss",
        ],
    )

    assert resolve_trip_dir(args) == (tmp_path / "root" / "train/course/phone").resolve()
    assert resolved_output_root(args) == (tmp_path / "out").resolve()
    assert nonnegative_max_epochs(args) == 0
    assert args.multi_gnss is False


def test_audit_cli_helpers_accept_overrides(tmp_path: Path) -> None:
    parser = argparse.ArgumentParser()
    add_data_root_trip_args(parser, default_root=tmp_path / "root")
    add_max_epochs_arg(parser, default=200)
    add_multi_gnss_arg(parser, default=False)
    add_output_dir_arg(parser)

    args = parser.parse_args(
        [
            "--data-root",
            str(tmp_path / "custom_root"),
            "--trip",
            "test/course/phone",
            "--max-epochs",
            "10",
            "--multi-gnss",
            "--output-dir",
            str(tmp_path / "custom_out"),
        ],
    )

    assert resolve_trip_dir(args) == (tmp_path / "custom_root" / "test/course/phone").resolve()
    assert resolved_output_root(args) == (tmp_path / "custom_out").resolve()
    assert nonnegative_max_epochs(args) == 10
    assert args.multi_gnss is True


def test_audit_cli_helpers_allow_data_root_and_trip_args_separately(tmp_path: Path) -> None:
    parser = argparse.ArgumentParser()
    add_data_root_arg(parser, default_root=tmp_path / "root")
    add_trip_arg(parser, help_text="custom trip help")

    args = parser.parse_args(["--trip", "train/course/phone"])

    assert resolve_trip_dir(args) == (tmp_path / "root" / "train/course/phone").resolve()


def test_audit_cli_helpers_support_required_output_dir(tmp_path: Path) -> None:
    parser = argparse.ArgumentParser()
    add_output_dir_arg(parser, required=True, help_text="required output")

    args = parser.parse_args(["--output-dir", str(tmp_path / "out")])

    assert resolved_output_root(args) == (tmp_path / "out").resolve()
