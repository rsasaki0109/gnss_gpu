"""Shared CLI helpers for GSDC2023 audit scripts."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


DEFAULT_AUDIT_OUTPUT_ROOT = Path(__file__).resolve().parent / "results"
TRIP_HELP = "trip in split/course/phone form"


def add_data_root_trip_args(
    parser: argparse.ArgumentParser,
    *,
    default_root: Path,
    trip_help: str = TRIP_HELP,
) -> None:
    add_data_root_arg(parser, default_root=default_root)
    add_trip_arg(parser, help_text=trip_help)


def add_data_root_arg(
    parser: argparse.ArgumentParser,
    *,
    default_root: Path,
) -> None:
    parser.add_argument("--data-root", type=Path, default=default_root)


def add_trip_arg(
    parser: argparse.ArgumentParser,
    *,
    help_text: str = TRIP_HELP,
) -> None:
    parser.add_argument("--trip", required=True, help=help_text)


def add_max_epochs_arg(
    parser: argparse.ArgumentParser,
    *,
    default: int = 0,
    help_text: str | None = None,
) -> None:
    parser.add_argument("--max-epochs", type=int, default=default, help=help_text)


def add_multi_gnss_arg(
    parser: argparse.ArgumentParser,
    *,
    default: bool = False,
    help_text: str | None = None,
) -> None:
    parser.add_argument("--multi-gnss", action=argparse.BooleanOptionalAction, default=default, help=help_text)


def add_output_dir_arg(
    parser: argparse.ArgumentParser,
    *,
    default: Path = DEFAULT_AUDIT_OUTPUT_ROOT,
    required: bool = False,
    help_text: str | None = None,
) -> None:
    kwargs: dict[str, Any] = {
        "type": Path,
        "help": help_text,
    }
    if required:
        kwargs["required"] = True
    else:
        kwargs["default"] = default
    parser.add_argument("--output-dir", **kwargs)


def resolve_trip_dir(args: Any) -> Path:
    return (Path(args.data_root) / str(args.trip)).resolve()


def resolved_output_root(args: Any) -> Path:
    return Path(args.output_dir).resolve()


def nonnegative_max_epochs(args: Any) -> int:
    return max(int(args.max_epochs), 0)


__all__ = [
    "DEFAULT_AUDIT_OUTPUT_ROOT",
    "TRIP_HELP",
    "add_data_root_arg",
    "add_data_root_trip_args",
    "add_max_epochs_arg",
    "add_multi_gnss_arg",
    "add_output_dir_arg",
    "add_trip_arg",
    "nonnegative_max_epochs",
    "resolve_trip_dir",
    "resolved_output_root",
]
