from __future__ import annotations

from pathlib import Path

import pytest

from experiments.run_ppc_float_candidates import (
    FROZEN_POLICY_ARGUMENTS,
    build_command,
    main,
)


def test_build_command_is_truth_free_and_route_frozen(tmp_path: Path) -> None:
    command, artifacts = build_command(
        tmp_path / "gnss_fuse",
        tmp_path / "dataset",
        tmp_path / "output",
        "nagoya_run2",
        max_epochs=300,
    )

    assert not any(argument.startswith("--reference") for argument in command)
    assert not any(argument.startswith("--truth") for argument in command)
    assert command[command.index("--lever-arm") + 1] == "0.593,0.670,1.216"
    assert command[-2:] == ["--max-epochs", "300"]
    assert list(FROZEN_POLICY_ARGUMENTS) == command[command.index("--preset") : -2]
    assert artifacts["position"].name == "float_candidate.pos"
    assert artifacts["pre_manifest"].name == "pre_run_manifest.json"
    assert artifacts["manifest"].parent.name == "nagoya_run2"


def test_build_command_omits_unbounded_max_epochs(tmp_path: Path) -> None:
    command, _ = build_command(
        tmp_path / "gnss_fuse",
        tmp_path / "dataset",
        tmp_path / "output",
        "tokyo_run1",
    )

    assert "--max-epochs" not in command
    assert command[command.index("--lever-arm") + 1] == "0.31,0,0.55"


def test_build_command_rejects_unknown_route(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsupported route"):
        build_command(
            tmp_path / "gnss_fuse",
            tmp_path / "dataset",
            tmp_path / "output",
            "tokyo_run4",
        )


def test_main_rejects_wrong_frozen_binary_hash(tmp_path: Path) -> None:
    binary = tmp_path / "gnss_fuse.exe"
    binary.write_bytes(b"binary")
    dataset = tmp_path / "dataset"
    dataset.mkdir()

    with pytest.raises(SystemExit):
        main(
            [
                "--binary",
                str(binary),
                "--dataset-root",
                str(dataset),
                "--output-root",
                str(tmp_path / "output"),
                "--expected-binary-sha256",
                "0" * 64,
            ]
        )
