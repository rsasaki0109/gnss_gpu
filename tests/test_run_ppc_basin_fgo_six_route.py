import json
from pathlib import Path

from experiments.run_ppc_basin_fgo_six_route import (
    _solver_command,
    _write_pre_run_manifest,
)


def test_solver_command_excludes_reference_and_emits_basin_stream(tmp_path: Path) -> None:
    route = tmp_path / "tokyo" / "run1"
    command, artifacts = _solver_command(
        tmp_path / "gnss_solve", route, tmp_path / "out", "route", 300
    )
    flattened = " ".join(command).lower()
    assert "reference.csv" not in flattened
    assert "--multisd-fgo-basin-jsonl" in command
    assert command[command.index("--multisd-fgo-shadow-top-k") + 1] == "4"
    assert command[command.index("--max-epochs") + 1] == "300"
    assert artifacts["basins"].name == "route.basins.jsonl"


def test_solver_command_adds_audited_native_imu_contract(tmp_path: Path) -> None:
    for city, expected_lever in (
        ("tokyo", ["0.31", "0.0", "0.55"]),
        ("nagoya", ["0.593", "0.67", "1.216"]),
    ):
        route = tmp_path / city / "run1"
        command, _ = _solver_command(
            tmp_path / "gnss_solve",
            route,
            tmp_path / "out",
            "route",
            30,
            native_imu=True,
        )
        imu_index = command.index("--multisd-fgo-imu")
        assert command[imu_index + 1] == str(route / "imu.csv")
        lever_index = command.index("--multisd-fgo-imu-lever-arm-flu")
        assert command[lever_index + 1 : lever_index + 4] == expected_lever
        assert "reference.csv" not in " ".join(command).lower()


def test_pre_run_manifest_freezes_truth_free_command_and_inputs(tmp_path: Path) -> None:
    binary = tmp_path / "gnss_solve"
    binary.write_bytes(b"binary")
    route = tmp_path / "tokyo" / "run1"
    route.mkdir(parents=True)
    for name in ("rover.obs", "base.obs", "base.nav", "imu.csv"):
        (route / name).write_text(name, encoding="utf-8")
    command, _ = _solver_command(
        binary, route, tmp_path / "out", "route", 300, native_imu=True
    )
    manifest = tmp_path / "out" / "route.run_manifest.json"

    _write_pre_run_manifest(
        manifest,
        binary=binary,
        route_dir=route,
        command=command,
        native_imu=True,
    )

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["production_input_truth"] is False
    assert payload["reference_in_command"] is False
    assert payload["command"] == command
    assert set(payload["inputs"]) == {"rover.obs", "base.obs", "base.nav", "imu.csv"}
