from pathlib import Path

from experiments.run_ppc_basin_fgo_six_route import _solver_command


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
