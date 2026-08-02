import json
from pathlib import Path

from experiments.run_ppc_basin_fgo_six_route import (
    _solver_command,
    _write_pre_run_manifest,
)


def test_tracker_policy_is_explicit_in_runner_source() -> None:
    source = Path("experiments/run_ppc_basin_fgo_six_route.py").read_text(
        encoding="utf-8"
    )
    assert '"--fix-min-streak",' in source
    assert "str(args.fix_min_streak)" in source
    assert '"--validation-gap-tolerance",' in source
    assert "str(args.validation_gap_tolerance)" in source
    assert '"--disjoint-holdout-consensus"' in source
    assert "str(args.disjoint_holdout_margin)" in source
    assert '"--disjoint-holdout-min-carrier-fraction"' in source
    assert '"--causal-imu-motion-consensus"' in source
    assert "str(args.causal_imu_motion_gate)" in source
    assert '"--causal-imu-motion-min-carrier-fraction"' in source


def test_solver_command_excludes_reference_and_emits_basin_stream(
    tmp_path: Path,
) -> None:
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


def test_solver_command_records_candidate_supply_policy(tmp_path: Path) -> None:
    route = tmp_path / "nagoya" / "run3"
    command, _ = _solver_command(
        tmp_path / "gnss_solve",
        route,
        tmp_path / "out",
        "candidate",
        300,
        16,
        candidate_groups=4,
        fallback_consensus_groups=2,
        fallback_consensus_separation_m=0.1,
        fallback_max_seed_separation_m=0.3,
        constellation_par=True,
        interleave_constellation_par=True,
        quality_ranked_par=True,
    )
    joined = " ".join(command)
    assert "--multisd-fgo-shadow-candidate-groups 4" in joined
    assert "--multisd-fgo-shadow-fallback-consensus-groups 2" in joined
    assert "--multisd-fgo-shadow-fallback-consensus-separation 0.1" in joined
    assert "--multisd-fgo-shadow-fallback-max-seed-separation 0.3" in joined
    assert "--multisd-fgo-shadow-constellation-par" in command
    assert "--multisd-fgo-shadow-interleave-constellation-par" in command
    assert "--multisd-fgo-shadow-quality-ranked-par" in command


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
