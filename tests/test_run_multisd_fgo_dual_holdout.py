import csv
from pathlib import Path

import experiments.run_multisd_fgo_dual_holdout as runner


def _write_shadow(path: Path, position: float, runtime_ms: float) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["epoch_index", "tow", "shadow_fixed", "x", "y", "z", "runtime_ms"]
        )
        writer.writerow([0, 1.0, 1, position, 0.0, 0.0, runtime_ms])


def test_dual_runner_uses_isolated_holdout_partitions(monkeypatch, tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    route = data_root / "tokyo" / "run1"
    route.mkdir(parents=True)
    (route / "reference.csv").write_text(
        "GPS TOW (s),ECEF X (m),ECEF Y (m),ECEF Z (m)\n1.0,10,0,0\n",
        encoding="utf-8",
    )
    calls: list[int] = []

    def fake_run_one(
        binary: Path,
        data: Path,
        output: Path,
        city: str,
        run: str,
        policy: runner.Policy,
        max_epochs: int,
        cuda_mode: str,
        resume: bool,
        analyze_only: bool,
    ) -> tuple[Path, Path, list[str]]:
        calls.append(policy.holdout_satellites)
        pos = output / f"{policy.name}.pos"
        shadow = output / f"{policy.name}.shadow.csv"
        pos.write_text("% unused\n", encoding="utf-8")
        _write_shadow(shadow, 10.0, float(policy.holdout_satellites))
        return pos, shadow, [str(binary), f"--holdout={policy.holdout_satellites}"]

    monkeypatch.setattr(runner, "_run_one", fake_run_one)
    payload = runner.run_dual_holdout(
        tmp_path / "gnss_solve",
        data_root,
        tmp_path / "out",
        "tokyo",
        "run1",
        max_epochs=1,
    )

    assert sorted(calls) == [3, 4]
    assert payload["process_isolation"] is True
    assert payload["excluded_estimator_inputs"] == [
        "imu",
        "lidar",
        "camera",
        "reference",
    ]
    assert payload["audit"]["result"]["correct"] == 1
    assert payload["audit"]["result"]["false"] == 0
    assert payload["audit"]["runtime"]["parallel_lower_bound_max_ms"] == 4.0
