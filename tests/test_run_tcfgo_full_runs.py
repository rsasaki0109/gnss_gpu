from __future__ import annotations

import csv
import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "experiments/run_tcfgo_full_runs.py"
SPEC = importlib.util.spec_from_file_location("run_tcfgo_full_runs", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_reference_epoch_count_excludes_header(tmp_path: Path):
    path = tmp_path / "reference.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["tow", "x"])
        writer.writerow([1.0, 2.0])
        writer.writerow([1.2, 3.0])

    assert MODULE._reference_epoch_count(path) == 2


def test_full_runner_declares_all_official_runs_and_variants():
    assert MODULE.RUNS == (
        ("tokyo", "run1"),
        ("tokyo", "run2"),
        ("tokyo", "run3"),
        ("nagoya", "run1"),
        ("nagoya", "run2"),
        ("nagoya", "run3"),
    )
    assert set(MODULE.VARIANTS) == {"baseline", "wcp", "switch", "wcp_switch"}


def test_phase_init_fix_count_has_one_data_feasibility_exception():
    assert MODULE._phase_init_static_fixes("tokyo", "run1") == 5
    assert MODULE._phase_init_static_fixes("nagoya", "run2") == 5
    assert MODULE._phase_init_static_fixes("nagoya", "run3") == 4


def test_resummarize_never_launches_a_replay():
    assert not MODULE._needs_run(
        force=False,
        refresh_switch=False,
        resummarize=True,
        variant="baseline",
        summary_exists=False,
    )
    assert not MODULE._needs_run(
        force=True,
        refresh_switch=True,
        resummarize=True,
        variant="switch",
        summary_exists=False,
    )


def test_missing_summary_launches_normal_replay():
    assert MODULE._needs_run(
        force=False,
        refresh_switch=False,
        resummarize=False,
        variant="baseline",
        summary_exists=False,
    )
