from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "experiments/run_recurrence_full_runs.py"
SPEC = importlib.util.spec_from_file_location("run_recurrence_full_runs", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_full_runner_declares_all_official_scopes():
    assert [f"{city}_{run}_full" for city, run, _ in MODULE.OFFICIAL_RUNS] == [
        "tokyo_run1_full",
        "tokyo_run2_full",
        "tokyo_run3_full",
        "nagoya_run1_full",
        "nagoya_run2_full",
        "nagoya_run3_full",
    ]


def test_summarize_recurrence_chunks(tmp_path: Path):
    paths = []
    for index, rows in enumerate(
        (
            [(1.0, 0.5, False), (2.0, 2.5, True)],
            [(4.0, 4.0, True), (float("nan"), float("nan"), False)],
        )
    ):
        path = tmp_path / f"chunk{index}.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["baseline_error_m", "selected_error_m", "recurrence_abstained"],
            )
            writer.writeheader()
            for baseline, selected, abstained in rows:
                writer.writerow(
                    {
                        "baseline_error_m": baseline,
                        "selected_error_m": selected,
                        "recurrence_abstained": abstained,
                    }
                )
        paths.append(path)

    result = MODULE._summarize_epoch_files(paths, requested_epochs=4, runtime_s=0.3)

    assert result["evaluated_epochs"] == 3
    assert result["coverage"] == pytest.approx(0.75)
    assert result["recurrence_abstained_epochs"] == 2
    assert result["recurrence_acceptance_rate"] == pytest.approx(1.0 / 3.0)
    assert result["selected_p50_m"] == pytest.approx(2.5)
    assert result["selected_p99_m"] == pytest.approx(3.97)
    assert result["improved_epochs"] == 1
    assert result["worsened_epochs"] == 1
    assert result["runtime_ms_per_evaluated_epoch"] == pytest.approx(100.0)


def test_raw_counterfactual_disables_all_safety_gates():
    assert MODULE._recurrence_mode_flags(False) == [
        "--recurrence-max-source-error-m",
        "20.0",
        "--recurrence-min-selected-probability",
        "0.05",
    ]
    assert MODULE._recurrence_mode_flags(True) == [
        "--recurrence-max-source-error-m",
        "0",
        "--recurrence-min-selected-probability",
        "0",
        "--recurrence-allow-boundary",
    ]


def test_chunk_resume_requires_matching_scope_and_safe_policy(tmp_path: Path):
    summary = tmp_path / "chunk.json"
    epochs = tmp_path / "chunk.csv"
    epochs.write_text("epoch\n0\n", encoding="utf-8")
    summary.write_text(
        json.dumps(
            {
                "start_epoch": 0,
                "requested_epochs": 500,
                "evaluated_epochs": 1,
                "skipped_epochs": 499,
                "recurrence_min_selected_probability": 0.05,
            }
        ),
        encoding="utf-8",
    )
    assert MODULE._chunk_is_complete(
        summary, epochs, start=0, count=500, raw=False
    )
    assert not MODULE._chunk_is_complete(
        summary, epochs, start=0, count=2000, raw=False
    )
    assert not MODULE._chunk_is_complete(
        summary, epochs, start=500, count=500, raw=False
    )

    payload = json.loads(summary.read_text(encoding="utf-8"))
    del payload["recurrence_min_selected_probability"]
    summary.write_text(json.dumps(payload), encoding="utf-8")
    assert not MODULE._chunk_is_complete(
        summary, epochs, start=0, count=500, raw=False
    )


def test_chunk_resume_requires_complete_raw_counterfactual_policy(tmp_path: Path):
    summary = tmp_path / "chunk.json"
    epochs = tmp_path / "chunk.csv"
    epochs.write_text("epoch\n0\n", encoding="utf-8")
    payload = {
        "start_epoch": 0,
        "requested_epochs": 1,
        "evaluated_epochs": 1,
        "skipped_epochs": 0,
        "recurrence_min_selected_probability": 0.0,
        "recurrence_max_source_error_m": 0.0,
        "recurrence_allow_boundary": True,
    }
    summary.write_text(json.dumps(payload), encoding="utf-8")
    assert MODULE._chunk_is_complete(
        summary, epochs, start=0, count=1, raw=True
    )

    payload["recurrence_allow_boundary"] = False
    summary.write_text(json.dumps(payload), encoding="utf-8")
    assert not MODULE._chunk_is_complete(
        summary, epochs, start=0, count=1, raw=True
    )


def test_terminal_chunk_accepts_absent_trailing_source_epochs(tmp_path: Path):
    summary = tmp_path / "chunk.json"
    epochs = tmp_path / "chunk.csv"
    epochs.write_text("epoch\n8000\n8001\n", encoding="utf-8")
    summary.write_text(
        json.dumps(
            {
                "start_epoch": 8000,
                "requested_epochs": 10,
                "evaluated_epochs": 2,
                "skipped_epochs": 3,
                "recurrence_min_selected_probability": 0.0,
                "recurrence_max_source_error_m": 0.0,
                "recurrence_allow_boundary": True,
            }
        ),
        encoding="utf-8",
    )

    assert MODULE._chunk_is_complete(
        summary, epochs, start=8000, count=10, raw=True
    )
    epochs.write_text("epoch\n8000\n", encoding="utf-8")
    assert not MODULE._chunk_is_complete(
        summary, epochs, start=8000, count=10, raw=True
    )
