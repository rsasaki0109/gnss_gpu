from __future__ import annotations

import csv
import hashlib
import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "experiments/run_tcfgo_blocked_spans.py"
SPEC = importlib.util.spec_from_file_location("run_tcfgo_blocked_spans", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_summarize_tcfgo_blocked_telemetry(tmp_path: Path):
    path = tmp_path / "telemetry.csv"
    fields = [
        "epoch",
        "pos_err_m",
        "n_wcp_factors",
        "n_switchable_pseudorange",
        "n_switched_pseudorange",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            [
                {
                    "epoch": "9",
                    "pos_err_m": "99.0",
                    "n_wcp_factors": "100",
                    "n_switchable_pseudorange": "100",
                    "n_switched_pseudorange": "100",
                },
                {
                    "epoch": "10",
                    "pos_err_m": "0.25",
                    "n_wcp_factors": "2",
                    "n_switchable_pseudorange": "4",
                    "n_switched_pseudorange": "1",
                },
                {
                    "epoch": "11",
                    "pos_err_m": "2.0",
                    "n_wcp_factors": "3",
                    "n_switchable_pseudorange": "5",
                    "n_switched_pseudorange": "0",
                },
            ]
        )

    result = MODULE._summarize(
        path,
        start_epoch=10,
        end_epoch=14,
        requested_epochs=4,
        runtime_s=0.2,
    )

    assert result["output_epochs"] == 2
    assert result["evaluated_epochs"] == 2
    assert result["coverage"] == pytest.approx(0.5)
    assert result["pass_0_5m"] == pytest.approx(0.5)
    assert result["pass_1m"] == pytest.approx(0.5)
    assert result["pass_3m"] == pytest.approx(1.0)
    assert result["n_wcp_factors"] == 5
    assert result["n_switchable_pseudorange"] == 9
    assert result["n_switched_pseudorange"] == 1
    assert result["runtime_ms_per_output_epoch"] == pytest.approx(100.0)
    assert result["warmup_epochs"] == 10
    assert result["run_status"] == "ok"
    assert result["failure_reason"] == ""


def test_causal_initialization_failure_is_an_explicit_abstention(tmp_path: Path):
    message = (
        "ValueError: insufficient static RTK FIX epochs for phase-1 init: 4\n"
    )
    assert MODULE._classify_expected_abstention(message) == (
        "insufficient_causal_static_fix_history"
    )
    assert MODULE._classify_expected_abstention("unrelated failure") is None

    path = tmp_path / "empty.csv"
    MODULE._write_empty_telemetry(path)
    result = MODULE._summarize(
        path,
        start_epoch=235,
        end_epoch=335,
        requested_epochs=100,
        runtime_s=0.1,
    )
    assert result["coverage"] == 0.0
    assert result["output_epochs"] == 0


def test_position_sha256_is_content_exact_and_missing_safe(tmp_path: Path):
    path = tmp_path / "solution.pos"
    payload = b"epoch position\n"
    path.write_bytes(payload)

    assert MODULE._sha256_or_empty(path) == hashlib.sha256(payload).hexdigest()
    assert MODULE._sha256_or_empty(tmp_path / "missing.pos") == ""
