from __future__ import annotations

from pathlib import Path
import hashlib
import json

import pytest

from experiments.compose_ppc_causal_float_selector import (
    _validate_inputs,
    compose_causal_float_selector,
)


def _write_safe(path: Path) -> None:
    path.write_text(
        "epoch_index,tow,shadow_fixed,status,x,y,z,source\n"
        "0,10.0,0,3,10,0,0,baseline_float\n"
        "1,10.2,1,4,20,0,0,imu_pf_fgo_fixed\n"
        "2,10.4,0,3,30,0,0,baseline_float\n"
        "3,10.6,0,3,40,0,0,baseline_float\n",
        encoding="utf-8",
    )


def _write_candidate(path: Path) -> None:
    path.write_text(
        "% test\n"
        "1 10.0 11 0 0 0 0 0 4\n"
        "1 10.2 21 0 0 0 0 0 4\n"
        "1 10.4 31 0 0 0 0 0 3\n"
        "1 10.6 41 0 0 0 0 0 3\n",
        encoding="utf-8",
    )


def test_selector_preserves_safe_fix_and_never_inherits_candidate_fix(
    tmp_path: Path,
) -> None:
    safe = tmp_path / "safe.csv"
    candidate = tmp_path / "candidate.pos"
    _write_safe(safe)
    _write_candidate(candidate)

    rows = compose_causal_float_selector(
        safe, candidate, health_window_epochs=2, health_fixed_fraction=0.5
    )

    assert [row["x"] for row in rows] == [11.0, 20.0, 31.0, 40.0]
    assert [row["status"] for row in rows] == [3, 4, 3, 3]
    assert [row["source"] for row in rows] == [
        "float_candidate_fixed_observation",
        "safe_fixed",
        "float_candidate_healthy",
        "safe_primary_float",
    ]
    assert sum(int(row["shadow_fixed"]) for row in rows) == 1


def test_selector_clears_health_after_gap(tmp_path: Path) -> None:
    safe = tmp_path / "safe.csv"
    safe.write_text(
        "tow,status,x,y,z\n10.0,3,10,0,0\n10.2,3,20,0,0\n20.0,3,30,0,0\n",
        encoding="utf-8",
    )
    candidate = tmp_path / "candidate.pos"
    candidate.write_text(
        "% test\n1 10.0 11 0 0 0 0 0 4\n1 10.2 21 0 0 0 0 0 3\n1 20.0 31 0 0 0 0 0 3\n",
        encoding="utf-8",
    )

    rows = compose_causal_float_selector(
        safe,
        candidate,
        health_window_epochs=2,
        health_fixed_fraction=0.5,
        maximum_gap_s=1.0,
    )

    assert [row["x"] for row in rows] == [11.0, 21.0, 30.0]
    assert rows[-1]["candidate_health_ready"] == 0


def test_selector_rejects_nonfinite_selected_candidate(tmp_path: Path) -> None:
    safe = tmp_path / "safe.csv"
    safe.write_text("tow,status,x,y,z\n10.0,3,10,0,0\n", encoding="utf-8")
    candidate = tmp_path / "candidate.pos"
    candidate.write_text("% test\n1 10.0 nan 0 0 0 0 0 4\n", encoding="utf-8")

    with pytest.raises(ValueError, match="non-finite selected"):
        compose_causal_float_selector(safe, candidate)


def test_selector_rejects_duplicate_safe_epoch(tmp_path: Path) -> None:
    safe = tmp_path / "safe.csv"
    safe.write_text(
        "tow,status,x,y,z\n10.0,3,10,0,0\n10.0,3,11,0,0\n",
        encoding="utf-8",
    )
    candidate = tmp_path / "candidate.pos"
    _write_candidate(candidate)

    with pytest.raises(ValueError, match="duplicate safe-output"):
        compose_causal_float_selector(safe, candidate)


def test_selector_uses_candidate_only_epoch_as_float(tmp_path: Path) -> None:
    safe = tmp_path / "safe.csv"
    safe.write_text("tow,status,x,y,z\n10.0,3,10,0,0\n", encoding="utf-8")
    candidate = tmp_path / "candidate.pos"
    candidate.write_text(
        "% test\n1 10.0 11 0 0 0 0 0 3\n1 10.2 12 0 0 0 0 0 3\n",
        encoding="utf-8",
    )

    rows = compose_causal_float_selector(safe, candidate)
    assert rows[-1]["source"] == "float_candidate_only"
    assert rows[-1]["status"] == 3


@pytest.mark.parametrize(
    ("window", "fraction", "gap"),
    [(0, 0.5, 1.0), (2, -0.1, 1.0), (2, 1.1, 1.0), (2, 0.5, 0.0)],
)
def test_selector_rejects_invalid_policy(
    tmp_path: Path, window: int, fraction: float, gap: float
) -> None:
    safe = tmp_path / "safe.csv"
    candidate = tmp_path / "candidate.pos"
    _write_safe(safe)
    _write_candidate(candidate)

    with pytest.raises(ValueError):
        compose_causal_float_selector(
            safe,
            candidate,
            health_window_epochs=window,
            health_fixed_fraction=fraction,
            maximum_gap_s=gap,
        )


def test_input_manifests_bind_truth_free_artifacts(tmp_path: Path) -> None:
    safe = tmp_path / "safe.csv"
    candidate = tmp_path / "candidate.pos"
    _write_safe(safe)
    _write_candidate(candidate)

    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    safe_summary = tmp_path / "safe.json"
    safe_summary.write_text(
        json.dumps(
            {
                "schema": "gnss_gpu_ppc_imu_safe_output_v1",
                "production_input_truth": False,
                "truth_usage": "none",
                "output_sha256": digest(safe),
            }
        ),
        encoding="utf-8",
    )
    candidate_manifest = tmp_path / "manifest.json"
    candidate_manifest.write_text(
        json.dumps(
            {
                "schema": "gnss_gpu_ppc_float_candidate_run_v1",
                "production_input_truth": False,
                "truth_usage": "none",
                "route": "tokyo_run1",
                "output_sha256": {"position": digest(candidate)},
            }
        ),
        encoding="utf-8",
    )

    _validate_inputs(safe, safe_summary, candidate, candidate_manifest)
    with pytest.raises(ValueError, match="candidate manifest"):
        _validate_inputs(
            safe,
            safe_summary,
            candidate,
            candidate_manifest,
            expected_route="nagoya_run1",
        )
    candidate.write_text(candidate.read_text(encoding="utf-8") + "% changed\n")
    with pytest.raises(ValueError, match="candidate manifest"):
        _validate_inputs(safe, safe_summary, candidate, candidate_manifest)
