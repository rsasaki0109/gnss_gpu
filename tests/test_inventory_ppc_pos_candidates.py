from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "experiments/inventory_ppc_pos_candidates.py"
SPEC = importlib.util.spec_from_file_location("inventory_ppc_pos_candidates", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _dataset(root: Path) -> None:
    starts = {"tokyo": 100.0, "nagoya": 500.0}
    for city, run, _ in MODULE.ROUTES:
        directory = root / city / run
        directory.mkdir(parents=True)
        offset = starts[city] + int(run[-1]) * 10.0
        directory.joinpath("reference.csv").write_text(
            "tow,x,y,z\n"
            f"{offset:.1f},0,0,0\n"
            f"{offset + 0.2:.1f},1,0,0\n",
            encoding="utf-8",
        )


def test_inventory_matches_route_and_sorts_score(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    candidates = tmp_path / "candidates"
    candidates.mkdir()
    _dataset(dataset)
    candidates.joinpath("good.pos").write_text(
        "% pos\n0 120.0 0 0 0 0 0 0 3\n0 120.2 1 0 0 0 0 0 3\n",
        encoding="utf-8",
    )
    candidates.joinpath("bad.pos").write_text(
        "% pos\n0 120.0 0 0 0 0 0 0 3\n0 120.2 3 0 0 0 0 0 3\n",
        encoding="utf-8",
    )
    candidates.joinpath("reference_oracle.pos").write_text(
        "% forbidden truth copy\n0 120.0 0 0 0 0 0 0 3\n0 120.2 1 0 0 0 0 0 3\n",
        encoding="utf-8",
    )

    result = MODULE.inventory([candidates], dataset, minimum_coverage=1.0)

    rows = result["top_by_route"]["tokyo_run2"]
    assert len(rows) == 2
    assert rows[0]["path"].endswith("good.pos")
    assert rows[0]["ppc_score_pct"] == 100.0
    assert rows[1]["ppc_score_pct"] == 0.0
    assert result["files_discovered"] == 3
    assert result["files_scanned"] == 2
    assert result["truth_derived_files_excluded"] == 1
    assert result["truth_contract"]["runtime_selector_input_permitted"] is False
