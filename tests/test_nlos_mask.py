"""Tests for PLATEAU NLOS mask loading and soft weight mapping."""

from __future__ import annotations

from pathlib import Path

import pytest

from gnss_gpu.nlos_mask import (
    NlosMaskTables,
    apply_mask_to_weights,
    epoch_prn_weights,
    load_nlos_mask_tables,
    load_nlos_prn_sets,
    nlos_weight_factor,
)


def _write_mask_csv(path: Path, rows: list[tuple[float, int, str, int]]) -> None:
    path.write_text(
        "tow,epoch_idx,prn,is_los\n"
        + "\n".join(f"{tow},{epoch},{prn},{is_los}" for tow, epoch, prn, is_los in rows)
        + "\n",
        encoding="utf-8",
    )


def test_load_nlos_prn_sets_parses_nlos_rows_only(tmp_path: Path):
    csv_path = tmp_path / "mask.csv"
    _write_mask_csv(
        csv_path,
        [
            (100.0, 0, "G01", 1),
            (100.0, 0, "G02", 0),
            (101.0, 1, "E05", 0),
        ],
    )

    out = load_nlos_prn_sets(csv_path)
    assert out == {0: {"G02"}, 1: {"E05"}}


def test_load_nlos_prn_sets_missing_file_returns_empty(tmp_path: Path):
    assert load_nlos_prn_sets(tmp_path / "missing.csv") == {}


def test_nlos_weight_factor_never_zero():
    assert nlos_weight_factor(is_nlos=False, is_strong=False, k_weak=3.0, k_strong=5.0) == 1.0
    assert nlos_weight_factor(is_nlos=True, is_strong=False, k_weak=3.0, k_strong=5.0) == pytest.approx(1.0 / 3.0)
    assert nlos_weight_factor(is_nlos=True, is_strong=True, k_weak=3.0, k_strong=5.0) == pytest.approx(1.0 / 5.0)
    assert nlos_weight_factor(
        is_nlos=True,
        is_strong=False,
        k_weak=0.0,
        k_strong=0.0,
        min_weight=0.05,
    ) == 0.05


def test_epoch_prn_weights_uses_strong_set_when_present():
    tables = NlosMaskTables(
        weak={0: {"G01", "G02"}},
        strong={0: {"G02"}},
    )
    weights = epoch_prn_weights(0, ["G01", "G02", "G03"], tables, k_weak=2.0, k_strong=4.0)
    assert weights["G01"] == pytest.approx(0.5)
    assert weights["G02"] == pytest.approx(0.25)
    assert weights["G03"] == 1.0


def test_apply_mask_to_weights_multiplies_base_weights(tmp_path: Path):
    csv_path = tmp_path / "weak.csv"
    _write_mask_csv(csv_path, [(1.0, 3, "G07", 0)])
    tables = load_nlos_mask_tables(csv_path)
    out = apply_mask_to_weights(3, ["G07", "G08"], [2.0, 1.5], tables, k_weak=2.0)
    assert out[0] == pytest.approx(1.0)
    assert out[1] == pytest.approx(1.5)
