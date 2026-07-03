"""Tests for PF NLOS production prep helper."""

from __future__ import annotations

from pathlib import Path

from experiments.prepare_pf_nlos_production import (
    _mask_csv,
    _plateau_dir,
    _triangle_cache,
)


def test_path_helpers_for_tokyo_run1():
    assert _plateau_dir(Path("E:/datasets/plateau"), "tokyo/run1") == Path(
        "E:/datasets/plateau/tokyo_run1"
    )
    assert _triangle_cache(Path("E:/datasets/plateau_cache"), "tokyo/run1") == Path(
        "E:/datasets/plateau_cache/tokyo_run1_triangles.npz"
    )
    assert _mask_csv("tokyo/run1").name == "tokyo_run1_per_epoch_nlos.csv"
