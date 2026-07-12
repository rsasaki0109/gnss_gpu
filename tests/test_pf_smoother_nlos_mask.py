"""Tests for PF smoother geometry NLOS mask plumbing."""

from __future__ import annotations

from pathlib import Path

import pytest

from gnss_gpu.nlos_mask import load_nlos_mask_tables
from gnss_gpu.pf_smoother_config import PfSmootherConfig


def _write_mask_csv(path: Path) -> None:
    path.write_text(
        "tow,epoch_idx,prn,is_los\n"
        "100.0,0,G01,0\n"
        "100.0,0,G02,1\n",
        encoding="utf-8",
    )


def test_pf_smoother_config_exposes_nlos_mask_fields():
    config = PfSmootherConfig(
        n_particles=10,
        sigma_pos=1.0,
        sigma_pr=3.0,
        position_update_sigma=None,
        predict_guide="spp",
        use_smoother=False,
        nlos_mask_csv="/tmp/mask.csv",
        nlos_strong_mask_csv="/tmp/strong.csv",
        nlos_k_weak=4.0,
        nlos_k_strong=6.0,
    )
    robust = config.parts().observations.robust
    assert robust.nlos_mask_csv == "/tmp/mask.csv"
    assert robust.nlos_strong_mask_csv == "/tmp/strong.csv"
    assert robust.nlos_k_weak == 4.0
    assert robust.nlos_k_strong == 6.0


def test_pf_smoother_run_loads_mask_tables_when_path_set(tmp_path: Path):
    mask_path = tmp_path / "mask.csv"
    _write_mask_csv(mask_path)
    config = PfSmootherConfig(
        n_particles=10,
        sigma_pos=1.0,
        sigma_pr=3.0,
        position_update_sigma=None,
        predict_guide="spp",
        use_smoother=False,
        nlos_mask_csv=str(mask_path),
    )
    robust = config.parts().observations.robust
    tables = None
    if str(robust.nlos_mask_csv).strip():
        tables = load_nlos_mask_tables(
            robust.nlos_mask_csv,
            robust.nlos_strong_mask_csv or None,
        )
    assert tables is not None
    assert tables.weak == {0: {"G01"}}


def test_pf_smoother_default_config_has_no_mask_path():
    config = PfSmootherConfig(
        n_particles=10,
        sigma_pos=1.0,
        sigma_pr=3.0,
        position_update_sigma=None,
        predict_guide="spp",
        use_smoother=False,
    )
    assert config.nlos_mask_csv == ""
    assert config.parts().observations.robust.nlos_k_weak == pytest.approx(3.0)
