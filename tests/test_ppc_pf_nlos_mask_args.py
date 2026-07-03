"""Smoke tests for PPC PF-domain NLOS mask CLI wiring."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gnss_gpu.nlos_mask import apply_mask_to_weights, load_nlos_mask_tables

REPO_ROOT = Path(__file__).resolve().parents[1]
EXP_PPC = REPO_ROOT / "experiments" / "exp_ppc_ctrbpf_fgo.py"


def test_exp_ppc_declares_pf_nlos_cli_flags():
    text = EXP_PPC.read_text(encoding="utf-8")
    assert "--pf-nlos-mask-path" in text
    assert "--pf-nlos-k-weak" in text
    assert "--pf-nlos-k-strong" in text
    assert "--pf-nlos-strong-mask-path" in text
    assert "--pf-nlos-preset" in text
    assert "pf_nlos_mask_tables=pf_nlos_tables_run" in text


def test_pf_nlos_mask_downweights_before_update_contract(tmp_path: Path):
    csv_path = tmp_path / "mask.csv"
    csv_path.write_text(
        "tow,epoch_idx,prn,is_los\n"
        "1.0,5,G07,0\n",
        encoding="utf-8",
    )
    tables = load_nlos_mask_tables(csv_path)
    weights = np.asarray([2.0, 1.0], dtype=np.float64)
    out = apply_mask_to_weights(5, ["G07", "G08"], weights, tables, k_weak=2.0)
    assert out[0] == pytest.approx(1.0)
    assert out[1] == pytest.approx(1.0)
