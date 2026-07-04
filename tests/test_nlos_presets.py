"""Tests for shared NLOS preset constants."""

from __future__ import annotations

from argparse import Namespace

from gnss_gpu.nlos_presets import (
    DEFAULT_NLOS_K_WEAK,
    PPC_NLOS_MASK_PATH_TEMPLATE,
    PPC_PF_NLOS_SOFT_K3_ARGV,
    apply_ppc_pf_nlos_soft_k3_preset,
)


def test_ppc_mask_path_template_has_city_run_placeholders():
    assert "{city}" in PPC_NLOS_MASK_PATH_TEMPLATE
    assert "{run}" in PPC_NLOS_MASK_PATH_TEMPLATE
    formatted = PPC_NLOS_MASK_PATH_TEMPLATE.format(city="tokyo", run="run1")
    assert formatted.endswith("tokyo_run1_per_epoch_nlos.csv")


def test_apply_ppc_pf_nlos_soft_k3_preset_sets_defaults():
    args = Namespace(pf_nlos_mask_path="", pf_nlos_k_weak=1.0, pf_nlos_k_strong=1.0)
    apply_ppc_pf_nlos_soft_k3_preset(args)
    assert args.pf_nlos_mask_path == PPC_NLOS_MASK_PATH_TEMPLATE
    assert args.pf_nlos_k_weak == DEFAULT_NLOS_K_WEAK
    assert args.pf_nlos_k_strong == DEFAULT_NLOS_K_WEAK


def test_apply_ppc_pf_nlos_soft_k3_preset_keeps_explicit_mask_path():
    args = Namespace(
        pf_nlos_mask_path="/custom/mask.csv",
        pf_nlos_k_weak=1.0,
        pf_nlos_k_strong=1.0,
    )
    apply_ppc_pf_nlos_soft_k3_preset(args)
    assert args.pf_nlos_mask_path == "/custom/mask.csv"


def test_ppc_pf_nlos_soft_k3_argv_matches_preset_helper():
    assert "--pf-nlos-mask-path" in PPC_PF_NLOS_SOFT_K3_ARGV
    assert str(DEFAULT_NLOS_K_WEAK) in PPC_PF_NLOS_SOFT_K3_ARGV
