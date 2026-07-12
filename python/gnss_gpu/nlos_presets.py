"""Shared NLOS soft-weight preset constants for PF smoother and PPC CLIs."""

from __future__ import annotations

from argparse import Namespace

PPC_NLOS_MASK_PATH_TEMPLATE = (
    "experiments/results/plateau_nlos_phase33/{city}_{run}_per_epoch_nlos.csv"
)

DEFAULT_NLOS_K_WEAK = 3.0
DEFAULT_NLOS_K_STRONG = 3.0

PPC_PF_NLOS_SOFT_K3_ARGV = [
    "--pf-nlos-mask-path",
    PPC_NLOS_MASK_PATH_TEMPLATE,
    "--pf-nlos-k-weak",
    str(DEFAULT_NLOS_K_WEAK),
    "--pf-nlos-k-strong",
    str(DEFAULT_NLOS_K_STRONG),
]


def apply_ppc_pf_nlos_soft_k3_preset(args: Namespace) -> None:
    """Fill PPC PF NLOS flags when ``--pf-nlos-preset soft-k3`` is selected."""
    if not str(getattr(args, "pf_nlos_mask_path", "")).strip():
        args.pf_nlos_mask_path = PPC_NLOS_MASK_PATH_TEMPLATE
    args.pf_nlos_k_weak = float(DEFAULT_NLOS_K_WEAK)
    args.pf_nlos_k_strong = float(DEFAULT_NLOS_K_STRONG)


def apply_pf_smoother_nlos_soft_k3_defaults(args: Namespace) -> None:
    """Set PF smoother NLOS k factors without enabling a mask path."""
    args.nlos_k_weak = float(DEFAULT_NLOS_K_WEAK)
    args.nlos_k_strong = float(DEFAULT_NLOS_K_STRONG)
