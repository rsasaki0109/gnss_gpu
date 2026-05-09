"""Shared helpers for the PPC FIX-rate predictor experiment scripts.

Centralised here to avoid cross-importing between experiment scripts.
"""

from __future__ import annotations


def _is_metadata_or_label(name: str) -> bool:
    """Return True if `name` is a metadata or label column that must not be
    fed to the classifier / regressor as a deployable feature.

    Excludes:
    - window-identity columns (city / run / window_index / window bounds)
    - sim_matched_epochs and actual FIX labels
    - demo5 solver internals (`rtk_*`, `solver_*`) used as training targets
    - prediction / error columns emitted by other model passes
    - constant-global-rate reference columns
    """
    metadata = {
        "city",
        "run",
        "window_index",
        "window_start_tow",
        "window_end_tow",
        "sim_matched_epochs",
        "actual_fixed_epochs",
        "actual_fix_rate_pct",
        "actual_fix_rate_fraction",
    }
    if name in metadata:
        return True
    if name.startswith(("rtk_", "solver_")):
        return True
    if name.endswith("_pred_fix_rate_pct") or name.endswith("_error_pp"):
        return True
    if name.startswith("constant_global_rate"):
        return True
    return False
