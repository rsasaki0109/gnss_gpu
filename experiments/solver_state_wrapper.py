"""Solver-state lightweight wrapper -- opt-in research path for the PPC ceiling.

The default product contract excludes ``rtk_*`` / ``solver_demo5_*`` columns
from runtime features.  See ``internal_docs/product_deliverable/README.md`` §1
and the "Why the residual is not closeable" section of
``internal_docs/product_deliverable/PLATEAU_BRIDGE_INTEGRATION.md``.

PR #42 identified three out-of-scope paths to lift the 8.13 pp tokyo/run2
run-MAE residual.  This module realises the first one
("Solver-state lightweight wrapper: expose a curated subset of demo5
ambiguity-fix-state indicators as runtime features") as a curated, opt-in
extension to the runtime feature contract.

Curated subset
--------------

Six demo5 ambiguity-fix-state indicators that are already materialised in the
augmented window CSV and used as training targets in
``train_ppc_solver_transition_surrogate_stack._transition_targets``:

* ``solver_demo5_ratio_mean``
* ``solver_demo5_ratio_p90``
* ``solver_demo5_ratio_p95``
* ``solver_demo5_ratio_mean_past_delta``
* ``rtk_lock_p90_p50``
* ``rtk_lock_p90_p50_past_delta``

No other ``rtk_*`` / ``solver_*`` column is exposed.  The default gatekeeper
``experiments/_common._is_metadata_or_label`` keeps dropping every
``rtk_*`` / ``solver_*`` column (including the curated six); this wrapper
is the only mechanism by which a research script can selectively
re-introduce the allowlist as runtime features.

Usage
-----

Research scripts that opt in to the lifted contract instantiate
``SolverStateWrapper``, call ``validate(df)`` to assert the columns are
present and numeric, and call ``runtime_feature_columns()`` to learn the
curated allowlist.  The adopted product training and inference paths
(``train_ppc_solver_transition_surrogate_stack.py``,
``product_inference_model.py``) do not import this module, so the deployed
metric (run MAE 1.79 pp) is unaffected.

Caveat
------

These six columns are *also* the binary-classification targets used by the
adopted product model.  Any research training script that consumes them as
runtime features must omit ``_transition_targets`` (or replace it with a
target derived from a different signal); otherwise the model trivially
solves its own targets and reports leak-inflated metrics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


CURATED_SOLVER_STATE_COLUMNS: tuple[str, ...] = (
    "solver_demo5_ratio_mean",
    "solver_demo5_ratio_p90",
    "solver_demo5_ratio_p95",
    "solver_demo5_ratio_mean_past_delta",
    "rtk_lock_p90_p50",
    "rtk_lock_p90_p50_past_delta",
)


@dataclass(frozen=True)
class SolverStateWrapper:
    """Curate solver state for opt-in runtime feature exposure.

    The curated allowlist is fixed at the module level via
    ``CURATED_SOLVER_STATE_COLUMNS``.  The constructor takes no arguments;
    custom column lists are intentionally not supported, so that the
    six-column contract cannot be widened by a caller.
    """

    def runtime_feature_columns(self) -> tuple[str, ...]:
        return CURATED_SOLVER_STATE_COLUMNS

    def validate(self, df: pd.DataFrame) -> None:
        duplicates = sorted(
            c for c in CURATED_SOLVER_STATE_COLUMNS if (df.columns == c).sum() > 1
        )
        if duplicates:
            raise ValueError(
                "SolverStateWrapper.validate: duplicate curated columns in "
                f"input DataFrame: {duplicates}.  Drop or rename duplicates "
                "before validation; pandas returns a DataFrame rather than a "
                "Series for duplicated labels and the curator cannot reason "
                "about which copy is authoritative."
            )
        missing = [c for c in CURATED_SOLVER_STATE_COLUMNS if c not in df.columns]
        if missing:
            raise KeyError(
                "SolverStateWrapper.validate: input DataFrame is missing "
                f"curated columns: {missing}.  The wrapper requires the demo5 "
                "ambiguity-fix-state indicators produced by the PPC "
                "augmentation pipeline."
            )
        for col in CURATED_SOLVER_STATE_COLUMNS:
            ser = df[col]
            if not pd.api.types.is_numeric_dtype(ser):
                raise TypeError(
                    f"SolverStateWrapper.validate: column {col!r} is not "
                    f"numeric (dtype={ser.dtype}); the curator only accepts "
                    "numeric solver-state columns."
                )

    def curate(
        self,
        df: pd.DataFrame,
        *,
        neutral_value: float = 0.0,
    ) -> pd.DataFrame:
        """Return ``df[curated_columns]`` with NaN/Inf replaced by ``neutral_value``.

        Default ``0.0`` lets linear and tree models treat missing solver state
        as "no information".  Callers that prefer column-median imputation
        must pre-process before calling.  ``neutral_value`` itself must be
        finite; otherwise the wrapper would emit non-finite values while
        claiming sanitisation.
        """
        if not np.isfinite(neutral_value):
            raise ValueError(
                "SolverStateWrapper.curate: neutral_value must be finite, "
                f"got {neutral_value!r}.  The curator's contract is that the "
                "returned DataFrame has no NaN/Inf in curated columns; "
                "passing a non-finite replacement would silently violate it."
            )
        self.validate(df)
        sub = df.loc[:, list(CURATED_SOLVER_STATE_COLUMNS)]
        arr = sub.to_numpy(dtype=float, copy=True)
        non_finite = ~np.isfinite(arr)
        if non_finite.any():
            n_rows = int(non_finite.any(axis=1).sum())
            LOGGER.warning(
                "SolverStateWrapper.curate: replaced non-finite values in "
                "%d / %d rows with %s",
                n_rows,
                len(arr),
                neutral_value,
            )
            arr[non_finite] = float(neutral_value)
        return pd.DataFrame(
            arr, columns=list(CURATED_SOLVER_STATE_COLUMNS), index=sub.index
        )


def is_curated_solver_state_column(name: str) -> bool:
    """Return True iff ``name`` is in the curated allowlist."""
    return name in CURATED_SOLVER_STATE_COLUMNS
