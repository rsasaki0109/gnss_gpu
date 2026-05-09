"""Tests for ``experiments.solver_state_wrapper``."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = REPO_ROOT / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

from solver_state_wrapper import (  # noqa: E402
    CURATED_SOLVER_STATE_COLUMNS,
    SolverStateWrapper,
    is_curated_solver_state_column,
)


def _frame_with_curated(extra_cols: dict[str, list] | None = None) -> pd.DataFrame:
    rows = 4
    data = {col: [1.0] * rows for col in CURATED_SOLVER_STATE_COLUMNS}
    if extra_cols:
        data.update(extra_cols)
    return pd.DataFrame(data)


def test_allowlist_size_and_membership():
    expected = {
        "solver_demo5_ratio_mean",
        "solver_demo5_ratio_p90",
        "solver_demo5_ratio_p95",
        "solver_demo5_ratio_mean_past_delta",
        "rtk_lock_p90_p50",
        "rtk_lock_p90_p50_past_delta",
    }
    assert set(CURATED_SOLVER_STATE_COLUMNS) == expected
    assert len(CURATED_SOLVER_STATE_COLUMNS) == len(expected)


def test_is_curated_solver_state_column_rejects_uncurated():
    assert is_curated_solver_state_column("rtk_lock_p90_p50") is True
    assert is_curated_solver_state_column("solver_demo5_ratio_mean") is True
    assert is_curated_solver_state_column("rtk_pos") is False
    assert is_curated_solver_state_column("solver_demo5_ratio_p99") is False
    assert is_curated_solver_state_column("city") is False


def test_runtime_feature_columns_matches_constant():
    assert SolverStateWrapper().runtime_feature_columns() == CURATED_SOLVER_STATE_COLUMNS


def test_validate_fails_on_missing_columns():
    df = pd.DataFrame({"solver_demo5_ratio_mean": [1.0]})
    with pytest.raises(KeyError, match="missing curated columns"):
        SolverStateWrapper().validate(df)


def test_validate_fails_on_non_numeric_column():
    df = _frame_with_curated()
    df["rtk_lock_p90_p50"] = ["a", "b", "c", "d"]
    with pytest.raises(TypeError, match="not numeric"):
        SolverStateWrapper().validate(df)


def test_curate_returns_only_curated_columns():
    df = _frame_with_curated(extra_cols={"city": ["x"] * 4, "actual_fix_rate_pct": [0.5] * 4})
    curated = SolverStateWrapper().curate(df)
    assert list(curated.columns) == list(CURATED_SOLVER_STATE_COLUMNS)


def test_curate_replaces_non_finite_values():
    df = _frame_with_curated()
    df.loc[0, "rtk_lock_p90_p50"] = float("nan")
    df.loc[1, "solver_demo5_ratio_mean"] = float("inf")
    df.loc[2, "solver_demo5_ratio_p95"] = float("-inf")
    curated = SolverStateWrapper().curate(df, neutral_value=-1.0)
    assert curated.loc[0, "rtk_lock_p90_p50"] == -1.0
    assert curated.loc[1, "solver_demo5_ratio_mean"] == -1.0
    assert curated.loc[2, "solver_demo5_ratio_p95"] == -1.0
    assert curated.loc[3, "rtk_lock_p90_p50"] == 1.0


def test_curate_passes_through_finite_values_unchanged():
    df = _frame_with_curated()
    df["rtk_lock_p90_p50"] = [10.0, 20.0, 30.0, 40.0]
    curated = SolverStateWrapper().curate(df)
    np.testing.assert_array_equal(
        curated["rtk_lock_p90_p50"].to_numpy(), np.array([10.0, 20.0, 30.0, 40.0])
    )


def test_default_product_gatekeeper_still_drops_uncurated_solver_columns():
    """The wrapper does not modify ``_common._is_metadata_or_label``; the
    default contract still excludes ``rtk_*`` / ``solver_*`` from runtime
    features unless callers explicitly opt in via the curated allowlist."""
    from _common import _is_metadata_or_label  # noqa: PLC0415

    assert _is_metadata_or_label("rtk_lock_p90_p50") is True
    assert _is_metadata_or_label("rtk_pos") is True
    assert _is_metadata_or_label("solver_demo5_ratio_mean") is True
    assert _is_metadata_or_label("solver_demo5_ratio_p99") is True


def test_constructor_rejects_columns_kwarg():
    """The allowlist is fixed; passing ``columns=...`` must fail so callers
    cannot widen the curated six-column contract."""
    with pytest.raises(TypeError):
        SolverStateWrapper(columns=("rtk_lock_p90_p50",))  # type: ignore[call-arg]


def test_curate_rejects_non_finite_neutral_value():
    df = _frame_with_curated()
    wrapper = SolverStateWrapper()
    with pytest.raises(ValueError, match="neutral_value must be finite"):
        wrapper.curate(df, neutral_value=float("nan"))
    with pytest.raises(ValueError, match="neutral_value must be finite"):
        wrapper.curate(df, neutral_value=float("inf"))


def test_validate_rejects_duplicate_curated_columns():
    df = _frame_with_curated()
    df["dup"] = [1.0, 2.0, 3.0, 4.0]
    df = df.rename(columns={"dup": "rtk_lock_p90_p50"})
    assert (df.columns == "rtk_lock_p90_p50").sum() == 2
    with pytest.raises(ValueError, match="duplicate curated columns"):
        SolverStateWrapper().validate(df)
