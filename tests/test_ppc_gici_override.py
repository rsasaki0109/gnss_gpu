"""Unit tests for the PPC2024 GICI cluster override (experiments/ppc_gici_override.py).

Pure-numpy logic, no native kernels — runs anywhere pytest + numpy are present.
Covers the documented Phase 43/71 rule: high-risk GICI pick + a tight low-RMS
xd_gici cluster within 0.8 m -> override; otherwise keep the pick.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

from ppc_gici_override import (  # noqa: E402
    RANKER_GICI_HIGH_RISK,
    gici_cluster_override,
)


def _cand(label: str, xyz, rms: float):
    """Build a (label, pos_ecef, diag_row, key) candidate tuple."""
    return (label, np.asarray(xyz, dtype=float), {"final_residual_rms": str(rms)}, (rms, rms))


def _tight_cluster(base, n=6, start_rms=0.20):
    """n xd_gici members packed within ~10 cm of `base`."""
    return [
        _cand(f"xd_gici_m{k}", np.asarray(base) + [0.10 + 0.01 * k, 0.0, 0.0], start_rms + 0.001 * k)
        for k in range(n)
    ]


def test_high_risk_set_matches_spec():
    assert RANKER_GICI_HIGH_RISK == {
        "xd_gici_c4",
        "xd_gici_oa",
        "xd_gici_combo",
        "xd_gici_z",
        "xd_gici_hs",
    }


def test_override_picks_tight_nearby_cluster():
    base = np.zeros(3)
    pick = _cand("xd_gici_c4", base, 0.30)  # high-risk pick
    collected = [pick, *_tight_cluster(base)]
    out = gici_cluster_override(pick, collected)
    assert out is not None
    assert out[0].startswith("xd_gici_m")


def test_far_low_rms_candidate_is_not_chosen():
    base = np.zeros(3)
    pick = _cand("xd_gici_c4", base, 0.30)
    far = _cand("xd_gici_far", [5.0, 0.0, 0.0], 0.01)  # best RMS but >0.8 m away
    collected = [pick, far, *_tight_cluster(base)]
    out = gici_cluster_override(pick, collected)
    assert out is not None
    assert out[0] != "xd_gici_far"
    assert float(np.linalg.norm(out[1] - pick[1])) <= 0.8


def test_pf_bridge_excluded_from_family():
    base = np.zeros(3)
    pick = _cand("xd_gici_c4", base, 0.30)
    # A pf_bridge sits right on the cluster but must never be the override target.
    bridge = _cand("pf_bridge", [0.1, 0.0, 0.0], 0.001)
    collected = [pick, bridge, *_tight_cluster(base)]
    out = gici_cluster_override(pick, collected)
    assert out is not None
    assert out[0] != "pf_bridge"


def test_no_override_when_cluster_too_small():
    base = np.zeros(3)
    pick = _cand("xd_gici_c4", base, 0.30)
    collected = [pick, *_tight_cluster(base, n=4)]  # only 4 < cluster50_min=6
    assert gici_cluster_override(pick, collected) is None


def test_no_override_when_cluster_outside_dist():
    base = np.zeros(3)
    pick = _cand("xd_gici_c4", base, 0.30)
    # A full 6-member cluster, but 2 m away from the pick.
    far_base = np.asarray([2.0, 0.0, 0.0])
    collected = [pick, *_tight_cluster(far_base)]
    assert gici_cluster_override(pick, collected) is None


def test_largest_cluster_wins_over_smaller():
    base = np.zeros(3)
    pick = _cand("xd_gici_c4", base, 0.30)
    # Two *spatially isolated* clusters (separated by >cluster_radius=0.5 m so
    # neither counts the other's members): a 7-member higher-RMS cluster near the
    # pick, and a 6-member lower-RMS cluster further out (still within 0.8 m).
    big = [
        _cand(f"xd_gici_m{k}", base + [0.10 + 0.005 * k, 0.0, 0.0], 0.25 + 0.001 * k)
        for k in range(7)
    ]
    small = [
        _cand(f"xd_gici_s{k}", base + [0.70 + 0.005 * k, 0.0, 0.0], 0.10 + 0.001 * k)
        for k in range(6)
    ]
    out = gici_cluster_override(pick, [pick, *big, *small])
    assert out is not None
    # The larger cluster (the "m" members) should win despite worse RMS.
    assert out[0].startswith("xd_gici_m")


def test_rms_rank_gate_excludes_low_ranked_members():
    base = np.zeros(3)
    pick = _cand("xd_gici_c4", base, 0.30)
    # 6 tight members but all with very high RMS so their family rms_rank is poor;
    # pad the family with 12 better-RMS but far-away members to push ranks > 12.
    cluster = _tight_cluster(base, n=6, start_rms=5.0)
    fillers = [
        _cand(f"xd_gici_f{k}", [10.0 + k, 0.0, 0.0], 0.10 + 0.001 * k) for k in range(12)
    ]
    out = gici_cluster_override(pick, [pick, *cluster, *fillers])
    assert out is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
