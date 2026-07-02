#!/usr/bin/env python3
"""GICI cluster override for the PPC2024 candidate-selection ranker.

Pure, dependency-light (numpy only) so it is unit-testable without the heavy
``exp_ppc_ctrbpf_fgo`` import graph. Implements the documented runtime rule from
``internal_docs/plan.md`` ("Phase 43 runtime GICI override summary"):

  When the base supervised ranker pick is a *high-risk* GICI variant, re-pick
  within the same ``xd_gici`` family toward a tight, low-RMS cluster near the
  original pick. This recovers some of the Phase 42 oracle span gains without
  using truth, and is the mode the Phase 43/71 production runs used for
  nagoya/run2.

Candidate items are ``(label, pos_ecef, diag_row, key)`` tuples, matching the
``collected`` list built in ``exp_ppc_ctrbpf_fgo``. Distances are Euclidean in
ECEF metres (≈ local ENU metres at these separations).
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

# High-risk GICI variants that trigger the override when picked by the ranker.
RANKER_GICI_HIGH_RISK = frozenset(
    {"xd_gici_c4", "xd_gici_oa", "xd_gici_combo", "xd_gici_z", "xd_gici_hs"}
)


def _rms(cand: Sequence) -> float:
    try:
        return float(cand[2]["final_residual_rms"])
    except (KeyError, TypeError, ValueError, IndexError):
        return float("nan")


def gici_cluster_override(
    pick: Sequence,
    collected: Sequence[Sequence],
    *,
    rms_rank_max: int = 12,
    cluster50_min: int = 6,
    dist_to_pick_max_m: float = 0.8,
    cluster_radius_m: float = 0.5,
) -> Optional[Sequence]:
    """Re-pick within the xd_gici family toward a tight low-RMS cluster.

    Among the ``xd_gici`` family (synthetic ``pf_bridge`` excluded), keep
    candidates whose rank by ``final_residual_rms`` within the family is
    ``<= rms_rank_max``, that sit in a cluster of ``>= cluster50_min`` family
    members within ``cluster_radius_m``, and lie within ``dist_to_pick_max_m`` of
    ``pick``. Return the one with the largest cluster (tie-break: lowest RMS), or
    ``None`` to keep the original pick.

    The defaults reproduce the documented Phase 43/71 thresholds (12 / 6 / 0.8).
    The caller is responsible for only invoking this when ``pick``'s label is in
    :data:`RANKER_GICI_HIGH_RISK`.
    """
    pick_pos = np.asarray(pick[1], dtype=np.float64)
    family = [
        c
        for c in collected
        if str(c[0]).startswith("xd_gici") and str(c[0]) != "pf_bridge"
    ]
    if len(family) < cluster50_min:
        return None
    rms_order = sorted(family, key=_rms)
    rms_rank = {id(c): r + 1 for r, c in enumerate(rms_order)}
    positions = {id(c): np.asarray(c[1], dtype=np.float64) for c in family}

    eligible: list[tuple[int, float, Sequence]] = []
    for c in family:
        cpos = positions[id(c)]
        if float(np.linalg.norm(cpos - pick_pos)) > dist_to_pick_max_m:
            continue
        if rms_rank[id(c)] > rms_rank_max:
            continue
        cluster50 = sum(
            1
            for o in family
            if float(np.linalg.norm(positions[id(o)] - cpos)) <= cluster_radius_m
        )
        if cluster50 < cluster50_min:
            continue
        eligible.append((cluster50, _rms(c), c))
    if not eligible:
        return None
    # Largest cluster wins; tie-break by lowest RMS.
    eligible.sort(key=lambda e: (-e[0], e[1]))
    return eligible[0][2]
