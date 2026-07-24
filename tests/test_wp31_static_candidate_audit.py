from __future__ import annotations

import numpy as np

from experiments.audit_wp31_static_candidates import audit_candidates


def test_audit_candidates_ranks_without_mutating_source() -> None:
    candidates = [
        {"candidate_id": 1, "position_ecef": [2.0, 0.0, 0.0]},
        {"candidate_id": 0, "position_ecef": [0.1, 0.0, 0.0]},
    ]
    rows = audit_candidates(candidates, np.zeros((3, 3)))
    assert [row["candidate_id"] for row in rows] == [0, 1]
    assert rows[0]["audit_sub50cm"] == 1
    assert "audit_error_m" not in candidates[0]
