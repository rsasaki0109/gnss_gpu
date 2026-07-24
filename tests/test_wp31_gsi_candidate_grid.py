from __future__ import annotations

from experiments.acquire_wp31_gsi_candidate_grid_cache import acquire_candidate_points


def test_acquire_candidate_points_keeps_ids() -> None:
    candidates = [{"candidate_id": 2, "position_ecef": [-3960000, 3350000, 3700000]}, {"candidate_id": 1, "position_ecef": [-3960001, 3350001, 3700001]}]
    def fetch(url: str) -> dict:
        return {"elevation": 2.0, "hsrc": "1m（レーザ）"} if "getelevation" in url else {"OutputData": {"geoidHeight": "36.0"}}
    points = acquire_candidate_points(candidates, fetch_json=fetch, max_workers=2)
    assert [row["candidate_id"] for row in points] == [1, 2]
    assert all(row["elevation_m"] == 2.0 for row in points)
