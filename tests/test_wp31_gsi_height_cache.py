from __future__ import annotations

from experiments.acquire_wp31_gsi_height_cache import build_cache


def test_build_cache_uses_candidate_median_and_offline_result() -> None:
    source = {
        "segment": [10, 20],
        "candidates": [
            {"position_ecef": [-3961767.0, 3349008.0, 3698310.0]},
            {"position_ecef": [-3961766.0, 3349009.0, 3698311.0]},
            {"position_ecef": [-3961765.0, 3349010.0, 3698312.0]},
        ],
    }
    calibration = {"calibration_points": [{"name": "accepted"}]}
    urls: list[str] = []

    def fetch(url: str) -> dict:
        urls.append(url)
        if "getelevation" in url:
            return {"elevation": 1.5, "hsrc": "1m"}
        return {"OutputData": {"geoidHeight": "36.4"}}

    result = build_cache(
        source,
        calibration,
        query_basis="candidate median",
        fetch_json=fetch,
        acquired_utc="2026-01-01T00:00:00Z",
    )

    assert result["runtime_network_required"] is False
    assert result["target_point"]["elevation_m"] == 1.5
    assert result["target_point"]["geoid_height_m"] == 36.4
    assert result["target_point"]["query_basis"] == "candidate median"
    assert len(urls) == 2
    assert "lat=" in urls[0] and "lon=" in urls[0]
