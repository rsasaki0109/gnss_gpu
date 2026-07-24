from experiments.acquire_wp31_gsi_calibration_cache import acquire_point


def test_acquire_point_records_official_height_provenance():
    responses = iter(
        [
            {"elevation": "12.3", "hsrc": "1m（レーザ）"},
            {"OutputData": {"geoidHeight": "37.4"}},
        ]
    )
    point = acquire_point(
        "anchor",
        [-3810241.8318932652, 3567866.785499446, 3652890.7702096337],
        fetch_json=lambda _url: next(responses),
    )
    assert point["elevation_m"] == 12.3
    assert point["geoid_height_m"] == 37.4
    assert point["dem_query_url"].startswith("https://cyberjapandata2.gsi.go.jp/")
