from __future__ import annotations

import json

from experiments.build_wp53_cross_basis_seed_pool import build_pool


def test_build_pool_discards_audit_fields(tmp_path) -> None:
    source = tmp_path / "rank1.json"
    source.write_text(
        json.dumps(
            {
                "production_input_truth": False,
                "hypotheses": [
                    {
                        "seed_id": 7,
                        "offset_ecef_m": [1, 2, 3],
                        "audit_median_error_m": 0.1,
                        "audit_sub50cm_epochs": 50,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    pool = build_pool(source, reference_rank=1)

    assert pool["production_input_truth"] is False
    assert pool["truth_usage"] == "none; audit fields are discarded"
    assert pool["seeds"] == [
        {"source_seed_id": 7, "offset_ecef_m": [1.0, 2.0, 3.0]}
    ]
    assert all(not any(key.startswith("audit_") for key in seed) for seed in pool["seeds"])
