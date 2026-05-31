from collections import Counter

from experiments.audit_gici_reference_pool import parse_selected_counts


def test_parse_selected_counts_handles_rank_suffix_and_commas() -> None:
    raw = "xd_gici_c4+rnk:12,pf_bridge+rnk:3,xd_gici_c4+rnk:2"

    assert parse_selected_counts(raw) == Counter(
        {
            "xd_gici_c4": 14,
            "pf_bridge": 3,
        },
    )


def test_parse_selected_counts_ignores_malformed_items() -> None:
    raw = "xd_gici_c4+rnk:12,bad,xd_gici_z:not_int,:4"

    assert parse_selected_counts(raw) == Counter(
        {
            "xd_gici_c4": 12,
        },
    )
