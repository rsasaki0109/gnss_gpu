"""Verify exact legacy metadata and explicitly changed actual-work counters."""

from copy import deepcopy

from bench_rbpf_move_field import encode


def check_metadata(original, actual, planned=True):
    left, right = deepcopy(original), deepcopy(actual)
    note = right.pop("move_plan")
    if not planned:
        assert note["fallback"] == 1 and note["planned"] == 0
        assert encode(left) == encode(right)
        return note
    assert note["planned"] == note["calls"] == 1 and note["fallback"] == 0
    assert note["particles"] == left["particles"] == right["particles"]
    assert note["logical_registrations"] == left["requested_moves"] == right["requested_moves"]
    assert note["executed_registrations"] + note["avoided_registrations"] == note["logical_registrations"]
    assert note["prefix_plans"] + note["prefix_replays"] == note["particles"]
    lm, rm = left.pop("move_key_memo"), right.pop("move_key_memo")
    assert lm["canonical_requests"] == note["logical_registrations"]
    assert rm["canonical_requests"] == note["executed_registrations"]
    for m in (lm, rm):
        assert m["canonical_requests"] == m["canonical_source_calls"] + m["canonical_identity_hits"] + m["canonical_empty_hits"]
        assert m["identity_entries"] == m["canonical_source_calls"] and m["canonical_bypass"] == 0
        assert not m["DD_reuse_enabled"]
        assert all(m[k] == 0 for k in m if k.startswith("DD_") and k != "DD_reuse_enabled")
        assert m["problem_entries"] == 0
    lc, rc = left.pop("native_canonical"), right.pop("native_canonical")
    assert lc["fallback_calls"] == rc["fallback_calls"] == 0
    assert lc["native_calls"] - rc["native_calls"] == lm["canonical_source_calls"] - rm["canonical_source_calls"] >= 0
    lr, rr = left.pop("move_reuse"), right.pop("move_reuse")
    for name in ("extend", "shrink"):
        assert lr["calls"][name] == rr["calls"][name]
        assert lr["requests"][name] == left["held_particles"]
        assert rr["requests"][name] == rr["calls"][name]
        assert lr["hits"][name] == lr["requests"][name] - lr["calls"][name]
        assert rr["hits"][name] == rr["uncacheable"][name] == lr["uncacheable"][name] == 0
    assert lr["extend_entries"] == rr["extend_entries"] and lr["shrink_entries"] == rr["shrink_entries"]
    assert lr["identity_entries"] >= rr["identity_entries"]
    assert encode(left) == encode(right)
    return note
