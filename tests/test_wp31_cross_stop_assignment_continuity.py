from experiments.audit_wp31_cross_stop_assignment_continuity import audit_continuity


def test_audit_continuity_requires_matching_generation():
    left = {1: {("G01", "G02", 190293673, 0), ("E01", "E02", 190293673, 3)}}
    right = {2: {("G01", "G02", 190293673, 4), ("E01", "E02", 190293673, 3)}}
    row = audit_continuity(left, right)[0]
    assert row["shared_raw_keys"] == 2
    assert row["shared_versioned_keys"] == 1
    assert row["continuous"] is True


def test_audit_continuity_rejects_raw_match_after_reset():
    left = {1: {("G01", "G02", 190293673, 0)}}
    right = {2: {("G01", "G02", 190293673, 7)}}
    row = audit_continuity(left, right)[0]
    assert row["shared_raw_keys"] == 1
    assert row["shared_versioned_keys"] == 0
    assert row["min_generation_delta"] == 7
    assert row["continuous"] is False
