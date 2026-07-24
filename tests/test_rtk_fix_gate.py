from gnss_gpu.rtk_fix_gate import trusted_fix_gate


def _decision(**overrides):
    values = dict(
        map_float_separation_m=0.2,
        map_ddpr_separation_m=1.0,
        last_ddpr_pairs=12,
        ddpr_age_epochs=3,
        max_float_separation_m=0.5,
        max_ddpr_separation_m=1.75,
        min_ddpr_pairs=9,
        max_ddpr_age_epochs=4,
    )
    values.update(overrides)
    return trusted_fix_gate(**values)


def test_trusted_fix_gate_accepts_supported_consensus() -> None:
    assert _decision().passed


def test_trusted_fix_gate_rejects_each_untrusted_axis() -> None:
    assert not _decision(map_float_separation_m=0.6).passed
    assert not _decision(map_ddpr_separation_m=1.8).passed
    assert not _decision(last_ddpr_pairs=8).passed
    assert not _decision(ddpr_age_epochs=5).passed
    assert not _decision(map_ddpr_separation_m=float("nan")).passed


def test_trusted_fix_gate_boundaries_are_inclusive() -> None:
    assert _decision(
        map_float_separation_m=0.5,
        map_ddpr_separation_m=1.75,
        last_ddpr_pairs=9,
        ddpr_age_epochs=4,
    ).passed
