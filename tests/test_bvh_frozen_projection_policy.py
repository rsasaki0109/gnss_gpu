"""Historical source routing and the narrowly scoped failed-fixture replacement."""

import hashlib
from types import SimpleNamespace

import bvh_frozen_projection_policy as policy
import pytest


def test_only_the_exact_initial_suite_is_ignored(tmp_path):
    assert policy.ignore_failed_initial_suite(policy.INITIAL)
    for path in (
        tmp_path / policy.INITIAL.name,
        policy.ROOT / "tests/test_bvh_tiled_geometry_checked.py",
        policy.PROJECTION,
        policy.ROOT / "tests/test_bvh_cpu_duplicates.py",
    ):
        assert not policy.ignore_failed_initial_suite(path)


def test_changed_failed_fixture_is_reported(monkeypatch):
    monkeypatch.setattr(policy, "INITIAL_SHA256", "0" * 64)
    with pytest.raises(AssertionError):
        policy.ignore_failed_initial_suite(policy.INITIAL)


def test_only_historical_projection_uses_the_frozen_source(tmp_path):
    modules = [
        SimpleNamespace(ROOT="untouched", GEN=SimpleNamespace(ROOT="untouched"))
        for _ in range(3)
    ]
    items = [
        SimpleNamespace(path=p, module=m)
        for p, m in zip(
            [
                policy.PROJECTION,
                tmp_path / policy.PROJECTION.name,
                policy.ROOT / "tests/test_bvh_iterative_build_projection.py",
            ],
            modules,
            strict=True,
        )
    ]
    policy.route_historical_projection(items)
    assert modules[0].ROOT == modules[0].GEN.ROOT == policy.BASE
    for module in modules[1:]:
        assert module.ROOT == module.GEN.ROOT == "untouched"


@pytest.mark.parametrize(
    "name,digest",
    [
        (
            "src/raytrace/bvh.cu",
            "7d9d06f128710f12eea70b09b8f0d475d16223957ddfdf41a7cf5ec77ac09d98",
        ),
        (
            "include/gnss_gpu/bvh.h",
            "50b44813b584f5b6778fb983e07b587b6dc8ee268326672de9efbbff7f496e35",
        ),
        (
            "python/gnss_gpu/_bvh_bindings.cpp",
            "a043d669a98b24772c9ce4a3639d21bb869d65a35c30843979190025e4b44845",
        ),
    ],
)
def test_frozen_historical_sources_are_the_original_raw_bytes(name, digest):
    assert hashlib.sha256((policy.BASE / name).read_bytes()).hexdigest() == digest
