"""Retain the frozen full-state accuracy body;select np.int32 field schema."""

import hashlib
import json
from pathlib import Path
from types import FunctionType

import bench_rbpf_joint_covariance_field as base
from rbpf_joint_covariance_field_schema import make_type


def main():
    root = Path(__file__).resolve().parents[1]
    plan = json.loads(
        (root / "dist/rbpf_joint_covariance_field_20261017/plan.json").read_text(
            encoding="utf8"
        )
    )
    name = "benchmarks/bench_rbpf_joint_covariance_field.py"
    if (
        hashlib.sha256((root / name).read_bytes()).hexdigest()
        != plan["source_sha256"][name]
    ):
        raise ValueError("Frozen actual-copy and saved-state body changed")
    fn = base.main
    run = FunctionType(
        fn.__code__,
        dict(fn.__globals__, make_type=make_type),
        fn.__name__,
        fn.__defaults__,
        fn.__closure__,
    )
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
