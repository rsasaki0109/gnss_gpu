"""Accuracy-only original PF state capture with checked covariance copies."""

import argparse
import ast
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import bench_rbpf_move_field as field
from rbpf_feedback_mass import NativeFeedbackMassStep
from rbpf_joint_covariance_sized import make_type

FIELD_SHA256 = "b21c348b72f37c61007a8c0acb7bdcf4114fc2911e9199cbdc279c8336086209"


def adapt(select, install):
    path = Path(field.__file__)
    if hashlib.sha256(path.read_bytes()).hexdigest() != FIELD_SHA256:
        raise ValueError("Frozen saved-state field body changed")
    node = next(
        n
        for n in ast.parse(path.read_bytes()).body
        if isinstance(n, ast.FunctionDef) and n.name == "main"
    )
    baseline = deepcopy(node)
    at = next(
        i
        for i, n in enumerate(node.body)
        if isinstance(n, ast.Import) and any(a.name == "wp16_run_rbpf" for a in n.names)
    )
    node.body[at + 1 : at + 1] = ast.parse("_d272_install(rbpf_fgo)").body
    restored = deepcopy(node)
    restored.body.pop(at + 1)
    assert ast.dump(restored, include_attributes=False) == ast.dump(
        baseline, include_attributes=False
    )
    scope = dict(field.main.__globals__, MoveBatchStep=select, _d272_install=install)
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
            str(path),
            "exec",
            dont_inherit=True,
        ),
        scope,
    )
    return scope["main"]


def main():
    parser = argparse.ArgumentParser(add_help=False)
    for name in ("extension", "parent-extension", "mass-extension", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--cluster-relin", type=int, choices=(0, 1), required=True)
    own, remaining = parser.parse_known_args()
    os.environ["RBPF_CLUSTER_RELIN"] = str(own.cluster_relin)
    installed, engines, configs, checked = [], [], [], []

    def install(module):
        original = module.RbpfGtsamRtkTc
        installed.append((module, original, module.RbpfFgo.__init__))
        chosen, engine = make_type(original)
        engines.append(engine)
        bulk = engine.copy_pairs

        def exact_copy(rtk, active, jm, key_x, dim, scope, literal):
            expected = rtk.nav.P.copy()
            view = SimpleNamespace(
                nav=SimpleNamespace(P=expected, na=rtk.nav.na), IB=rtk.IB
            )
            literal(view, active, jm)
            result = bulk(rtk, active, jm, key_x, dim, scope, literal)
            assert (
                rtk.nav.P.dtype == expected.dtype
                and rtk.nav.P.shape == expected.shape
                and rtk.nav.P.tobytes() == expected.tobytes()
            )
            checked.append(
                dict(
                    position_dim=dim,
                    active=len(active),
                    pairs=len(active) * (len(active) - 1) // 2,
                )
            )
            return result

        engine.copy_pairs = exact_copy
        module.RbpfGtsamRtkTc = chosen

    def select(runtime, scorer, mode):
        assert mode == "native_cpu" and runtime.cfg.cluster_relin == own.cluster_relin
        configs.append(vars(runtime.cfg).copy())
        return NativeFeedbackMassStep(
            runtime,
            scorer,
            mode,
            extension=own.extension,
            parent_extension=own.parent_extension,
            mass_extension=own.mass_extension,
        )

    run = adapt(select, install)
    sys.argv = [sys.argv[0], *remaining, "--output-dir", str(own.output_dir)]
    try:
        code = run()
    finally:
        for module, original, initialize in reversed(installed):
            module.RbpfGtsamRtkTc, module.RbpfFgo.__init__ = original, initialize
    assert code == 0 and len(engines) == 1
    engine = engines[0]
    assert engine.full_matrix_calls > 0 and engine.fallback_calls == 0
    assert len(checked) == engine.calls + engine.small_literal_calls
    (own.output_dir / "joint.json").write_text(
        json.dumps(
            dict(
                configs=configs,
                engine_counts=engine.counts(),
                small_literal_calls=engine.small_literal_calls,
                checked_copy_calls=checked,
                all_actual_copy_bits_equal=True,
                AST_proofs=engine.proofs,
                original_field_body_retained_with_install_only=True,
                accuracy_only=True,
                aliases_restored=True,
            ),
            indent=2,
        )
        + "\n",
        encoding="utf8",
    )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
