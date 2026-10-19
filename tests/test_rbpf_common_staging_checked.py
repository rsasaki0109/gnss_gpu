"""Common replay keeps all logical registration, callback and full PF state."""

from copy import deepcopy
import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
from bench_rbpf_feedback_mass import _bind_fixture_callbacks  # noqa: E402
from rbpf_common_staging import CommonStagingStep  # noqa: E402
from rbpf_feedback_cases import make_feedback_case  # noqa: E402
from rbpf_feedback_mass import NativeFeedbackMassStep  # noqa: E402
from rbpf_move_cases import state  # noqa: E402
from rbpf_register_lu import RegisterLUScorer  # noqa: E402
from test_rbpf_native_staging import exact  # noqa: E402
from test_rbpf_fused_staging import (  # noqa: E402,F401
    memo_trial,
    test_all_candidate_values_order_priors_callbacks_and_counters,
    test_custom_keys_and_partial_exception_order,
    test_original_python_prior_arithmetic_bits,
    test_callback_appended_candidates_preserve_iteration,
    test_callback_mutation_invalidates_or_bypasses_aliases,
    test_call_local_native_references_are_released,
)


@pytest.fixture(scope="module")
def paths():
    names = ("DLL", "STAGING_EXT", "CANONICAL_EXT", "PARENT_EXT", "MASS_EXT")
    result = {name: os.environ.get("GNSS_GPU_RBPF_COMMON_" + name) for name in names}
    if not all(result.values()):
        pytest.skip("Explicit independent common extension and frozen numeric builds required")
    return result


@pytest.fixture(scope="module")
def native(paths):
    spec = importlib.util.spec_from_file_location("_rbpf_native_staging_probe", paths["STAGING_EXT"])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def module(native):
    return SimpleNamespace(stage=native.stage, stage_fused=native.stage_common)


@pytest.fixture(scope="module")
def scorer(paths):
    with RegisterLUScorer(paths["DLL"]) as owned:
        yield owned


def options(paths):
    return dict(extension=paths["CANONICAL_EXT"], parent_extension=paths["PARENT_EXT"], mass_extension=paths["MASS_EXT"])


def legacy(metadata):
    return {key: value for key, value in metadata.items() if key not in ("native_staging", "common_staging")}


def test_original_controls_and_grouped_path_exercised(native, module):
    original, _ = memo_trial(module, "original")
    off, _ = memo_trial(SimpleNamespace(stage=native.stage, stage_fused=native.stage_fused), "on")
    on, stats = memo_trial(module, "on")
    exact(original, off)
    exact(original, on)
    assert stats["replayed_particles"] > 0 and stats["replayed_moves"] > 0


@pytest.mark.parametrize("backend", ["scalar", "native_cpu", "gpu"])
@pytest.mark.parametrize("cluster", [0, 1])
@pytest.mark.parametrize("seed", [20260710, 20260711])
def test_full_pf_states_rng_feedback_and_all_legacy_counts(paths, scorer, backend, cluster, seed):
    base = make_feedback_case(64, 6, rows=64, groups=4, seed=seed, feedback=True, cluster=bool(cluster))
    cases = [deepcopy(base) for _ in range(3)]
    for case in cases:
        _bind_fixture_callbacks(case[1])
    controllers = [
        NativeFeedbackMassStep(cases[0][0], scorer, backend, **options(paths)),
        CommonStagingStep(cases[1][0], scorer, backend, staging_extension=paths["STAGING_EXT"], common=False, **options(paths)),
        CommonStagingStep(cases[2][0], scorer, backend, staging_extension=paths["STAGING_EXT"], common=True, **options(paths)),
    ]
    for case, controller in zip(cases, controllers):
        case[0].step = controller
    replayed = 0
    for _ in range(4):
        snapshots = []
        for (runtime, rtk, context, *_), controller in zip(cases, controllers):
            output = controller(rtk, context)
            snapshots.append((state(runtime, rtk, output), deepcopy(legacy(controller.last))))
        exact(snapshots[0], snapshots[1])
        exact(snapshots[0], snapshots[2])
        details = controllers[2].last.get("common_staging")
        if details and details["enabled"]:
            replayed += details["replayed_moves"]
        for runtime, rtk, *_ in cases:
            rtk.epoch2 += 1
    assert replayed > 0
