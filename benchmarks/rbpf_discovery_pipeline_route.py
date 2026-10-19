"""Continuous RTK arm bindings; common clock and gate protocol unchanged."""

from rbpf_async_score import AsyncScorer
from rbpf_feedback_mass import NativeFeedbackMassStep
from rbpf_move_plan_route import retaining_type, VARIANTS
from rbpf_prior_pipeline import PriorPipelineStep
from rbpf_register_lu import RegisterLUScorer


def make_scorer(dll, variant):
    if variant not in VARIANTS:
        raise ValueError("Unknown RTK arm")
    if variant in ("off", "candidate"):
        return AsyncScorer(dll, blocked=variant == "off")
    return RegisterLUScorer(dll)


def make_step(runtime, scorer, mode, variant, **extensions):
    if variant not in VARIANTS or mode != "gpu" or runtime.cfg.cluster_relin != 0:
        raise ValueError("Expected a declared GPU/c0 RTK arm")
    cls = PriorPipelineStep if variant in ("off", "candidate") else NativeFeedbackMassStep
    return retaining_type(cls)(runtime, scorer, mode, **extensions)
