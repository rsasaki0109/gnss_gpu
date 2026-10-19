"""Account for pipeline scheduling while checking original PF metadata."""

from copy import deepcopy

from rbpf_move_plan_validation import check_metadata


def check_pipeline_metadata(original, actual):
    right = deepcopy(actual)
    pipeline = right.pop("discovery_pipeline")
    prior = right.pop("prior_pipeline")
    plan = check_metadata(original, right)
    assert pipeline["discoveries"] == pipeline["complete_launches"] == pipeline["finishes"] == 1
    assert pipeline["drains"] == 0
    assert pipeline["records_after_begin"] == prior["prepared_records"] == plan["logical_registrations"]
    assert prior["prepared_particles"] == plan["particles"] and prior["literal_weight_calls"] == 0
    assert pipeline["hypotheses_submitted"] == actual["batch"]["unique"]
    return dict(move_plan=plan, discovery_pipeline=pipeline, prior_pipeline=prior)
