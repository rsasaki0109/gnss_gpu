"""Portable collection of frozen, explicitly provisioned RBPF experiments."""

import ast
import importlib.util
import inspect
from pathlib import Path
import sys
import textwrap

import pytest
import rbpf_campaign_source_policy as original

ROOT = original.ROOT
SNAPSHOT = original.SNAPSHOT
PUBLISHED = set(['benchmarks/audit_bvh_grouped_candidates.py', 'benchmarks/audit_bvh_grouped_directions.py', 'benchmarks/audit_bvh_lazy_groups.py', 'benchmarks/audit_bvh_tiled_geometry.py', 'benchmarks/bench_bvh_candidate_registers_compare.py', 'benchmarks/bench_bvh_group_mirrors_compare.py', 'benchmarks/bench_bvh_grouped_candidates.py', 'benchmarks/bench_bvh_grouped_directions.py', 'benchmarks/bench_bvh_lazy_groups.py', 'benchmarks/bench_bvh_los_frontier_batches_compare.py', 'benchmarks/bench_bvh_los_frontier_compare.py', 'benchmarks/bench_bvh_los_frontier_urban_compare.py', 'benchmarks/bench_bvh_medium_ray_chunks_compare.py', 'benchmarks/bench_bvh_ray_chunks_compare.py', 'benchmarks/bench_bvh_tiled_geometry.py', 'benchmarks/bench_composite_geometry_compare.py', 'benchmarks/bench_diffraction_loss_compare.py', 'benchmarks/bench_double_reflection_invariants_compare.py', 'benchmarks/bench_fgo_lapack_probe.py', 'benchmarks/bench_pf_certified_division.py', 'benchmarks/bench_pf_kf_postmerge_compare.py', 'benchmarks/bench_pf_postmerge_compare.py', 'benchmarks/bench_pf_private_weights_compare.py', 'benchmarks/bench_pf_private_weights_same_context_compare.py', 'benchmarks/bench_pf_rounding_cell.py', 'benchmarks/bench_rbpf_ambiguity_bookkeeping.py', 'benchmarks/bench_rbpf_ambiguity_bookkeeping_field.py', 'benchmarks/bench_rbpf_ambiguity_bookkeeping_route.py', 'benchmarks/bench_rbpf_ambiguity_initx.py', 'benchmarks/bench_rbpf_ambiguity_initx_checked.py', 'benchmarks/bench_rbpf_ambiguity_initx_checked_field.py', 'benchmarks/bench_rbpf_ambiguity_initx_checked_route.py', 'benchmarks/bench_rbpf_ambiguity_initx_field.py', 'benchmarks/bench_rbpf_ambiguity_initx_route.py', 'benchmarks/bench_rbpf_block_lu.py', 'benchmarks/bench_rbpf_call_profile.py', 'benchmarks/bench_rbpf_call_profile_field.py', 'benchmarks/bench_rbpf_cluster_batch.py', 'benchmarks/bench_rbpf_cluster_batch_field.py', 'benchmarks/bench_rbpf_cluster_fast_guard.py', 'benchmarks/bench_rbpf_cluster_fast_guard_field.py', 'benchmarks/bench_rbpf_cluster_fast_guard_field_checked.py', 'benchmarks/bench_rbpf_cluster_fast_guard_step.py', 'benchmarks/bench_rbpf_cluster_parameter_reuse.py', 'benchmarks/bench_rbpf_cluster_parameter_reuse_field.py', 'benchmarks/bench_rbpf_cluster_parameter_reuse_step.py', 'benchmarks/bench_rbpf_common_staging.py', 'benchmarks/bench_rbpf_compact_scratch.py', 'benchmarks/bench_rbpf_compact_scratch_field.py', 'benchmarks/bench_rbpf_compiled_moves.py', 'benchmarks/bench_rbpf_component_mass.py', 'benchmarks/bench_rbpf_component_mass_field.py', 'benchmarks/bench_rbpf_copy_work_field.py', 'benchmarks/bench_rbpf_crash_isolation.py', 'benchmarks/bench_rbpf_dd_pack_field.py', 'benchmarks/bench_rbpf_dd_pack_likelihood.py', 'benchmarks/bench_rbpf_dd_pack_runtime_field.py', 'benchmarks/bench_rbpf_dd_pack_step.py', 'benchmarks/bench_rbpf_direct_view.py', 'benchmarks/bench_rbpf_direct_view_field.py', 'benchmarks/bench_rbpf_discovery_pipeline.py', 'benchmarks/bench_rbpf_discovery_pipeline_field.py', 'benchmarks/bench_rbpf_discovery_pipeline_route.py', 'benchmarks/bench_rbpf_epoch_clocks_field.py', 'benchmarks/bench_rbpf_epoch_lockstep_field.py', 'benchmarks/bench_rbpf_epoch_memo.py', 'benchmarks/bench_rbpf_epoch_memo_field.py', 'benchmarks/bench_rbpf_extend_factor.py', 'benchmarks/bench_rbpf_extend_support.py', 'benchmarks/bench_rbpf_extend_support_field.py', 'benchmarks/bench_rbpf_extend_support_field_checked.py', 'benchmarks/bench_rbpf_feedback_mass.py', 'benchmarks/bench_rbpf_feedback_mass_field.py', 'benchmarks/bench_rbpf_fgo_gpu_feasibility.py', 'benchmarks/bench_rbpf_first_api.py', 'benchmarks/bench_rbpf_fixed_lu.py', 'benchmarks/bench_rbpf_fixed_lu_field.py', 'benchmarks/bench_rbpf_fused_staging.py', 'benchmarks/bench_rbpf_fused_staging_field.py', 'benchmarks/bench_rbpf_gc_clocks_field.py', 'benchmarks/bench_rbpf_gpu_graph.py', 'benchmarks/bench_rbpf_gpu_graph_field.py', 'benchmarks/bench_rbpf_gpu_stages.py', 'benchmarks/bench_rbpf_gpu_stream.py', 'benchmarks/bench_rbpf_gpu_stream_field.py', 'benchmarks/bench_rbpf_gpu_timeline.py', 'benchmarks/bench_rbpf_idle_score_trace.py', 'benchmarks/bench_rbpf_initial_share.py', 'benchmarks/bench_rbpf_inprocess_pair_checked_field.py', 'benchmarks/bench_rbpf_inprocess_pair_field.py', 'benchmarks/bench_rbpf_irls_identity.py', 'benchmarks/bench_rbpf_irls_identity_field.py', 'benchmarks/bench_rbpf_irls_replay.py', 'benchmarks/bench_rbpf_irls_replay_capture.py', 'benchmarks/bench_rbpf_joint_covariance.py', 'benchmarks/bench_rbpf_joint_covariance_field.py', 'benchmarks/bench_rbpf_joint_covariance_field_checked.py', 'benchmarks/bench_rbpf_joint_covariance_route.py', 'benchmarks/bench_rbpf_lifecycle_field.py', 'benchmarks/bench_rbpf_likelihood_batch.py', 'benchmarks/bench_rbpf_likelihood_transport.py', 'benchmarks/bench_rbpf_lu_phases.py', 'benchmarks/bench_rbpf_move_batch.py', 'benchmarks/bench_rbpf_move_field.py', 'benchmarks/bench_rbpf_move_key_memo.py', 'benchmarks/bench_rbpf_move_phase_profile.py', 'benchmarks/bench_rbpf_move_phase_profile_field.py', 'benchmarks/bench_rbpf_move_plan.py', 'benchmarks/bench_rbpf_move_plan_field.py', 'benchmarks/bench_rbpf_move_plan_route.py', 'benchmarks/bench_rbpf_move_reuse.py', 'benchmarks/bench_rbpf_move_reuse_field.py', 'benchmarks/bench_rbpf_native_call_trace_field.py', 'benchmarks/bench_rbpf_native_canonical.py', 'benchmarks/bench_rbpf_native_canonical_field.py', 'benchmarks/bench_rbpf_native_idle_trace.py', 'benchmarks/bench_rbpf_native_staging.py', 'benchmarks/bench_rbpf_native_staging_field.py', 'benchmarks/bench_rbpf_ordered_keys.py', 'benchmarks/bench_rbpf_ordered_keys_field.py', 'benchmarks/bench_rbpf_pair_residuals_field.py', 'benchmarks/bench_rbpf_parent_cluster_field.py', 'benchmarks/bench_rbpf_parent_score.py', 'benchmarks/bench_rbpf_parent_score_field.py', 'benchmarks/bench_rbpf_pf_boundaries.py', 'benchmarks/bench_rbpf_pf_boundaries_field.py', 'benchmarks/bench_rbpf_phase_profile.py', 'benchmarks/bench_rbpf_phase_profile_field.py', 'benchmarks/bench_rbpf_pinned_transport.py', 'benchmarks/bench_rbpf_pinned_transport_field.py', 'benchmarks/bench_rbpf_preprocess_bundle_field.py', 'benchmarks/bench_rbpf_preprocess_bundle_route.py', 'benchmarks/bench_rbpf_process_profile.py', 'benchmarks/bench_rbpf_process_profile_field.py', 'benchmarks/bench_rbpf_proposal_field.py', 'benchmarks/bench_rbpf_proposal_reuse.py', 'benchmarks/bench_rbpf_qcedit_batch.py', 'benchmarks/bench_rbpf_qcedit_batch_field.py', 'benchmarks/bench_rbpf_qcedit_batch_field_checked.py', 'benchmarks/bench_rbpf_qcedit_batch_route.py', 'benchmarks/bench_rbpf_qcedit_batch_route_checked.py', 'benchmarks/bench_rbpf_register_lu.py', 'benchmarks/bench_rbpf_register_lu_back_cache.py', 'benchmarks/bench_rbpf_register_lu_field.py', 'benchmarks/bench_rbpf_register_lu_skip_swap.py', 'benchmarks/bench_rbpf_rtk_work_profile_field.py', 'benchmarks/bench_rbpf_score_host_phases.py', 'benchmarks/bench_rbpf_score_index_memo.py', 'benchmarks/bench_rbpf_score_index_memo_field.py', 'benchmarks/bench_rbpf_shrink.py', 'benchmarks/bench_rbpf_shrink_field.py', 'benchmarks/bench_rbpf_sparse_normal.py', 'benchmarks/bench_rbpf_sparse_normal_field.py', 'benchmarks/bench_rbpf_transition_wake_field.py', 'benchmarks/bench_rbpf_versioned_staging.py', 'benchmarks/bench_rbpf_versioned_staging_field.py', 'benchmarks/bench_rbpf_view_release.py', 'benchmarks/bench_rbpf_view_release_field.py', 'benchmarks/bench_rbpf_warp_field.py', 'benchmarks/bench_rbpf_warp_likelihood.py', 'benchmarks/bench_rbpf_weight_rows.py', 'benchmarks/bench_reflection_geometry_compare.py', 'benchmarks/bench_urban_atmosphere_batch_compare.py', 'benchmarks/bvh_grouped_candidates_metrics.py', 'benchmarks/bvh_lazy_groups_metrics.py', 'benchmarks/bvh_tiled_geometry_metrics.py', 'benchmarks/check_pf_kf_peeled_mismatch.py', 'benchmarks/check_pf_magic_mod.py', 'benchmarks/probes/bvh_grouped_candidates/generate.py', 'benchmarks/probes/bvh_grouped_candidates/generate_checked.py', 'benchmarks/probes/bvh_grouped_directions/generate.py', 'benchmarks/probes/bvh_iterative_build/generate.py', 'benchmarks/probes/bvh_lazy_groups/generate.py', 'benchmarks/probes/bvh_tiled_geometry/generate.py', 'benchmarks/probes/bvh_tiled_geometry/generate_frozen.py', 'benchmarks/probes/rbpf_compiled_moves/generate.py', 'benchmarks/probes/rbpf_scalar_20261019/experiments/rbpf_fgo/gtsam_rtk_standalone.py', 'benchmarks/probes/rbpf_scalar_20261019/experiments/rbpf_fgo/rbpf_fgo.py', 'benchmarks/rbpf_ambiguity_bookkeeping.py', 'benchmarks/rbpf_ambiguity_initx.py', 'benchmarks/rbpf_ambiguity_initx_checked.py', 'benchmarks/rbpf_async_score.py', 'benchmarks/rbpf_call_profile.py', 'benchmarks/rbpf_cluster_batch.py', 'benchmarks/rbpf_cluster_fast_guard.py', 'benchmarks/rbpf_cluster_parameter_reuse.py', 'benchmarks/rbpf_common_staging.py', 'benchmarks/rbpf_compact_scratch.py', 'benchmarks/rbpf_compiled_moves.py', 'benchmarks/rbpf_component_mass.py', 'benchmarks/rbpf_copy_work.py', 'benchmarks/rbpf_dd_pack.py', 'benchmarks/rbpf_dd_pack_runtime.py', 'benchmarks/rbpf_direct_view.py', 'benchmarks/rbpf_discovery_pipeline.py', 'benchmarks/rbpf_discovery_pipeline_route.py', 'benchmarks/rbpf_discovery_pipeline_validation.py', 'benchmarks/rbpf_epoch_clocks.py', 'benchmarks/rbpf_epoch_lockstep.py', 'benchmarks/rbpf_epoch_memo.py', 'benchmarks/rbpf_extend_factor.py', 'benchmarks/rbpf_extend_support.py', 'benchmarks/rbpf_feedback_cases.py', 'benchmarks/rbpf_feedback_mass.py', 'benchmarks/rbpf_first_api.py', 'benchmarks/rbpf_fixed_lu.py', 'benchmarks/rbpf_fused_staging.py', 'benchmarks/rbpf_gc_clocks.py', 'benchmarks/rbpf_gpu_frontier.py', 'benchmarks/rbpf_gpu_graph.py', 'benchmarks/rbpf_gpu_stages.py', 'benchmarks/rbpf_gpu_stream.py', 'benchmarks/rbpf_initial_share.py', 'benchmarks/rbpf_inprocess_pair.py', 'benchmarks/rbpf_irls_identity.py', 'benchmarks/rbpf_irls_replay.py', 'benchmarks/rbpf_joint_covariance.py', 'benchmarks/rbpf_joint_covariance_cases.py', 'benchmarks/rbpf_joint_covariance_field_schema.py', 'benchmarks/rbpf_joint_covariance_sized.py', 'benchmarks/rbpf_lifecycle_validation.py', 'benchmarks/rbpf_likelihood_batch.py', 'benchmarks/rbpf_likelihood_cases.py', 'benchmarks/rbpf_likelihood_runtime_adapter.py', 'benchmarks/rbpf_lu_phases.py', 'benchmarks/rbpf_move_batch.py', 'benchmarks/rbpf_move_cases.py', 'benchmarks/rbpf_move_key_memo.py', 'benchmarks/rbpf_move_phase_profile.py', 'benchmarks/rbpf_move_plan.py', 'benchmarks/rbpf_move_plan_checked.py', 'benchmarks/rbpf_move_plan_route.py', 'benchmarks/rbpf_move_plan_validation.py', 'benchmarks/rbpf_move_reuse.py', 'benchmarks/rbpf_native_call_trace.py', 'benchmarks/rbpf_native_canonical.py', 'benchmarks/rbpf_native_staging.py', 'benchmarks/rbpf_ordered_keys.py', 'benchmarks/rbpf_pair_residuals.py', 'benchmarks/rbpf_parent_measurement.py', 'benchmarks/rbpf_parent_score.py', 'benchmarks/rbpf_pf_boundaries.py', 'benchmarks/rbpf_phase_profile.py', 'benchmarks/rbpf_pinned_transport.py', 'benchmarks/rbpf_preprocess_bundle.py', 'benchmarks/rbpf_prior_pipeline.py', 'benchmarks/rbpf_process_lifecycle.py', 'benchmarks/rbpf_proposal_reuse.py', 'benchmarks/rbpf_qcedit_batch.py', 'benchmarks/rbpf_qcedit_batch_field_schema.py', 'benchmarks/rbpf_qcedit_batch_list.py', 'benchmarks/rbpf_qcedit_batch_schema.py', 'benchmarks/rbpf_qcedit_capture_check.py', 'benchmarks/rbpf_register_lu.py', 'benchmarks/rbpf_rtk_saved_partition.py', 'benchmarks/rbpf_rtk_work_profile.py', 'benchmarks/rbpf_score_host_phases.py', 'benchmarks/rbpf_score_host_phases_v2.py', 'benchmarks/rbpf_score_index_memo.py', 'benchmarks/rbpf_shrink.py', 'benchmarks/rbpf_sparse_normal.py', 'benchmarks/rbpf_transition_wake.py', 'benchmarks/rbpf_versioned_staging.py', 'benchmarks/rbpf_view_release.py', 'benchmarks/rbpf_warp_batch.py', 'benchmarks/rbpf_weight_rows.py', 'benchmarks/rbpf_windows_debug.py', 'benchmarks/rbpf_windows_debug_checked.py', 'tests/bvh_frozen_projection_policy.py', 'tests/conftest.py', 'tests/fixtures/rbpf_cluster_batch_guards_rejected_20261017.py', 'tests/fixtures/rbpf_move_plan_rejected_20261018.py', 'tests/fixtures/rbpf_score_host_phases_rejected_20261017.py', 'tests/fixtures/rbpf_score_host_phases_v2_rejected_20261017.py', 'tests/oracles/composite_geometry_scalar.py', 'tests/oracles/double_geometry_baseline.py', 'tests/oracles/double_invariants_scalar.py', 'tests/oracles/reflection_barycentric_baseline.py', 'tests/oracles/reflection_cross_baseline.py', 'tests/oracles/reflection_geometry_scalar.py', 'tests/oracles/reflection_segment_baseline.py', 'tests/rbpf_campaign_source_policy.py', 'tests/rbpf_gc_clocks_platform.py', 'tests/test_bvh_cpu_duplicates.py', 'tests/test_bvh_frozen_projection_policy.py', 'tests/test_bvh_group_mirrors.py', 'tests/test_bvh_grouped_candidates.py', 'tests/test_bvh_grouped_candidates_projection.py', 'tests/test_bvh_grouped_candidates_sanitizer.py', 'tests/test_bvh_grouped_directions.py', 'tests/test_bvh_grouped_directions_projection.py', 'tests/test_bvh_grouped_directions_sanitizer.py', 'tests/test_bvh_iterative_build_projection.py', 'tests/test_bvh_iterative_cpu.py', 'tests/test_bvh_lazy_groups.py', 'tests/test_bvh_lazy_groups_projection.py', 'tests/test_bvh_lazy_groups_sanitizer.py', 'tests/test_bvh_los_frontier.py', 'tests/test_bvh_los_frontier_batches.py', 'tests/test_bvh_medium_ray_chunks.py', 'tests/test_bvh_ray_chunks.py', 'tests/test_bvh_tiled_geometry.py', 'tests/test_bvh_tiled_geometry_checked.py', 'tests/test_bvh_tiled_geometry_projection.py', 'tests/test_bvh_tiled_geometry_sanitizer.py', 'tests/test_composite_geometry.py', 'tests/test_double_geometry.py', 'tests/test_double_invariants.py', 'tests/test_pf_private_weights.py', 'tests/test_rbpf_ambiguity_bookkeeping.py', 'tests/test_rbpf_ambiguity_initx.py', 'tests/test_rbpf_ambiguity_initx_checked.py', 'tests/test_rbpf_block_lu.py', 'tests/test_rbpf_call_profile.py', 'tests/test_rbpf_campaign_source_policy.py', 'tests/test_rbpf_cluster_batch.py', 'tests/test_rbpf_cluster_batch_guards_checked.py', 'tests/test_rbpf_cluster_fast_guard.py', 'tests/test_rbpf_cluster_fast_guard_bindings.py', 'tests/test_rbpf_cluster_fast_guard_paths.py', 'tests/test_rbpf_cluster_parameter_reuse.py', 'tests/test_rbpf_cluster_parameter_reuse_bindings.py', 'tests/test_rbpf_cluster_parameter_reuse_paths.py', 'tests/test_rbpf_cluster_parameter_reuse_rows.py', 'tests/test_rbpf_common_staging.py', 'tests/test_rbpf_common_staging_checked.py', 'tests/test_rbpf_common_staging_guards.py', 'tests/test_rbpf_compact_scratch.py', 'tests/test_rbpf_compact_scratch_edges.py', 'tests/test_rbpf_compiled_moves.py', 'tests/test_rbpf_compiled_moves_projection.py', 'tests/test_rbpf_component_mass.py', 'tests/test_rbpf_copy_work.py', 'tests/test_rbpf_dd_pack.py', 'tests/test_rbpf_dd_pack_runtime.py', 'tests/test_rbpf_direct_view.py', 'tests/test_rbpf_direct_view_observer.py', 'tests/test_rbpf_direct_view_schedule.py', 'tests/test_rbpf_discovery_pipeline.py', 'tests/test_rbpf_discovery_pipeline_route.py', 'tests/test_rbpf_discovery_pipeline_schedule.py', 'tests/test_rbpf_epoch_clocks.py', 'tests/test_rbpf_epoch_lockstep.py', 'tests/test_rbpf_epoch_memo.py', 'tests/test_rbpf_extend_factor.py', 'tests/test_rbpf_extend_factor_pivots.py', 'tests/test_rbpf_extend_support.py', 'tests/test_rbpf_feedback_mass_campaign.py', 'tests/test_rbpf_first_api.py', 'tests/test_rbpf_fixed_lu.py', 'tests/test_rbpf_fused_staging.py', 'tests/test_rbpf_gc_clocks.py', 'tests/test_rbpf_gc_clocks_platform.py', 'tests/test_rbpf_gpu_frontier.py', 'tests/test_rbpf_gpu_graph.py', 'tests/test_rbpf_gpu_stages.py', 'tests/test_rbpf_gpu_stages_checked.py', 'tests/test_rbpf_gpu_stream.py', 'tests/test_rbpf_initial_share.py', 'tests/test_rbpf_inprocess_pair.py', 'tests/test_rbpf_inprocess_pair_checked.py', 'tests/test_rbpf_inprocess_pair_serialization.py', 'tests/test_rbpf_irls_identity.py', 'tests/test_rbpf_irls_replay.py', 'tests/test_rbpf_joint_covariance.py', 'tests/test_rbpf_joint_covariance_field_schema.py', 'tests/test_rbpf_joint_covariance_sized.py', 'tests/test_rbpf_lifecycle_validation.py', 'tests/test_rbpf_likelihood_batch_probe.py', 'tests/test_rbpf_likelihood_packed_probe.py', 'tests/test_rbpf_lu_phases.py', 'tests/test_rbpf_lu_phases_edges.py', 'tests/test_rbpf_move_batch.py', 'tests/test_rbpf_move_key_memo.py', 'tests/test_rbpf_move_phase_profile.py', 'tests/test_rbpf_move_plan_checked.py', 'tests/test_rbpf_move_plan_route.py', 'tests/test_rbpf_move_plan_schedule.py', 'tests/test_rbpf_move_reuse.py', 'tests/test_rbpf_native_call_trace.py', 'tests/test_rbpf_native_canonical.py', 'tests/test_rbpf_native_staging.py', 'tests/test_rbpf_ordered_keys.py', 'tests/test_rbpf_pair_residuals.py', 'tests/test_rbpf_parent_measurement.py', 'tests/test_rbpf_parent_score.py', 'tests/test_rbpf_pf_boundaries.py', 'tests/test_rbpf_phase_profile.py', 'tests/test_rbpf_pinned_transport.py', 'tests/test_rbpf_preprocess_bundle.py', 'tests/test_rbpf_prior_pipeline.py', 'tests/test_rbpf_process_lifecycle.py', 'tests/test_rbpf_process_profile.py', 'tests/test_rbpf_proposal_reuse.py', 'tests/test_rbpf_qcedit_batch.py', 'tests/test_rbpf_qcedit_batch_field_schema.py', 'tests/test_rbpf_qcedit_batch_list.py', 'tests/test_rbpf_qcedit_batch_schema.py', 'tests/test_rbpf_register_lu.py', 'tests/test_rbpf_register_lu_back_cache.py', 'tests/test_rbpf_register_lu_extreme_rhs.py', 'tests/test_rbpf_register_lu_skip_swap.py', 'tests/test_rbpf_rtk_saved_partition.py', 'tests/test_rbpf_rtk_work_profile.py', 'tests/test_rbpf_score_host_phases_checked.py', 'tests/test_rbpf_score_index_memo.py', 'tests/test_rbpf_shrink.py', 'tests/test_rbpf_sparse_normal.py', 'tests/test_rbpf_sparse_normal_extremes.py', 'tests/test_rbpf_transition_wake.py', 'tests/test_rbpf_versioned_staging.py', 'tests/test_rbpf_versioned_staging_slots.py', 'tests/test_rbpf_view_release.py', 'tests/test_rbpf_view_release_observer.py', 'tests/test_rbpf_warp_likelihood.py', 'tests/test_rbpf_weight_rows.py', 'tests/test_rbpf_windows_debug.py', 'tests/test_rbpf_windows_debug_checked.py', 'tests/test_rbpf_windows_debug_platform.py', 'tests/test_rbpf_windows_debug_timeout.py', 'tests/test_reflection_barycentric.py', 'tests/test_reflection_cross.py', 'tests/test_reflection_geometry.py', 'tests/test_reflection_segment.py', 'tests/test_urban_atmosphere_batch.py'])
GTSAM_MODULES = set(['test_rbpf_ambiguity_bookkeeping.py', 'test_rbpf_ambiguity_initx.py', 'test_rbpf_ambiguity_initx_checked.py', 'test_rbpf_joint_covariance.py', 'test_rbpf_joint_covariance_field_schema.py', 'test_rbpf_joint_covariance_sized.py', 'test_rbpf_preprocess_bundle.py', 'test_rbpf_qcedit_batch.py', 'test_rbpf_qcedit_batch_field_schema.py', 'test_rbpf_qcedit_batch_list.py', 'test_rbpf_qcedit_batch_schema.py'])


def pytest_configure(config):
    original.pytest_configure(config)


class OptionalGtsamModule(pytest.Module):
    def collect(self):
        if any(importlib.util.find_spec(name) is None for name in ("gtsam", "cssrlib")):
            pytest.skip("Historical GTSAM probe requires its original optional dependencies")
        return super().collect()


def pytest_pycollect_makemodule(module_path, parent):
    if module_path.name in GTSAM_MODULES:
        return OptionalGtsamModule.from_parent(parent, path=module_path)
    return None


def _published(module):
    name = getattr(module, "__file__", None)
    if not name:
        return False
    try:
        return Path(name).relative_to(ROOT).as_posix() in PUBLISHED
    except ValueError:
        return False


def route_oracle_aliases():
    paths = {ROOT / "experiments/rbpf_fgo/rbpf_fgo.py": original.SCALAR,
             ROOT / "experiments/rbpf_fgo/gtsam_rtk_standalone.py": original.JOINT}
    for module in list(sys.modules.values()):
        if not _published(module):
            continue
        for name, value in list(vars(module).items()):
            if isinstance(value, Path) and value in paths:
                setattr(module, name, paths[value])
        if Path(module.__file__).name == "test_rbpf_rtk_saved_partition.py":
            module.ROOT = SNAPSHOT


def _missing(function, seen=None, depth=0):
    seen = set() if seen is None else seen
    if not inspect.isfunction(function) or function in seen or depth > 2:
        return set()
    seen.add(function)
    module = sys.modules.get(function.__module__)
    if not _published(module):
        return set()
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    except (OSError, TypeError, IndentationError):
        return set()
    scope = dict(function.__globals__)
    def path(node):
        if isinstance(node, ast.Name):
            value = scope.get(node.id)
            return value if isinstance(value, Path) else None
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            left = path(node.left)
            if left is not None and isinstance(node.right, ast.Constant) and isinstance(node.right.value, str):
                return left / node.right.value
        return None
    missing = set()
    provided = {node.args[0].value for node in ast.walk(tree)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "setenv" and node.args
                and isinstance(node.args[0], ast.Constant)}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            value = path(node.value)
            if value is not None:
                scope[node.targets[0].id] = value
        if isinstance(node, ast.Subscript) and ast.unparse(node.value) == "os.environ" and isinstance(node.slice, ast.Constant):
            name = node.slice.value
            if isinstance(name, str) and name.startswith("GNSS_GPU_RBPF_") and name not in provided:
                if not function.__globals__.get("os", __import__("os")).environ.get(name):
                    missing.add(name)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in ("read_text", "read_bytes"):
            value = path(node.func.value)
            if value is not None and value.is_relative_to(ROOT / "dist") and not value.is_file():
                missing.add(value.relative_to(ROOT).as_posix())
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            callee = function.__globals__.get(node.func.id)
            missing.update(_missing(callee, seen, depth + 1))
    return missing


def pytest_collection_modifyitems(items):
    original.pytest_collection_modifyitems(items)
    route_oracle_aliases()
    cache = {}
    for item in items:
        functions = [getattr(item, "obj", None)]
        info = getattr(item, "_fixtureinfo", None)
        if info is not None:
            functions.extend(definition.func for definitions in info.name2fixturedefs.values() for definition in definitions)
        missing = set()
        for function in functions:
            if not inspect.isfunction(function):
                continue
            if function not in cache:
                cache[function] = _missing(function)
            missing.update(cache[function])
        if missing:
            item.add_marker(pytest.mark.skip(reason="Historical native probe needs explicit local inputs: " + ", ".join(sorted(missing))))
