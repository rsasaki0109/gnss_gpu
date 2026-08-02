#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

case "$(uname -s)" in
  MINGW*|MSYS*|CYGWIN*) export PYTHONPATH="python;." ;;
  *) export PYTHONPATH="python:." ;;
esac

# Keep the CPU smoke tier explicit and stable. These tests cover pure-Python
# loaders, experiment helpers, and visualization utilities without relying on
# compiled CUDA extensions or known-failing verification suites.
pytest -q \
  tests/test_cycle_slip.py \
  tests/test_doppler.py \
  tests/test_ephemeris.py \
  tests/test_exception_handling_policy.py \
  tests/test_fetch_plateau_subset.py \
  tests/test_fetch_urbannav_hk_subset.py \
  tests/test_fetch_urbannav_subset.py \
  tests/test_io.py \
  tests/test_lambda_ambiguity.py \
  tests/test_local_fgo.py \
  tests/test_nmea_writer.py \
  tests/test_optional_backend_contract.py \
  tests/test_ppc.py \
  tests/test_evaluation_contract.py \
  tests/test_evidence.py \
  tests/test_ddpr_profiles.py \
  tests/test_multihypothesis_navigation.py \
  tests/test_realtime_runtime.py \
  tests/test_cross_domain_validation.py \
  tests/test_ambiguity_basin_pf.py \
  tests/test_audit_ppc_basin_fgo_candidate_supply.py \
  tests/test_audit_ppc_basin_fgo_cpu_gpu_parity.py \
  tests/test_audit_ppc_basin_fgo_tracker.py \
  tests/test_audit_ppc_causal_float_selector.py \
  tests/test_audit_ppc_imu_contract.py \
  tests/test_audit_ppc_imu_fgo_health.py \
  tests/test_basin_fgo_bridge.py \
  tests/test_basin_fgo_promotion.py \
  tests/test_basin_ffbsi.py \
  tests/test_basin_imu_bridge.py \
  tests/test_build_ppc_imu_pf_fgo_evidence.py \
  tests/test_compose_ppc_safe_trajectory.py \
  tests/test_compose_ppc_safe_basin_union.py \
  tests/test_compose_ppc_imu_safe_output.py \
  tests/test_compose_ppc_causal_float_selector.py \
  tests/test_evaluate_ppc_official_score.py \
  tests/test_evaluate_ppc_official_suite.py \
  tests/test_inject_ppc_basin_fault.py \
  tests/test_inject_ppc_imu_fault.py \
  tests/test_inventory_ppc_pos_candidates.py \
  tests/test_pf_imu_preint_adapter.py \
  tests/test_ppc_imu_adapter.py \
  tests/test_ppc_score.py \
  tests/test_ppc_causal_float_selector_evidence.py \
  tests/test_run_ppc_basin_fgo_six_route.py \
  tests/test_run_ppc_basin_fgo_tracker.py \
  tests/test_run_ppc_float_candidates.py \
  tests/test_audit_ppc_safe_basin_union_cv.py \
  tests/test_export_pf_seed_pos.py \
  tests/test_phase6_ros2_soak.py \
  tests/test_release_bundle.py \
  tests/test_tokyo_candidate_supply_audit.py \
  tests/test_wp172_pf_seeded_rtk_consensus.py \
  tests/test_wp173_lambda_fix_declarations.py \
  tests/test_v030_production_promotion_audit.py \
  tests/test_v030_public_demo.py \
  tests/test_result_artifact_policy.py \
  ros2/gnss_gpu_ros/test/test_filters.py \
  ros2/gnss_gpu_ros/test/test_lifecycle_core.py \
  ros2/gnss_gpu_ros/test/test_lifecycle_node_contract.py \
  tests/test_run_demo.py \
  tests/test_sbas.py \
  tests/test_urbannav.py \
  tests/test_viz.py
