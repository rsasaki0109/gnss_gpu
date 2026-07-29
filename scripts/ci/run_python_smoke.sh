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
  tests/test_release_bundle.py \
  tests/test_v030_public_demo.py \
  tests/test_result_artifact_policy.py \
  ros2/gnss_gpu_ros/test/test_filters.py \
  ros2/gnss_gpu_ros/test/test_lifecycle_core.py \
  ros2/gnss_gpu_ros/test/test_lifecycle_node_contract.py \
  tests/test_run_demo.py \
  tests/test_sbas.py \
  tests/test_urbannav.py \
  tests/test_viz.py
