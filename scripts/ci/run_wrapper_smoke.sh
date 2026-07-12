#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="python:."

# CPU-only wrapper validation for public Python APIs.  Native-backed wrapper
# suites construct CUDA objects even in validation tests, so they belong to
# the build-cuda job and cannot run on this extension-free runner.
wrapper_tests=()
for test_file in tests/test_*_wrapper.py; do
  case "$test_file" in
    tests/test_pf_wrapper.py|tests/test_pf3d_wrapper.py|\
    tests/test_pf_device_wrapper.py|tests/test_raytrace_bvh_wrapper.py|\
    tests/test_svgd_wrapper.py)
      continue
      ;;
  esac
  wrapper_tests+=("$test_file")
done

pytest -q "${wrapper_tests[@]}"
