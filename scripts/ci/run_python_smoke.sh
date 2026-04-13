#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="python:."

# Keep the CPU smoke tier explicit and stable. These tests cover pure-Python
# loaders, experiment helpers, and visualization utilities without relying on
# compiled CUDA extensions or known-failing verification suites.
pytest -q \
  tests/test_cycle_slip.py \
  tests/test_doppler.py \
  tests/test_ephemeris.py \
  tests/test_fetch_plateau_subset.py \
  tests/test_fetch_urbannav_hk_subset.py \
  tests/test_fetch_urbannav_subset.py \
  tests/test_io.py \
  tests/test_nmea_writer.py \
  tests/test_ppc.py \
  tests/test_sbas.py \
  tests/test_urbannav.py \
  tests/test_viz.py
