#!/usr/bin/env bash
# Fetch and build the pinned RB-FGO-PF dependencies next to this directory.
#
#   bash setup_deps.sh [--python PY] [--skip-pip] [--skip-gtsam]
#
# Stages (each is skipped when its output already exists):
#   1. .venv/            Python 3.12 venv + requirements.txt (incl. the pinned
#                        cssrlib-numba fork). --python PY uses an existing
#                        interpreter instead and skips venv creation.
#   2. tc/               inuex35/tightly-coupled-gnss-imu-fgo at the pinned
#                        commit; the runtime imports gnss_fgo from tc/src.
#   3. deps/gtsam*       inuex35/gtsam at the pinned commit (it carries the
#                        DoubleDifference*Factor[Arm] factors the PyPI wheel
#                        lacks), configured with the options of the build that
#                        produced the published results, built, pip-installed.
#
# Stages 1 and 2 were exercised on 2026-10-06. Stage 3 reproduces the original
# CMake cache options (Visual Studio 17 2022, Release) but has not yet been
# run end-to-end on a clean machine.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TC_REPO=https://github.com/inuex35/tightly-coupled-gnss-imu-fgo.git
TC_COMMIT=5cbfec443216ccec1350cceb9ae95cdb11ce32b6
GTSAM_REPO=https://github.com/inuex35/gtsam.git
GTSAM_COMMIT=3c2f54c28df195cab828a97924a3c6e5546e07fa

PY=""
SKIP_PIP=0
SKIP_GTSAM=0
while [ $# -gt 0 ]; do
  case "$1" in
    --python) PY="$2"; shift 2 ;;
    --skip-pip) SKIP_PIP=1; shift ;;
    --skip-gtsam) SKIP_GTSAM=1; shift ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

if [ -z "$PY" ]; then
  if [ ! -d "$HERE/.venv" ]; then
    python3.12 -m venv "$HERE/.venv" 2>/dev/null || py -3.12 -m venv "$HERE/.venv"
  fi
  if [ -x "$HERE/.venv/Scripts/python.exe" ]; then
    PY="$HERE/.venv/Scripts/python.exe"
  else
    PY="$HERE/.venv/bin/python"
  fi
fi
echo "[setup] python: $PY ($("$PY" -c 'import sys; print(sys.version.split()[0])'))"

if [ "$SKIP_PIP" = 0 ]; then
  "$PY" -m pip install -r "$HERE/requirements.txt"
fi

clone_at() {  # clone_at <repo> <commit> <dir>
  if [ ! -d "$3/.git" ]; then
    git clone --filter=blob:none "$1" "$3"
  fi
  git -C "$3" fetch --quiet origin "$2" || true
  git -C "$3" checkout --quiet "$2"
  echo "[setup] $3 at $(git -C "$3" rev-parse HEAD)"
}

clone_at "$TC_REPO" "$TC_COMMIT" "$HERE/tc"

if [ "$SKIP_GTSAM" = 0 ]; then
  mkdir -p "$HERE/deps"
  clone_at "$GTSAM_REPO" "$GTSAM_COMMIT" "$HERE/deps/gtsam"
  cmake -S "$HERE/deps/gtsam" -B "$HERE/deps/gtsam_build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DGTSAM_BUILD_PYTHON=ON -DGTSAM_PYTHON_VERSION=3.12 \
    -DPYTHON_EXECUTABLE="$PY" \
    -DGTSAM_BUILD_TESTS=OFF -DGTSAM_BUILD_EXAMPLES_ALWAYS=OFF \
    -DGTSAM_BUILD_UNSTABLE=OFF -DGTSAM_USE_BOOST_FEATURES=OFF \
    -DGTSAM_ENABLE_BOOST_SERIALIZATION=OFF -DGTSAM_WITH_TBB=OFF \
    -DGTSAM_USE_SYSTEM_EIGEN=OFF -DGTSAM_WITH_EIGEN_MKL=OFF \
    -DGTSAM_ALLOW_DEPRECATED_SINCE_V43=ON \
    -DGTSAM_POSE3_EXPMAP=ON -DGTSAM_ROT3_EXPMAP=ON
  cmake --build "$HERE/deps/gtsam_build" --config Release --parallel
  "$PY" -m pip install "$HERE/deps/gtsam_build/python"
fi

"$PY" - <<'EOF'
import gtsam
missing = [n for n in ("DoubleDifferencePseudorangeFactorArm",
                       "DoubleDifferenceCarrierPhaseFactorArm") if not hasattr(gtsam, n)]
print("[setup] gtsam", gtsam.__file__, "custom DD factors:", "missing " + ",".join(missing) if missing else "ok")
EOF
