#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="python:.${PYTHONPATH:+:$PYTHONPATH}"

python3 experiments/build_paper_assets.py
python3 experiments/build_githubio_summary.py
npm run site:smoke
