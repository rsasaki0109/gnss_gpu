#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

if command -v actionlint >/dev/null 2>&1; then
  ACTIONLINT_BIN="$(command -v actionlint)"
else
  if ! command -v gh >/dev/null 2>&1; then
    echo "gh CLI is required to download actionlint" >&2
    exit 1
  fi

  TMP_DIR="$(mktemp -d)"
  trap 'rm -rf "$TMP_DIR"' EXIT

  cd "$TMP_DIR"
  case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*)
      gh release download \
        -R rhysd/actionlint \
        --pattern 'actionlint_*_windows_amd64.zip' \
        --clobber
      unzip -q actionlint_*_windows_amd64.zip
      ACTIONLINT_BIN="$TMP_DIR/actionlint.exe"
      ;;
    *)
      gh release download \
        -R rhysd/actionlint \
        --pattern 'actionlint_*_linux_amd64.tar.gz' \
        --clobber
      tar -xzf actionlint_*_linux_amd64.tar.gz
      ACTIONLINT_BIN="$TMP_DIR/actionlint"
      ;;
  esac
  cd "$ROOT_DIR"
fi

"$ACTIONLINT_BIN" -color -oneline .github/workflows/*.yml
