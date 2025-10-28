#!/usr/bin/env bash
set -euo pipefail

DEVICE=${1:-}

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
ROOT_DIR=$(cd "$SCRIPT_DIR/.." && pwd)

cd "$ROOT_DIR"

if [ -d .venv ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

if [ -n "$DEVICE" ]; then
  python -m src.app --once --device "$DEVICE"
else
  python -m src.app --once
fi
