#!/usr/bin/env bash
set -euo pipefail
TD_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
TD_INTERPRETER="${TD_PYTHON:-$TD_ROOT/../KHQuant/.venv/bin/python}"
cd "$TD_ROOT"
export PYTHONPATH="$TD_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$TD_INTERPRETER" -m pytest tests -q
