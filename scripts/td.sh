#!/usr/bin/env bash
set -euo pipefail
TD_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
TD_INTERPRETER="${TD_PYTHON:-$TD_ROOT/../KHQuant/.venv/bin/python}"
if [[ ! -x "$TD_INTERPRETER" ]]; then
  echo "Python unavailable: $TD_INTERPRETER; set TD_PYTHON to a Python 3.12 environment." >&2
  exit 2
fi
exec "$TD_INTERPRETER" "$TD_ROOT/run.py" "$@"
