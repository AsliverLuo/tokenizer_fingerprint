#!/bin/sh
set -eu

if command -v python3 >/dev/null 2>&1; then
  exec python3 run.py "$@"
fi

if command -v python >/dev/null 2>&1; then
  exec python run.py "$@"
fi

echo "python3 or python was not found in PATH" >&2
exit 127
