#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
exec "${EVAL_PYTHON:-$ROOT/.venv-eval/bin/python}" "$ROOT/ggr-eval/evaluate.py" "$@"
