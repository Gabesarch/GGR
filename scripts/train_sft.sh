#!/usr/bin/env bash
set -euo pipefail
GGR_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$GGR_ROOT"
condition=${1:-ggr}
if (( $# )); then shift; fi
case "$condition" in ggr|no_gaze) ;; *) echo 'Usage: scripts/train_sft.sh {ggr|no_gaze} [key=value ...]' >&2; exit 2;; esac
NUM_GPUS=${NUM_GPUS:-8}
if (( 64 % NUM_GPUS != 0 )); then echo 'NUM_GPUS must divide global batch size 64' >&2; exit 2; fi
python_bin=${SFT_PYTHON:-$GGR_ROOT/.venv-sft/bin/python}
export PYTHONPATH="$GGR_ROOT/ggr-sft/src${PYTHONPATH:+:$PYTHONPATH}"
export LLAMAFACTORY_SAVE_FULL_STATE_ONLY_LAST_CHECKPOINT=1
"$python_bin" - "$condition" <<'PY'
import json,sys
from pathlib import Path
names=json.loads(Path('ggr-sft/data/dataset_info.json').read_text())
missing={sys.argv[1],'ggr_validation'}-names.keys()
if missing:
    raise SystemExit('Register training and validation data first: python scripts/prepare_sft_data.py --help. Missing: '+', '.join(sorted(missing)))
PY
exec "$python_bin" -m torch.distributed.run --standalone --nproc_per_node="$NUM_GPUS" \
  -m llamafactory.train.ggr "ggr-sft/configs/$condition.yaml" \
  "gradient_accumulation_steps=$((64 / NUM_GPUS))" "$@"
