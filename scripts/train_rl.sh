#!/usr/bin/env bash
set -euo pipefail
GGR_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export GGR_ROOT
cd "$GGR_ROOT"
condition=${1:-ggr}
if (( $# )); then shift; fi
case "$condition" in ggr|no_gaze) ;; *) echo 'Usage: scripts/train_rl.sh {ggr|no_gaze} [Hydra overrides ...]' >&2; exit 2;; esac
NUM_GPUS=${NUM_GPUS:-8}
python_bin=${RL_PYTHON:-$GGR_ROOT/.venv-rl/bin/python}
export PYTHONPATH="$GGR_ROOT/ggr-rl${PYTHONPATH:+:$PYTHONPATH}"
for split in train validation; do
  if [[ ! -s "data/rl/$split.jsonl" ]]; then
    echo 'Prepare RL data first: python scripts/prepare_rl_data.py' >&2; exit 2
  fi
done
exec "$python_bin" -m verl.trainer.main_ppo \
  --config-path="$GGR_ROOT/ggr-rl/configs" --config-name="$condition" \
  "trainer.n_gpus_per_node=$NUM_GPUS" \
  "actor_rollout_ref.actor.fsdp_config.fsdp_size=$NUM_GPUS" "$@"
