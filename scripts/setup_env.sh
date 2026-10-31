#!/usr/bin/env bash
# Install the separate, pinned evaluation / SFT / RL environments in one command.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

components=(eval sft rl)
if [[ $# -gt 0 ]]; then
  components=("$@")
fi
for component in "${components[@]}"; do
  case "$component" in
    eval|sft|rl) ;;
    -h|--help)
      echo 'Usage: bash scripts/setup_env.sh [eval] [sft] [rl] (default: all)'
      exit 0 ;;
    *) echo "Unknown component: $component" >&2; exit 2 ;;
  esac
done

if command -v uv >/dev/null 2>&1; then
  uv_cmd="$(command -v uv)"
else
  python3 -m venv .venv-setup
  .venv-setup/bin/python -m pip install --upgrade 'uv>=0.8.14'
  uv_cmd="$PWD/.venv-setup/bin/uv"
fi

for component in "${components[@]}"; do
  env_dir=".venv-${component}"
  if [[ ! -x "$env_dir/bin/python" ]]; then
    "$uv_cmd" venv --python 3.12 --seed "$env_dir"
  fi
  py="$env_dir/bin/python"
  if [[ "$component" == sft ]]; then
    torch_version=2.11.0
    vision_version=0.26.0
  else
    torch_version=2.10.0
    vision_version=0.25.0
  fi
  "$uv_cmd" pip install --python "$py" \
    "torch==$torch_version" "torchvision==$vision_version" "torchaudio==$torch_version" \
    --index-url https://download.pytorch.org/whl/cu128
  resolution=(--constraint "configs/constraints-${component}.txt")
  if [[ "$component" == eval || "$component" == rl ]]; then
    resolution+=(--override "configs/overrides-${component}.txt")
  fi
  "$uv_cmd" pip install --python "$py" "${resolution[@]}" -r "ggr-${component}/requirements.txt"
  if [[ "$component" == sft || "$component" == rl ]]; then
    "$uv_cmd" pip install --python "$py" packaging ninja wheel setuptools
    "$uv_cmd" pip install --python "$py" "${resolution[@]}" 'flash-attn==2.8.4' --no-build-isolation
  fi
  "$uv_cmd" pip install --python "$py" --no-deps -e "ggr-${component}"
  "$py" scripts/check_environment.py --component "$component"
done

echo 'Setup complete. Activate an environment with: source .venv-eval/bin/activate'
