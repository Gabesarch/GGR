# GGR: Learning Grounded Reasoning with Human Gaze

Code for **GGR (Gaze-Guided Reasoner)** and the **No-Gaze** control, using Qwen3-VL-8B-Instruct.

## Evaluation

```bash
bash scripts/setup_env.sh eval
source .venv-eval/bin/activate
hf auth login

# Quick check.
bash scripts/eval.sh --condition ggr-sft --tasks chartqa --limit 5 --repeats 1 --output outputs/check

# Full evaluation.
bash scripts/eval.sh --condition ggr-rl --output outputs/ggr-rl
```

Conditions: `ggr-sft`, `ggr-rl`, `no-gaze-sft`, `no-gaze-rl`. Models and datasets download from Hugging Face.
Evaluation covers **ChartQA-Pro, ChartQA, CountBenchQA, CountQA, MME RealWorld Lite, Visual Probe**, their mean accuracy, and **gaze alignment** in bits/fixation. Results and samples are saved under the output directory. See [evaluation](docs/EVALUATION.md) for GPU requirements and options.

## Setup

On Linux with NVIDIA GPUs, a CUDA 12.8-compatible driver, and the CUDA development toolkit:

```bash
bash scripts/setup_env.sh
```

This creates `.venv-eval`, `.venv-sft`, and `.venv-rl` with pinned dependencies. Run commands from the repository root. Hugging Face access uses `hf auth login` or `HF_TOKEN`.

## Download

```bash
source .venv-sft/bin/activate
python scripts/prepare_sft_data.py --condition all --register
python scripts/prepare_rl_data.py
python scripts/download_models.py --condition ggr-sft --condition ggr-rl
```

The SFT registry starts empty. `prepare_sft_data.py --register` adds the downloaded GGR and No-Gaze datasets and validation set to `ggr-sft/data/dataset_info.json`.

| Models | SFT | RL |
|---|---|---|
| GGR | [GGR-SFT](https://huggingface.co/gsarch/Qwen3-VL-8B-GGR-SFT) | [GGR-RL](https://huggingface.co/gsarch/Qwen3-VL-8B-GGR-RL) |
| No-Gaze | [No-Gaze SFT](https://huggingface.co/gsarch/Qwen3-VL-8B-Gaze-SFT) | [No-Gaze RL](https://huggingface.co/gsarch/Qwen3-VL-8B-No-Gaze-RL) |

Data: [human inputs](https://huggingface.co/datasets/gsarch/backfill-stage1-human-inputs), [SFT](https://huggingface.co/datasets/gsarch/Gaze-SFT), [RL](https://huggingface.co/datasets/gsarch/GGR-RL-Data).
See [release uploads](docs/RELEASE.md) for availability and scripts to upload missing artifacts.

## Training

```bash
bash scripts/train_sft.sh ggr
bash scripts/train_rl.sh ggr

# No-Gaze control.
bash scripts/train_sft.sh no_gaze
bash scripts/train_rl.sh no_gaze
```

Use the best checkpoint determined by validation. Launchers default to eight GPUs; see [training](docs/TRAINING.md) for checkpoint selection and overrides.

For single-stage annotation from human inputs, follow [data preparation](docs/DATA.md).

## Acknowledgements

Built on [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory), [veRL](https://github.com/verl-project/verl), and [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval). See [third-party notices](THIRD_PARTY.md).
