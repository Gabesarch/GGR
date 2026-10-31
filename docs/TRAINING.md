# Training

Run commands from the repository root after `bash scripts/setup_env.sh`.

## SFT

The checked-in `ggr-sft/data/dataset_info.json` is empty. Download and register
the GGR-SFT or No-Gaze SFT dataset, including its validation split:

```bash
.venv-sft/bin/python scripts/prepare_sft_data.py --register
```

The dataset preparer writes the selected entries into this registry. Each entry
uses ShareGPT messages (`role` / `content`), `system`, and `images`; `media_dir`
is the repository root. Training uses the shared `prompts/ggr.txt` system prompt.

```bash
bash scripts/train_sft.sh ggr
bash scripts/train_sft.sh no_gaze
```

Both recipes start from `Qwen/Qwen3-VL-8B-Instruct`: full language-model SFT,
frozen vision tower, five epochs, batch size 64, learning rate 5e-6, cosine
schedule, 3% warmup, zero weight decay, BF16 and ZeRO-3. The image area range is
2,000,000–8,388,608 pixels and the context limit is 32,768 tokens. Validation runs
every 100 updates and checkpoints are saved each epoch. Use the best checkpoint
as determined by validation.

The launchers default to eight GPUs. `NUM_GPUS` may be any divisor of 64 for SFT;
gradient accumulation preserves the effective batch size. Additional LLaMA-Factory
`key=value` arguments pass through to the trainer, for example
`resume_from_checkpoint=outputs/sft/ggr/checkpoint-1161`.

## RL

```bash
.venv-rl/bin/python scripts/prepare_rl_data.py
bash scripts/train_rl.sh ggr
bash scripts/train_rl.sh no_gaze
```

The launchers download the released SFT models from Hugging Face. To use your
best checkpoint selected by validation:

```bash
bash scripts/train_rl.sh ggr actor_rollout_ref.model.path=outputs/sft/ggr/best
```

RL uses GSPO with GRPO advantages, eight rollouts per prompt, 256 prompts per
batch, a 128-example optimizer minibatch, and 400 updates. The learning rate warms
up for 20 updates to 2e-5 and decays linearly to 7e-6. Chart, counting, and search
receive equal sampling weights. The run uses the original correctness and format
rewards, counting's `exponential_half` curve, and the response-length penalty.
The prompt/response token limits are 16,384/14,336; image processing uses
2,000,000–10,000,000 pixels. Checkpoints and validation run every 40 updates.

The retained local runtime contains the original Qwen model patches, domain
sampler, image processing, async vLLM rollout, FSDP2 optimizer, and reward code.
The release registers only the FSDP2/vLLM single-turn training path and DAPO reward
manager. `ggr-sft/LICENSE` and `ggr-rl/LICENSE` retain their upstream licenses.

## Upload RL data

The RL dataset contains 138,651 training and 1,946 validation rows. A maintainer
can package the original JSONL files with embedded image bytes and upload them:

```bash
.venv-rl/bin/python scripts/upload_rl_dataset.py \
  --train data/source/rl/train.jsonl \
  --validation data/source/rl/validation.jsonl
# Repeat with --push to package and upload to gsarch/GGR-RL-Data.
```

The first command validates the exact source counts and hashes. Packaging retains
all prompts, answers, domain labels, reward routing, and original image bytes.
Use `--image-root` if source images have moved. Dataset creation is private by
default; `--public` explicitly requests a public repository.
