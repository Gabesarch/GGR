# Release artifacts

All consumer commands load models and data from Hugging Face. IDs and validation-selected checkpoints are in `configs/release.json`. Check access with:

```bash
python scripts/check_hub.py
```

On October 20, 2026, all four models, the human inputs, and the SFT training configurations in `gsarch/Gaze-SFT` were available privately. The 100-row SFT validation package still requires upload. The RL dataset repository exists, but its data files have not yet been uploaded. Access to private repositories requires an authorized Hugging Face login. `configs/hub_status.json` records this check.

## Models

Supply a local Hugging Face checkpoint directory for each condition to publish. The default action validates weights and checks chat-template parity without uploading.

```bash
python scripts/upload_models.py --list
python scripts/upload_models.py --source ggr-rl=models/ggr-rl
python scripts/upload_models.py --source ggr-rl=models/ggr-rl --upload
```

Repeat `--source CONDITION=DIRECTORY` to package multiple models. `--prepare-only --output-dir outputs/model-release` stages files locally. `--namespace` defaults to `gsarch`; `--exist-ok` permits updating a private repository. Uploads include weights, tokenizer, processor, the condition prompt, and validation-selection metadata.

## SFT validation data

Set `SFT_SOURCE` to the SFT snapshot root and `IMAGE_ROOT` to the image root. The validation snapshot is `validation/validation.json`:

```bash
python scripts/upload_sft_dataset.py --source-root "$SFT_SOURCE" \
  --image-root "$IMAGE_ROOT" --condition validation \
  --repo-id gsarch/GGR-SFT-Validation --push
```

The script validates the selected snapshots and packages image assets with portable references. See [data preparation](DATA.md) for the schema and backfill commands.

## RL data

Set `RL_TRAIN` and `RL_VALIDATION` to the original RL JSONL files:

```bash
python scripts/upload_rl_dataset.py --train "$RL_TRAIN" --validation "$RL_VALIDATION"
python scripts/upload_rl_dataset.py --train "$RL_TRAIN" --validation "$RL_VALIDATION" --push
```

The uploader checks the source hashes and packages all 138,651 training rows and 1,946 validation rows with embedded images and reward-routing metadata. Uploads are private by default. Publishing an artifact publicly is a separate release step.
