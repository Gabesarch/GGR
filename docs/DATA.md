# Data

The release provides **GGR-SFT** and **No-Gaze SFT** datasets constructed with
single-stage annotation for chart understanding, counting, and visual search.
Download the accepted SFT data for training:

```bash
python scripts/prepare_sft_data.py --condition all --register
```

The SFT configurations are available in `gsarch/Gaze-SFT`.
The common 100-row validation set requires a separate upload to
`gsarch/GGR-SFT-Validation` using the command below. Until then, use
`--condition ggr` or `--condition no_gaze` to prepare training data only.
The repository's `ggr-sft/data/dataset_info.json`
starts empty. The command adds only `ggr`, `no_gaze`, and `ggr_validation` entries,
writes ShareGPT JSON under `data/sft/`, and downloads images into
`data/sft/images/`. Launch training from the repository root. Select the best
checkpoint using validation.

To publish the validation snapshot, set
`SFT_SOURCE` to the SFT snapshot root and `IMAGE_ROOT` to its image root.
Source files are `ggr/train.json`, `no_gaze/train.json`, and
`validation/validation.json`. Then run:

```bash
python scripts/upload_sft_dataset.py --source-root "$SFT_SOURCE" --condition validation
python scripts/upload_sft_dataset.py --source-root "$SFT_SOURCE" \
  --image-root "$IMAGE_ROOT" --condition validation \
  --repo-id gsarch/GGR-SFT-Validation --push
```

The first command checks the exact source hashes; the second packages and uploads
the validation dataset to `gsarch/GGR-SFT-Validation` (private by default; add `--public` to
create a public release). Original user/assistant messages, original answers,
row order, and IDs are retained. Images use portable paths. The accepted training
files are complete snapshots, while their upstream candidate generation/judging
pools were partial. The uploader records that distinction.
Use `--source validation=/path/to/validation.json` to select a source file
directly; repeat `--source CONDITION=FILE` for other selected conditions.

## Single-stage annotation

The existing `gsarch/backfill-stage1-human-inputs` dataset contains **6,411 train
trials** and **698 held-out validation trials**, with native images, questions,
source answers, and ordered human points. It is private; use `hf auth login`
with access to the dataset. The default revision is pinned in the generator.
Training inputs exclude every exact decoded RGB image appearing in validation,
regardless of question. Human review corrections and exclusions are already
applied. This reviewed input export creates a new backfill dataset; it does not
reconstruct the previously accepted teacher samples byte for byte.

```bash
source .venv-rl/bin/activate
# Start in another terminal; vLLM downloads the teacher from Hugging Face.
python backfill/serve.py --tensor-parallel-size 4
python backfill/generate.py --condition ggr --dry-run --limit 1
python backfill/generate.py --condition ggr
python backfill/generate.py --condition no_gaze
python backfill/build_sft.py --run-dir data/backfill/ggr --output data/generated/ggr
python backfill/build_sft.py --run-dir data/backfill/no_gaze --output data/generated/no_gaze
python scripts/prepare_sft_data.py --condition ggr --package data/generated/ggr --register
python scripts/prepare_sft_data.py --condition no_gaze --package data/generated/no_gaze --register
```

Choose GPU assignments and tensor parallelism for your hardware. Generation runs
only when the required model servers are available. Each trial gets
25 teacher samples with the original category prompts, 16,000-token limit,
temperature 0.95, top-p 0.95, top-k 40, and reasoning disabled. GGR includes human
points as a spatial prior; No-Gaze supplies only image and question. Neither
teacher sees the answer. Coordinates remain in normalized `[0,1000]` space;
input order, duplicates, and out-of-bounds human points are preserved.

Samples are retained when answer alignment and visual correctness pass and all
nine error flags are false. The original ground truth is retained.
SFT formatting uses the original coordinate normalization, hidden-annotation
cleanup, and old-cohort category balancing (25 samples per image/question,
maximum category ratio 1.25, counting targets 6–20 weighted 3:1 after coverage),
then appends the new cohorts. An incomplete generation/judging run cannot become a finished
SFT dataset; rerunning the same command retries failed samples.

Teacher and judge models download from Hugging Face. Both conditions use the
teacher endpoint for visual judging. Set `--base-url`,
`--judge-base-url`, `--teacher-model`, or `--judge-model` for other servers/models.
Changing model revisions can change accepted samples. See
`python backfill/generate.py --help` for worker and output options.
