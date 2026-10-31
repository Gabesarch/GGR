# Evaluation

```bash
bash scripts/eval.sh --condition ggr-rl --output outputs/ggr-rl
```

Conditions: **GGR-SFT** (`ggr-sft`), **GGR-RL** (`ggr-rl`), **No-Gaze SFT** (`no-gaze-sft`), and **No-Gaze RL** (`no-gaze-rl`). Models and datasets download from Hugging Face. Benchmark dataset revisions are pinned in `ggr-eval/ggr_eval/datasets.json`. `--model REPO_OR_CHECKPOINT` evaluates another checkpoint with the shared prompt and image settings. Use `--tensor-parallel-size N` and `--judge-tensor-parallel-size N` to distribute the actor and judge across GPUs. The actor exits before the judge loads. Qwen3-32B judging needs sufficient GPU memory (roughly 64 GB for weights alone in BF16).

`--dry-run` prints the complete plan without downloading weights. `--tasks chartqa,countqa --limit 5 --repeats 1` performs a partial smoke run. Full results require all examples. `--stage score` scores completed saved generations. Use a new output directory for a new generation run.

| Benchmark | Hugging Face dataset | Split | Examples |
|---|---|---|---:|
| ChartQA Pro | `ahmed-masry/ChartQAPro` | test | 1,948 |
| ChartQA | `lmms-lab/ChartQA` | test | 2,500 |
| CountBenchQA | `vikhyatk/CountBenchQA` | test | 491 |
| CountQA | `Jayant-Sravan/CountQA` | test, expanded QA pairs | 1,528 |
| MME RealWorld Lite | `yifanzhang114/MME-RealWorld-lite-lmms-eval` | train | 1,919 |
| Visual Probe | `Mini-o3/VisualProbe_{Easy,Medium,Hard}` | validation | 141 / 268 / 106 |
| Gaze alignment | `gsarch/held_out_gaze` | train | 698 |

The three Visual Probe subsets are pooled by example count. Overall is the equal-weight mean of the six benchmark scores. Benchmark summaries report the mean and SEM of three evaluation runs. Saved files include the raw responses, per-example scores, judge decisions, dataset fingerprints, sample hashes, and `results.json`.

Generation uses temperature 1, top-p .95, top-k 20, presence penalty 1.5, a 32,768-token context, and 14,336 output tokens. Repeat seeds are `--seed + repeat_index`, starting at zero; the default three runs use seeds 1, 2, and 3, recorded in `protocol.json`. Image resizing uses `qwen-vl-utils` with an area range of 2,000,000–10,000,000 pixels. The task prompts, final-answer parsers, and scoring functions are in `ggr-eval/ggr_eval/scorers`; `source_provenance.json` records their origins. Visual Probe uses `Qwen/Qwen3-32B` with the non-thinking template and sampling settings: temperature .7, top-p .8, top-k 20, min-p 0, and 2,048 output tokens. Answer truncation uses the `Qwen/Qwen3-8B` tokenizer and retains the last 8,096 tokens.

Gaze alignment uses three responses per trial per run. It keeps valid `(x,y)` and `[x,y]` points in the 0–1000 plane, fits Gaussian bandwidth and uniform-mixture weight on the fixed 336-trial calibration split, and scores the other 362 trials. The grid is sigma `{25,35,50,70,100,140,200}` and epsilon `{.01,.03,.1,.2,.4,.7,.9}`. Results are bits per fixation averaged over trials; SEM is across scoring trials. The calibration split is reproducible from SHA-256 of `314159:doc_id`.

The release uses a compact runner around the retained scoring code. Batch scheduling and stochastic generation can change individual responses across hardware or versions. The released models use the best checkpoint determined by validation.
