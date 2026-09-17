# Context Cost Verifier (`ccv`)

Shrinking an LLM prompt does not always make it cheaper. Output tokens are the expensive part of inference, and a model given less context often **writes longer answers** and **gets more of them wrong**. `ccv` measures that trade-off *before* a context change ships:

- **Physical accounting.** Each request records input and output tokens, prefill time (time to first token), decode time, GPU-seconds, GPU utilisation and energy. These are measured on your own hardware, not taken from a price list.
- **Exact token attribution.** The served model's own tokenizer and chat template assign every prompt token to one of six labelled segments.
- **Ablation engine.** An exact Shapley analysis over all 64 subsets of the six segments, plus truncation, LLMLingua-2 compression and gold-document reordering.
- **Statistics.** A 30-repeat noise floor, paired bootstrap CIs, exact McNemar tests, and Benjamini–Hochberg correction.
- **Fault-tolerant runner.** Works with any OpenAI-compatible endpoint. It is resumable and survives dropped free-tier GPU sessions.

It runs at zero cost on a free Kaggle T4 with `Qwen2.5-7B-Instruct-AWQ` and `Llama-3.1-8B-Instruct-AWQ` served by vLLM.

**Project documentation:** [plan](docs/PLAN.md) · [progress and next steps](docs/PROGRESS.md) · [decision log](docs/DECISIONS.md) · [runbook](docs/RUNBOOK.md) · [UML sources](docs/uml)

## Install

```bash
pip install -e ".[dev]"          # laptop: authoring, tests, analysis (no GPU needed)
pip install -e ".[gpu]"          # GPU box: adds NVML sampling (nvidia-ml-py)
pip install -e ".[compress]"     # only for `ccv compress` (LLMLingua-2)
```

The PyPI distribution is named `context-cost-verifier`; the command is `ccv`. vLLM is **not** a dependency. It runs as a separate server on Linux with an NVIDIA GPU.

## Quick start (Kaggle T4)

Open `notebooks/ccv_kaggle.ipynb` on Kaggle and use these settings:
- Accelerator: GPU T4 x2
- Internet: On

Set `MODEL`, then choose **Save & Run All**. To resume later, attach the previous version's output as an input and run again. Under the hood the notebook does this:

```bash
ccv serve --model qwen2.5-7b -- --attention-backend TRITON_ATTN &      # T4: fp16, Triton attention
ccv run --model qwen2.5-7b --suite noise shapley ablations \
        --out results/qwen2.5-7b.jsonl --max-minutes 500               # rerun the same command to resume
```

Then, on any machine:

```bash
ccv analyze results/*.jsonl --out analysis/     # CSV tables, summary.md and PNG figures
```

## Commands

| Command | What it does |
|---|---|
| `ccv serve --model KEY [-- extra vLLM args]` | Starts vLLM with measurement-safe flags: `--max-num-seqs 1` (unbatched), `--no-enable-prefix-caching`, `--enable-per-request-metrics`, fp16 and the right tool parser. Add `--print-only` to just print the command. |
| `ccv run --model KEY --suite ... --out FILE` | Runs suites against `--endpoint` (default `http://localhost:8000/v1`). Other flags: `--max-minutes`, `--limit`, `--dry-run`, `--api-key`, `--tokenizer`. |
| `ccv attribute TASK_ID --model KEY [--config ...]` | Prints the exact tokens per segment for one task and config. |
| `ccv compress [--rates 0.8 0.5 0.2]` | Caches LLMLingua-2 compressions in `benchmark/compressed.json`, so GPU runs never need llmlingua. |
| `ccv analyze RESULTS... --out DIR` | Grades outputs and computes the noise floor, Shapley values and interactions, paired tests, output expansion, the per-token cost fit and Pareto frontiers. |

**Model presets:**
- `qwen2.5-7b`: `Qwen/Qwen2.5-7B-Instruct-AWQ`, `hermes` tool parser.
- `llama3.1-8b`: `hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4`, `llama3_json` tool parser with vLLM's JSON tool template, and the template date pinned.
- `qwen2.5-0.5b`: smoke tests.

Any other served model works with `--tokenizer <hf-id>`; without a tokenizer, attribution is skipped.

**Suites:**

| Suite | Configs | Repeats per task |
|---|---|---|
| `calibrate` | full context, query only | 3, 1 |
| `noise` | full context | 30 |
| `shapley` | all 64 segment subsets | 1 |
| `ablations` | truncate / LLMLingua-2 at 0.8, 0.5, 0.2 + gold doc first / middle / last | 1 |

## Benchmark tasks

`benchmark/tasks/{qa,extraction,tool_call}/*.yaml`. Any segment value written as `"@name"` is loaded from `benchmark/shared/name.yaml`.

```yaml
id: qa-01
workload: qa                      # qa | extraction | tool_call
segments:                         # all six are required; the order is the subset-mask bit order
  system: "@kestrel_persona"
  instructions: "Reply with the value and its unit only..."
  tools: "@kestrel_tools"         # OpenAI function tools
  examples: "Q: ... A: ..."
  retrieved: [{title: ..., text: ...}]
  history: [{role: user, content: ...}, {role: assistant, content: ...}]
query: What's the maximum payload my drone can carry?
gold_doc: 2                       # index of the answer-bearing doc (used by reorder)
grader: {type: qa, answers: ["12.5 kg"], distractors: ["9 kg"]}
```

**Graders are machine-checkable:**
- `qa`: the normalized answer is present and no competing value is.
- `json`: the first JSON object matches every expected field; with `strict`, extra keys fail.
- `tool_call`: the first tool call has the right name and arguments.

Grading happens at analysis time, so fixing a grader never requires a GPU re-run.

**How the segments become a request:**
- **System message:** `system`, then `## Instructions`, then `## Examples`.
- **Chat turns:** `history`.
- **Last user message:** `## Documents`, then `## Question`.
- **`tools`:** passed as the request's `tools` parameter.

A removed segment takes its header and separator with it.

**Authoring rules**
- Use **fictional** companies, products, IDs and dates, so answers can only come from the context.
- Fill all six segments; about 1.5–3k tokens per task.
  - Instructions should carry brevity and format rules.
  - `retrieved` holds one gold doc plus near-miss distractors.
- In about a third of tasks, put one needed fact only in `history`, so answering needs both `retrieved` and `history`.
- **Calibration:**
  1. Draft about 15 tasks per workload.
  2. Run `--suite calibrate` on both models.
  3. Keep tasks whose full context passes 3/3 and whose query-only run fails, on both models.
  4. Freeze 10 per workload with a git tag.
- Run `ccv run --dry-run` to validate every task and see prompt sizes before using any GPU time.

## Result records (JSONL, one per request)

| Group | Fields |
|---|---|
| Identity | `run_id`, `request_hash` (config hash), `task_hash`, `model`, `suite`, `task_id`, `workload`, `config_id`, `repeat`, `ts` |
| Tokens | `prefill_tokens`, `decode_tokens`, `finish_reason` (`length` = hit `max_tokens`) |
| Time | `prefill_ms` (server TTFT), `decode_ms` (server generation time), `queue_ms`, `client_ttft_ms`, `client_e2e_ms`, `decode_tok_s`, `gpu_seconds`, `server_metrics` |
| GPU (NVML) | `gpu_util_mean`, `gpu_mem_max_mb`, `energy_j` (null when the endpoint is remote) |
| Attribution | `attribution` (tokens per segment, plus `query`, `template` and `total`), `attribution_ok` (local total equals the server's prompt tokens), `compression` |
| Output | `output`, `tool_calls` |
| Status | `status`, `error`, `env` (ccv and server versions, GPU, host, sampling) |

## Method notes

**Measurement protocol**
- **Unbatched and uncached.** `--max-num-seqs 1` means a request owns the GPU from scheduling to its last token, so `gpu_seconds = prefill_ms + decode_ms`. Prefix caching is off, so repeats pay the full prefill cost.
- **Greedy decoding.** `temperature=0`, `seed=0`, `max_tokens=1024`. Capped outputs are reported as censored, not dropped.
- **Warm-up and ordering.** Three unrecorded warm-up requests run first, and runs are shuffled within each suite so drift spreads across configs.

**Noise floor**
- 30 identical repeats of the full context for every task.
- Effects smaller than the 95th percentile of repeat-to-repeat differences are labelled "within noise".

**Shapley analysis**
- Shapley values φ over all 64 subsets satisfy Σφ = v(all) − v(none), so each segment's cost and accuracy credit is fair.
- Pairwise Shapley interaction indices expose redundant (negative) and synergistic (positive) segment pairs. Leave-one-out values are reported alongside for contrast.

**Cost models**
- **Physical:** GPU-seconds and energy.
- **Price list:** `prefill + 5 × decode` tokens.
- Linear fits of prefill and decode time per token give the measured decode:prefill ratio, which is compared with the 5:1 ratio used in commercial price lists.

**Scope.** Unbatched T4 timings are not batched production serving; the report states this limitation explicitly.

## Development

```bash
uv venv && uv pip install -e ".[dev]"
uv run pytest            # math, graders, variants, exact attribution, GPU-free end-to-end pipeline
```

`tests/test_ccv.py` runs the whole pipeline against a fake vLLM-style server:

`run → resume after a half-written line → analyze → figures`

It needs the Qwen2.5-0.5B and Llama-3.1 tokenizers from the Hugging Face Hub (downloaded once).

**Repository layout**

| Path | Contents |
|---|---|
| `ccv/` | cli, tasks, attribution, ablations, runner, analysis, report |
| `benchmark/` | tasks, shared segments, LLMLingua-2 cache |
| `notebooks/ccv_kaggle.ipynb` | GPU runs on Kaggle |
| `docs/uml/` | PlantUML diagrams for the project report (use-case, class, sequence, activity, deployment) |
| `results/` | raw JSONL traces |
| `analysis/` | generated tables and figures |

## License

MIT (see `LICENSE`). Bundled third-party components and model licenses are listed in `NOTICE`.

Undergraduate major project by L. Prasanna, Shashank and K. Mmohith (IV B.Tech CSE-IDDMP), under the guidance of Dr. G. Vijayakumari.
