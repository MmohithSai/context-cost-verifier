# Runbook

Operational steps for running `ccv`. For what we are building see [PLAN.md](PLAN.md), for current status see [PROGRESS.md](PROGRESS.md), for why things are built this way see [DECISIONS.md](DECISIONS.md).

---

## 1. Laptop setup (no GPU needed)

```bash
uv venv && uv pip install -e ".[dev]"
```

Everyday commands:

```bash
uv run pytest                                              # full suite, ~60 s
ccv run --model qwen2.5-7b --suite noise shapley --out results/x.jsonl --dry-run
ccv attribute qa-01 --model qwen2.5-7b                     # per-segment token table
ccv attribute qa-01 --model qwen2.5-7b --config truncate:0.5
ccv analyze results/*.jsonl --out analysis/                # tables, summary.md, figures
ccv serve --model qwen2.5-7b --print-only                  # show the exact vLLM command
```

The first `pytest` run downloads the Qwen2.5-0.5B and Llama-3.1 tokenizers (a few MB, cached afterwards). No model weights are downloaded on the laptop.

---

## 2. Writing a benchmark task

1. Copy the closest example: [qa-01](../benchmark/tasks/qa/qa-01.yaml), [ex-01](../benchmark/tasks/extraction/ex-01.yaml) or [tc-01](../benchmark/tasks/tool_call/tc-01.yaml).
2. Keep everything inside the fictional Kestrel Robotics world, and invent new products, IDs and dates rather than reusing existing ones.
3. Fill all six segments. Reuse the shared persona and tool catalog with `"@kestrel_persona"` and `"@kestrel_tools"`.
4. Put the answer in exactly one retrieved doc and set `gold_doc` to its index (0-based). Add near-miss values to the other docs and list them as `distractors`.
5. For about a third of tasks, make one required fact appear **only** in `history`.
6. Check it:

```bash
ccv run --model qwen2.5-7b --suite noise --out results/scratch.jsonl --dry-run
```

This validates every task file offline and prints prompt sizes; aim for 1.5–3k tokens.

7. Inspect the split:

```bash
ccv attribute <task-id> --model qwen2.5-7b
```

**Checklist per task:** all six segments non-empty · one gold doc · distractors listed · fictional entities only · grader type matches the workload · history alternates user/assistant and starts with user · total 1.5–3k tokens.

---

## 3. A Kaggle GPU session

**Settings:** Accelerator **GPU T4 x2**, Internet **On** (needs a phone-verified account). The notebook uses GPU 0.

1. Open [`notebooks/ccv_kaggle.ipynb`](../notebooks/ccv_kaggle.ipynb) on Kaggle (upload it, or use *File → Import Notebook*).
2. In the first code cell set `MODEL`, `SUITES`, `LIMIT`, `SESSION_MINUTES` and `REPO`.
3. **To resume a previous run:** *Add Input → Your Work →* this notebook's previous version. The restore cell copies `results/<model>.jsonl` out of it; finished runs are then skipped.
4. **Save Version → Save & Run All (Commit).** The session runs in the background and its output is saved as a new version.
5. Read the sanity-check cell output when it finishes (§5 below).
6. Repeat until the suites are complete. `--max-minutes` stops the runner before Kaggle kills the session, so results are always saved.

**Smoke test first.** Before any real sweep: `MODEL = "qwen2.5-0.5b"`, `SUITES = "noise"`, `LIMIT = "--limit 5"`. Then the same with `qwen2.5-7b`.

**Order of the real runs:** `calibrate` (before freezing the benchmark) → `noise` → `shapley` → `ablations`. The notebook default `SUITES = "noise shapley ablations"` runs them in that order and resumes wherever it left off.

**Refreshing the compression cache** (only after `benchmark/tasks` changes): set `COMPRESS = True` for one run, then commit the updated `benchmark/compressed.json` from the notebook output. It runs in its own virtual environment before vLLM starts, so `llmlingua` cannot disturb vLLM's dependencies.

---

## 4. After a session

Download the notebook output, put `results/<model>.jsonl` next to the repo's `results/`, then:

```bash
ccv analyze results/*.jsonl --out analysis/
```

Read `analysis/summary.md` first. Results from two accounts merge by concatenation, because every record carries its own `run_id`:

```bash
cat results/qwen2.5-7b.part*.jsonl > results/qwen2.5-7b.jsonl
```

---

## 5. Checks that must pass

| Check | Where | If it fails |
|---|---|---|
| `attribution_ok` true for every record | notebook sanity cell, `runs.csv` | **Stop and fix before spending GPU hours.** See §6. |
| `server_metrics` true | same | `--enable-per-request-metrics` missing, or `stream_options.include_usage` not honoured; client timings still work but are less precise |
| `energy_j` non-null | same | NVML energy counter unsupported; drop energy from the report, GPU-seconds still stand |
| Tool calls parse on a `tc-01` request | `tool_calls` non-null | wrong `--tool-call-parser`, or the Llama template is not being used |
| Baseline passes on a calibrated task | `summary.md` accuracy | the task is not calibrated; fix or drop it |
| No duplicate `run_id` | `len(set(run_id)) == len(records)` | resume logic bug; do not merge the file |

---

## 6. Troubleshooting

**`attribution_ok` is false**
The local prompt and the server's prompt differ. In order of likelihood:
1. The server is using a different chat template than `Prompter`. For Llama, `ccv serve` passes the vendored template; anything that starts vLLM by hand must pass the same `--chat-template`.
2. vLLM changed how it serializes tools (see [DECISIONS.md](DECISIONS.md) D8) — update `_canon_tool` in `tasks.py`.
3. `chat_template_kwargs` did not reach the server, so the Llama date is not pinned (D9).
Diagnose by comparing counts: `ccv attribute <task-id> --model <model>` versus `prefill_tokens` in the record for the same task and config.

**vLLM will not start on the T4**
Check `/kaggle/working/vllm.log`. Common causes: a dtype other than fp16 (T4 has no bfloat16), an attention backend that needs compute capability ≥ 8.0 (keep `--attention-backend TRITON_ATTN`), or a vLLM version that regressed on Turing — pin the last version that worked and record it in [DECISIONS.md](DECISIONS.md).

**Out of memory when loading**
Lower `--max-model-len` (8192 is plenty for these prompts) or add `--gpu-memory-utilization 0.85`. Make sure nothing else holds the GPU, especially a leftover `ccv compress` process.

**Tool calls come back as plain text**
The parser is wrong for the model (`hermes` for Qwen2.5, `llama3_json` for Llama 3.1), or `--enable-auto-tool-choice` is missing. `ccv serve` sets both; check `ccv serve --model <key> --print-only`.

**The session died mid-sweep**
Nothing to do: rerun the notebook with the previous output attached. At most the in-flight request is lost, and a half-written last line is repaired on the next append.

**Weekly GPU quota is exhausted**
Switch to a teammate's account with a different `MODEL`, or wait for the reset. The runner's resume logic makes this free of consequence.

**`ccv compress` fails or drags in the wrong `transformers`**
Run it in the notebook's dedicated venv cell. If `llmlingua` itself is broken, fall back to calling the LLMLingua-2 token-classification model directly (D16) and keep writing the same cache format.

**`ccv analyze` says "no records found" or tables are empty**
The JSONL has no `ok` records for the current task hashes. Tasks edited after a run are dropped by design — the load step prints how many were skipped as stale. Re-run those tasks or revert the task edit.

---

## 7. Release checklist (week 8)

- [ ] `uv run pytest` clean; `pip install .` in a clean venv; `ccv --help` works.
- [ ] `git tag benchmark-v1.0` on the frozen benchmark, `v1.0` on the tool.
- [ ] `analysis/` regenerated from the final traces, figures copied into the report.
- [ ] Traces published (GitHub release or a Hugging Face dataset) — **ask the team before anything becomes public**.
- [ ] README quick start followed end to end by someone who has not run it before.
- [ ] Formal report: UML rendered to PNG, test-case table filled from the suite, results chapter using the final figures.
