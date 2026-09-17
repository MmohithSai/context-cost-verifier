# Project plan: Context Cost Verifier (`ccv`)

**Status:** approved 2026-09-17 and being executed. Live status lives in [PROGRESS.md](PROGRESS.md); the reasons behind each technical choice live in [DECISIONS.md](DECISIONS.md); day-to-day operations live in [RUNBOOK.md](RUNBOOK.md).

---

## 1. What this project is

From the abstract (`UG-Major Project (1).pdf`): shrinking an LLM prompt does not reliably make it cheaper. Output tokens dominate inference cost, so a model that loses context often **writes longer answers** and **gets more of them wrong**. `ccv` measures that trade-off *before* a context change ships, using physical hardware measurements instead of price lists.

### Deliverables promised in the abstract

| # | Deliverable | Where it lives |
|---|---|---|
| 1 | `ccv` CLI: pip-installable, MIT, resumable runner, any OpenAI-compatible endpoint | [`ccv/`](../ccv) |
| 2 | ~30 calibrated tasks over 3 workloads with machine-checkable graders | [`benchmark/`](../benchmark) |
| 3 | 10-page findings report with cost–accuracy Pareto plots | `report/` (to be written) |
| 4 | Raw JSONL traces: physical metrics, config hashes, token attributions | `results/` |

### Decisions from the team (2026-09-17)

| Decision | Value | Consequence |
|---|---|---|
| Deadline | mid-November 2026 | 8-week schedule from 21 Sep 2026 |
| GPU | Kaggle T4 (16 GB) | Resumable runner and a session time budget are mandatory |
| University documents | Formal report also required (SRS, UML, test cases) | Extra documentation track, see §8 |
| Model scope | Qwen2.5-7B required, Llama-3.1-8B is a stretch goal | Llama runs on a teammate's Kaggle account once the pipeline is proven |

**If the schedule slips, cut in this order:** (1) Llama's 64-subset sweep, keeping its noise and ablation runs; (2) the reorder family; (3) the `ccv attribute` command.

---

## 2. Architecture

```
benchmark/tasks/*.yaml → tasks.py (request for a config) → attribution.py (exact tokens per segment)
                           ↑ ablations.py (64 subsets, truncate, reorder, LLMLingua-2 cache)
runner.py ── HTTP stream ──► vLLM on Kaggle T4 (or any OpenAI-compatible API)
   │  + NVML sampler thread (GPU utilisation, memory, energy)
   ▼
results/<model>.jsonl  (append + fsync after every record, resumable)
   ▼
analysis.py (grade, Shapley, statistics) → report.py (figures) → report/ (LaTeX) + docs/
```

Every module is one file, and `ccv` never imports vLLM: the server is a separate process reached over HTTP.

---

## 3. Components as built

### 3.1 `ccv/tasks.py` — tasks, prompt layout, graders

**Six ablatable segments**, whose fixed order is also the bit order of the subset masks:

`system, instructions, tools, examples, retrieved, history`

`query` is always kept. Role headers, special tokens and any template-injected default system prompt are counted separately as `template`.

**Segments → request:**
- System message: `system`, then `## Instructions`, then `## Examples`. If all three are ablated there is no system message at all.
- `history` becomes prior user/assistant turns.
- Final user message: `## Documents` (numbered), then `## Question`.
- `tools` is passed as the request's `tools` parameter.
- Each piece carries its own header and leading separator, so removing a segment removes exactly its characters.

**Graders** run at *analysis* time, so fixing a grader never costs GPU time:
- `qa`: SQuAD-style normalization; passes when a gold answer appears and no competing value does. Punctuation becomes a space, so `12.5` never matches `125`, and matching is word-boundary aware, so `112.5 kg` does not match `12.5 kg`.
- `json`: first parseable JSON object; every expected field must match after normalization; `strict: true` rejects extra keys.
- `tool_call`: first tool call must have the right name and arguments; arguments that arrive as JSON strings are accepted (a known Llama quirk); a tool call written as prose fails.

### 3.2 `ccv/attribution.py` — exact per-segment token counts

1. Render the chat template with each segment replaced by a unique placeholder.
2. Substitute the real text back, recording each segment's character span, then assert the result equals a direct render (this catches templates that rewrite content).
3. Tokenize with `add_special_tokens=False` and offset mapping, exactly as vLLM does; each token belongs to the span containing its start offset, and everything else counts as `template`.
4. `tools` is charged the marginal cost of adding the tools parameter, which includes the template's tool preamble.
5. `attribution_ok` records whether the local total equals the server's `usage.prompt_tokens`.

Verified exact on both the Qwen2.5 and Llama-3.1 tokenizers, and against the fake server in the test suite.

### 3.3 `ccv/ablations.py` — the context variants

| `config_id` | Meaning |
|---|---|
| `subset:<6 bits>` | Keep exactly those segments. `subset:111111` is the baseline, `subset:000000` is query only. |
| `truncate:{0.8,0.5,0.2}` | Keep that fraction of tokens: head of `examples` and of each retrieved doc; `history` keeps the most recent whole turns that fit. |
| `lingua2:{0.8,0.5,0.2}` | The same three segments compressed by LLMLingua-2, read from `benchmark/compressed.json`. |
| `reorder:{first,middle,last}` | Move the answer-bearing doc to that position. |

**Suites:**

| Suite | Configs | Repeats | Runs per task |
|---|---|---|---|
| `calibrate` | full context, query only | 3, 1 | 4 |
| `noise` | full context | 30 | 30 |
| `shapley` | all 64 subsets | 1 | 64 |
| `ablations` | 3 truncate + 3 lingua2 + 3 reorder | 1 | 9 |

`truncate` and `lingua2` share rates so the report can compare them at equal token budgets.

### 3.4 `ccv/runner.py` — measurement and fault tolerance

- **Identity:** `run_id = sha256(model, suite, task_id, task_hash, config_id, repeat, sampling)`; `request_hash` is the abstract's "config hash".
- **Resume:** reads the JSONL, skips a half-written final line, and skips run IDs already recorded as `ok`. Error records are retried next launch.
- **Order:** suites run in the order given; runs are shuffled within a suite so thermal drift spreads across configs. Three unrecorded warm-up requests run first.
- **Request:** streamed `POST /chat/completions` with `stream_options.include_usage`, `temperature=0`, `seed=0`, `max_tokens=1024`.
- **Timing:** server values from vLLM's `--enable-per-request-metrics` when present, client-measured TTFT and total otherwise. `gpu_seconds = prefill_ms + decode_ms`, valid because runs are unbatched.
- **GPU:** an NVML thread samples utilisation, peak memory and energy every 100 ms; null for a remote endpoint.
- **Durability:** every record is appended, flushed and `fsync`ed.
- **Budget:** `--max-minutes` stops starting new requests so a Kaggle commit finishes and saves its output.
- **`--dry-run`:** builds every request and its attribution, then prints counts, prompt sizes and an estimated GPU time. No server needed.

### 3.5 `ccv/analysis.py` — statistics

- **Noise floor** per task, from the 30 identical repeats: SD, range, the 95th percentile of repeat-to-repeat differences, identical-output rate, pass flip rate, and timing coefficients of variation. Effects below that percentile are labelled "within noise".
- **Shapley** over all 64 subsets for pass rate, output tokens, input tokens, GPU-seconds and energy, plus pairwise interaction indices and leave-one-out values for contrast. Σφ = v(all) − v(none) holds by construction and is asserted in the tests.
- **Paired comparisons** of every config against the baseline: exact McNemar for accuracy, a 10,000-resample paired bootstrap for cost metrics, and Benjamini–Hochberg correction per model and workload.
- **Output expansion:** the output-token ratio versus baseline, with median, p90, max, the share above 1.5× and 2×, the capped-run rate, and **how often a context reduction raised total cost**.
- **Cost models:** physical (GPU-seconds, energy) and price-list (`input + 5 × output`); linear per-token fits give the measured decode:prefill ratio to compare against the commercial 5:1.
- **Pareto frontier** per model and workload under both cost models.

Outputs: one CSV per table plus `summary.md`.

### 3.6 `ccv/report.py` — figures

Ten PNG figures: Shapley bars for three metrics, interaction heatmaps, the expansion CCDF, truncation versus LLMLingua-2 at equal budget, accuracy by gold-document position, Pareto plots, and the per-token cost fit. They use a colorblind-validated palette; the CSV tables are their accessible table view.

### 3.7 `ccv/cli.py`

`ccv serve | run | attribute | compress | analyze`. `serve` is the single source of truth for server flags, so the notebook and the code can never drift apart.

---

## 4. Benchmark design

**Content rules**
- Fictional companies, products, IDs and dates only, so answers cannot come from model memory. The current fictional world is "Kestrel Robotics" (cargo drones and warehouse robots).
- Drafts may be written with LLM help but are checked by hand. The benchmark is MIT-licensed.

**Every task fills all six segments** (about 1.5–3k tokens):

| Segment | Guidance |
|---|---|
| system | 150–300 tokens, shared persona |
| instructions | 60–150 tokens of brevity and format rules, so removing them should lengthen outputs |
| tools | shared 8-tool catalog; required for tool-calling tasks, a realistic distractor elsewhere |
| examples | 150–400 tokens |
| retrieved | 4–6 docs: one gold plus near-miss distractors |
| history | 2–4 turns |

- In about a third of tasks, one needed fact lives **only** in `history`, so answering needs `retrieved` and `history` together. That is what produces non-trivial Shapley interactions.

**Calibration**
1. Draft about 15 tasks per workload.
2. Run `--suite calibrate` on both models (about 180 requests per model).
3. Keep tasks whose full context passes 3/3 on both models **and** whose query-only run fails on both.
4. Keep 10 per workload with varied lengths.
5. Freeze with git tag `benchmark-v1.0`; every record stores `task_hash`, and analysis drops records whose task has changed since.

---

## 5. Experiment protocol

1. **Smoke test:** `qwen2.5-0.5b` with `--limit 5`, then the 7B model. Check that metrics are non-null, `attribution_ok` is true and tool calls parse. Measure tokens per second and update the budget below.
2. **Calibrate** both models, then freeze the benchmark.
3. **Qwen2.5-7B:** `noise`, then `shapley`, then `ablations`, resuming across commits.
4. **Llama-3.1-8B (stretch):** the same suites on a second account.
5. **Analyze** on the laptop.

**Compute budget per model** (30 tasks): 900 noise + 1,920 Shapley + 270 ablations ≈ **3,090 requests**. At roughly 6 s each that is about 5 GPU-hours, so about 15–20 GPU-hours in total including calibration and debugging. That fits one Kaggle account's weekly quota; a second account buys wall-clock time, not quota.

If the noise floor shows outputs changing between identical runs, give `ablations` 3 repeats; `shapley` stays at 1.

---

## 6. Schedule (8 weeks, 21 Sep – 15 Nov 2026)

Live checkboxes are in [PROGRESS.md](PROGRESS.md).

| Week | Build | Kaggle runs | Documents |
|---|---|---|---|
| 1 (21 Sep) | Repo, packaging, `MODELS`, `ccv serve`, notebook | **De-risk:** both AWQ models on a T4, per-request metrics, tool calls, tokens/s | SRS draft, use-case and architecture diagrams |
| 2 (28 Sep) | tasks, attribution, tests; start ~45 draft tasks | none | Class, sequence, activity diagrams |
| 3 (5 Oct) | runner, resume test, dry run; finish draft tasks | 0.5B smoke, then calibrate both models | Data design |
| 4 (12 Oct) | ablations, compression cache; freeze `benchmark-v1.0` | Qwen: noise, then Shapley | Test-case table skeleton |
| 5 (19 Oct) | analysis on partial Qwen data | Qwen: ablations; Llama: noise, Shapley | Implementation chapter |
| 6 (26 Oct) | figures, README | Llama: ablations; re-run errors | Testing chapter |
| 7 (2 Nov) | polish, clean-venv install, tag v1.0 | buffer | **Findings report**, results chapter |
| 8 (9 Nov) | release traces, demo rehearsal | none | Assemble formal report, slides |

**Ownership:** A = runner, serve, Kaggle operations. B = benchmark tasks, graders, calibration, testing chapter. C = attribution, ablations, analysis, figures, results chapter. Everyone: SRS, UML, findings report.

---

## 7. Findings report outline (10 pages)

1. Introduction and motivation.
2. Related work: context rot, lost in the middle, Shi et al. 2023, LLMLingua-2.
3. System design: measurement, attribution, resumable runner.
4. Benchmark design and calibration.
5. Method: noise floor, ablation families, statistics.
6. Results: noise floor; Shapley cost and accuracy credit, interactions, versus leave-one-out; truncation versus LLMLingua-2; lost in the middle; output expansion tails; Pareto frontiers; measured versus 5:1 price ratio.
7. Deployment recommendations.
8. Limitations: small models, unbatched T4 versus batched production serving, 30 tasks, greedy decoding, AWQ kernels.
9. Conclusion.

## 8. Formal report outline (university)

1. Introduction.
2. Literature survey.
3. System analysis: existing system and limits, proposed system, feasibility (technical, zero-budget economic, operational).
4. **SRS:** functional requirements (run suites, resume, attribution, ablations, grading, statistics, reports); non-functional (reproducibility, fault tolerance, portability, low overhead); hardware and software requirements.
5. **Design:** architecture, UML from [`docs/uml/`](uml) (use-case, class, sequence, activity, deployment), data design (task YAML and the JSONL record schema).
6. Implementation: modules and key algorithms (exact Shapley, attribution, resumable runner).
7. **Testing:** test-case table (ID, description, input, expected, actual, status) built from `tests/test_ccv.py` plus manual CLI cases such as a dead endpoint.
8. Results: findings-report figures and CLI screenshots.
9. Conclusion and future scope.
10. References. 11. Appendix: CLI usage and a sample JSONL record.

---

## 9. Risks

| Risk | Mitigation |
|---|---|
| vLLM regresses on T4 | Pin the newest version that passes the week-1 smoke test; the server version is recorded in every record |
| `llmlingua` 0.2.2 (Apr 2024) clashes with vLLM's dependencies | Run `ccv compress` in a separate venv before vLLM starts; fallback is ~30 lines calling the LLMLingua-2 token-classification model directly |
| Local attribution disagrees with the server | `attribution_ok` flag plus tests; fix template or tool-dict handling |
| Tokenizer offset quirks | Tokens are assigned by start offset; tests assert per-segment sums; verified contiguous on both tokenizers |
| Runaway generations | `max_tokens=1024`; capped runs reported as censored, never dropped |
| Lost sessions or quota | append + fsync, resume, `--max-minutes`, restore from previous commit, second account |
| Timing validity | Unbatched, no prefix cache, warm-up, shuffled order, same GPU type; the unbatched-versus-batched limitation is stated in the report |

## 10. Verification

- `uv run pytest` — Shapley properties, statistics, graders, variants, exact attribution on both tokenizers, and a GPU-free end-to-end pipeline run including kill-and-resume.
- `ccv run --dry-run --suite shapley` — 64 requests per task with consistent attribution, no GPU needed.
- Kaggle smoke test — non-null `prefill_ms`, `decode_ms`, `energy_j`, and `attribution_ok` true.
- `ccv analyze` — every figure and table generated; Σφ identity holds on real data.
- Clean venv `pip install .` then `ccv --help`.
