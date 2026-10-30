# Context Cost Verifier (`ccv`): Measuring Whether Shrinking an LLM Prompt Really Saves Cost

*Undergraduate major project — formal report (draft in Markdown; convert into the university template).*

> **Status of this draft.** Chapters 1–7, 9 and the appendices are complete. Chapter 8 (Results) is filled from the real Kaggle runs; see [findings_report.md](findings_report.md) for the full analysis and figures. Raw material, including every challenge met during development, is in [`docs/REPORT_NOTES.md`](../docs/REPORT_NOTES.md).

---

## Abstract

Production LLM systems routinely shrink their prompts — truncating documents, dropping old conversation turns, compressing context with tools such as LLMLingua-2 — on the assumption that fewer input tokens means lower cost. That assumption ignores two effects. First, output tokens are far more expensive than input tokens: commercial price lists charge roughly five times more per output token, and on the hardware every output token is a separate sequential decode step. Second, a model that loses context can write *longer* answers and get more of them *wrong*. `ccv` (Context Cost Verifier) is an open-source command-line tool that measures this trade-off before a context change ships. It sends benchmark tasks to any OpenAI-compatible endpoint, records physical cost on the GPU (prefill time, decode time, GPU-seconds, energy), attributes every prompt token exactly to one of six context segments using the served model's own tokenizer and chat template, and runs an exact Shapley analysis over all 64 subsets of those segments. Statistical tests (exact McNemar, paired bootstrap, Benjamini–Hochberg) separate real effects from run-to-run noise. The study runs at zero cost on a free Kaggle T4 GPU with Qwen2.5-7B-Instruct (AWQ) served by vLLM.

---

## 1. Introduction

### 1.1 Background

A request to a large language model has two phases. **Prefill** processes the whole prompt in parallel and produces the first output token; its cost grows with prompt length but is highly parallel on a GPU. **Decode** then produces the answer one token at a time; every token is a full forward pass, so decode cost grows with answer length and is memory-bandwidth bound. Commercial APIs reflect this by pricing output tokens at several times the price of input tokens.

Prompts in real applications are assembled from several parts: a system persona, instructions, tool definitions, few-shot examples, retrieved documents, and conversation history. When a prompt gets long, engineers trim it. Whether that actually saves money depends on what happens to the *output*: if the model, deprived of context, rambles, hedges, asks clarifying questions or answers wrongly and needs a retry, the saving on input can be wiped out.

### 1.2 Problem statement

There is no lightweight, reproducible way for an engineer to answer, *before* shipping a context change: **which parts of my prompt are worth their tokens, and does removing them really make requests cheaper on my hardware?** Price lists give a per-token price but say nothing about how output length and accuracy respond to context, and accuracy benchmarks report quality but not cost.

### 1.3 Objectives

1. Build a pip-installable, MIT-licensed CLI that measures physical inference cost per request on real hardware against any OpenAI-compatible endpoint.
2. Attribute every prompt token exactly to a named context segment, using the served model's own tokenizer and chat template.
3. Quantify each segment's contribution to accuracy and to cost with an exact Shapley analysis over all 64 segment subsets, and compare realistic reduction strategies (truncation, LLMLingua-2 compression, document reordering) at equal token budgets.
4. Separate genuine effects from run-to-run noise with a measured noise floor and appropriate paired statistics.
5. Ship a calibrated benchmark of ~30 tasks over three workloads with machine-checkable graders, and the raw JSONL traces.

### 1.4 Scope

- One GPU type (NVIDIA T4, 16 GB, free on Kaggle), one main model (Qwen2.5-7B-Instruct, AWQ 4-bit) served by vLLM 0.29.
- Three workloads: question answering, structured extraction to JSON, and tool calling.
- Unbatched serving, so that each request's GPU time is attributable to it. Batched production serving is out of scope and discussed as a limitation.
- Llama-3.1-8B is supported by the tool and tested offline, but its GPU sweep was cut for time (§9.2).

### 1.5 Organisation of the report

Chapter 2 surveys related work. Chapter 3 analyses the existing approaches and the feasibility of the proposed system. Chapter 4 is the software requirements specification. Chapter 5 presents the design with UML, Chapter 6 the implementation and key algorithms, and Chapter 7 the testing. Chapter 8 reports results, and Chapter 9 concludes with future scope.

---

## 2. Literature survey

| Work | Finding | Relevance to `ccv` |
|---|---|---|
| Liu et al., 2023, *Lost in the Middle: How Language Models Use Long Contexts* | Accuracy is highest when the relevant document is at the start or end of the context and drops when it is in the middle | Motivates the `reorder:{first,middle,last}` ablation family |
| Shi et al., 2023, *Large Language Models Can Be Easily Distracted by Irrelevant Context* | Adding irrelevant context measurably reduces accuracy on reasoning tasks | Motivates near-miss distractor documents in every task, and the question of whether removing context can *help* |
| Hong et al. (Chroma), 2025, *Context Rot* | Performance degrades non-uniformly as input length grows, even on simple tasks | Motivates measuring rather than assuming the effect of context size |
| Pan et al., 2024, *LLMLingua-2: Data Distillation for Efficient and Faithful Task-Agnostic Prompt Compression* | A token-classification model can compress prompts 2–5× with modest accuracy loss | The learned-compression baseline compared against plain truncation at equal budgets |
| Jiang et al., 2023, *LLMLingua* | Coarse-to-fine prompt compression with a small LM | Predecessor of LLMLingua-2 |
| Shapley, 1953, *A Value for n-Person Games*; Lundberg & Lee, 2017, *SHAP* | The Shapley value is the unique attribution satisfying efficiency, symmetry, dummy and additivity | The attribution method for segment contributions; with six segments it is computed exactly over 64 subsets rather than approximated |
| Grabisch & Roubens, 1999, *An axiomatic approach to the concept of interaction among players* | The Shapley interaction index measures synergy between pairs of players | Reports which segments only help together (e.g. retrieved documents and history) |
| Kwon et al., 2023, *Efficient Memory Management for LLM Serving with PagedAttention* (vLLM) | High-throughput serving engine with continuous batching and prefix caching | The serving engine; `ccv` deliberately disables batching and prefix caching for measurement validity |
| Lin et al., 2024, *AWQ: Activation-aware Weight Quantization* | 4-bit weight quantisation with small accuracy loss | Makes a 7B model fit and run on a 16 GB T4 |
| McNemar, 1947; Efron & Tibshirani, 1993; Benjamini & Hochberg, 1995 | Paired test for proportions; bootstrap confidence intervals; false discovery rate control | The statistical toolkit (§6.5) |

**Gap.** Prior work measures either accuracy as a function of context or throughput of a serving system, but not both on the same requests, and not with exact per-segment attribution. Commercial cost calculators assume a fixed price ratio and ignore output length changes. `ccv` joins physical cost, exact token attribution and accuracy on the same runs.

---

## 3. System analysis

### 3.1 Existing approaches and their limits

1. **Price-list arithmetic.** Multiply input tokens by the input price. Ignores output length changes and accuracy loss entirely; assumes a fixed output:input price ratio (commonly 5:1) that may not match the hardware.
2. **Accuracy benchmarks with long contexts** (needle-in-a-haystack, long-context QA). Report quality but not cost, and rarely let you ablate *your* prompt's parts.
3. **Ad-hoc A/B tests in production.** Expensive, slow, confounded by traffic changes and batching, and only possible after shipping.
4. **Leave-one-out ablation.** Removes one segment at a time. Misses interactions: if two segments are each sufficient on their own, leave-one-out says neither matters.

### 3.2 Proposed system

A CLI that, for a fixed benchmark and model, (a) measures cost on the physical device per request, (b) attributes prompt tokens exactly, (c) evaluates all 64 segment subsets so Shapley values and pairwise interactions are exact, (d) compares realistic reduction strategies at equal token budgets, and (e) reports results with noise-aware statistics and Pareto frontiers of cost against accuracy.

### 3.3 Feasibility

- **Technical.** vLLM 0.29 supports AWQ on the T4's Turing architecture with fp16 and the Triton attention backend, and exposes per-request server timings. Hugging Face tokenizers expose character offsets, which makes exact attribution possible. All four models used are ungated on Hugging Face. Verified on 17 Sep 2026 (see `docs/DECISIONS.md`).
- **Economic.** Zero budget: Kaggle provides a free T4 with a weekly GPU quota (~30 hours); all software is open source. The full Qwen study needs about 6 GPU-hours.
- **Operational.** Free GPU sessions are time-limited and can die. The runner is resumable, writes every record durably, and stops itself before Kaggle's session limit, so no completed work is ever lost.

---

## 4. Software requirements specification

### 4.1 Purpose and users

`ccv` serves ML and platform engineers who own a prompt and want to know whether a context change saves cost, and researchers studying context effects. Users interact through a command line.

### 4.2 Functional requirements

| ID | Requirement |
|---|---|
| FR1 | Run named suites (`calibrate`, `noise`, `shapley`, `ablations`) of benchmark tasks against an OpenAI-compatible chat endpoint |
| FR2 | Resume an interrupted run without repeating finished requests or losing completed records |
| FR3 | Record per request: prompt and output tokens, client and server TTFT, decode time, GPU-seconds, GPU utilisation, peak memory, energy, finish reason, output text and tool calls |
| FR4 | Attribute every prompt token exactly to one of `system, instructions, tools, examples, retrieved, history, query, template`, and flag whether the local count matches the server's |
| FR5 | Generate context variants: all 64 segment subsets; truncation at 0.8/0.5/0.2; LLMLingua-2 compression at the same rates; gold-document reordering to first/middle/last |
| FR6 | Precompute and cache LLMLingua-2 compressions so GPU runs never need the compression library |
| FR7 | Grade outputs for three workloads (QA, JSON extraction, tool call) at analysis time |
| FR8 | Compute the noise floor, exact Shapley values and pairwise interactions, paired significance tests, output-expansion statistics, per-token cost fits and Pareto frontiers |
| FR9 | Produce CSV tables, a Markdown summary and PNG figures |
| FR10 | Start the vLLM server with measurement-safe flags from a single command |
| FR11 | Validate all tasks and estimate run time offline (`--dry-run`) without a GPU or server |
| FR12 | Report a per-task calibration table (passes with full context, fails with query only) |

### 4.3 Non-functional requirements

| ID | Requirement | How it is met |
|---|---|---|
| NFR1 Reproducibility | Same inputs give the same requests and comparable measurements | Greedy decoding, fixed seed, pinned template date, task hashes in every record, config hashes |
| NFR2 Fault tolerance | A killed session loses at most the in-flight request | Append + flush + `fsync` per record; resume repairs a half-written last line; `--max-minutes` |
| NFR3 Measurement validity | GPU time per request is attributable to that request | Unbatched server (`--max-num-seqs 1`), prefix caching off, warm-up, shuffled order |
| NFR4 Portability | Works with any OpenAI-compatible endpoint; authoring and analysis need no GPU | vLLM is a separate process reached over HTTP; analysis is pure Python |
| NFR5 Low overhead | Measurement must not perturb the GPU | NVML sampling every 100 ms in a background thread; streaming client |
| NFR6 Maintainability | Small enough to read in one sitting | Seven modules, ~1,450 lines; stdlib `argparse` |
| NFR7 Accessibility of results | Figures readable by colour-blind readers | Colour-blind-validated palette; every figure has a CSV table view |

### 4.4 Hardware and software requirements

| | Development / analysis | Measurement |
|---|---|---|
| Hardware | Any PC; no GPU needed (developed on Windows 11, RTX 3050 4 GB) | NVIDIA GPU with ≥ 16 GB (Kaggle T4) |
| OS | Windows, Linux or macOS | Linux |
| Software | Python ≥ 3.10; `httpx`, `pyyaml`, `transformers`, `jinja2`, `numpy`, `scipy`, `pandas`, `matplotlib` | vLLM 0.29, `nvidia-ml-py`; `llmlingua` only for `ccv compress` |

---

## 5. System design

### 5.1 Architecture

```
benchmark/tasks/*.yaml ─► tasks.py (request for a config) ─► attribution.py (exact tokens per segment)
                              ▲ ablations.py (64 subsets, truncate, reorder, LLMLingua-2 cache)
runner.py ── streamed HTTP ──► vLLM server on a Kaggle T4 (or any OpenAI-compatible API)
   │  + NVML sampler thread (utilisation, memory, energy)
   ▼
results/<model>.jsonl  (append + fsync, resumable)
   ▼
analysis.py (grade, Shapley, statistics) ─► report.py (figures) ─► analysis/*.csv, summary.md, *.png
```

### 5.2 UML diagrams

Sources are in [`docs/uml/`](../docs/uml) (render with the PlantUML extension for VS Code, or `java -jar plantuml.jar docs/uml/*.puml`):

| Diagram | File | Shows |
|---|---|---|
| Use case | `use_case.puml` | LLM app developer and researcher actors, the vLLM server as an external system; author tasks, start a server, run suites (noise, Shapley, ablations), resume, attribute, analyse |
| Class | `class.puml` | Module-level design (Python modules shown as classes): `Task`, `Grader`, `Config`, `Suite`, `Prompter`, runner and analysis |
| Sequence | `sequence.puml` | One `ccv run`: CLI → runner → variant → layout → `Prompter` → NVML sampler → vLLM server → `results.jsonl` |
| Activity | `activity.puml` | The resumable run loop: load tasks, skip finished runs, warm up, budget check, build, stream, append |
| Deployment | `deployment.puml` | Team laptop (authoring, LLMLingua-2, analysis) and the Kaggle VM with the T4 running vLLM and the runner over HTTP |

### 5.3 Data design

**Task file (YAML).** One file per task under `benchmark/tasks/<workload>/`. Fields: `id`, `workload`, `segments` (`system`, `instructions`, `tools`, `examples`, `retrieved` as a list of `{title, text}`, `history` as alternating user/assistant turns), `query`, `gold_doc` (index of the answer-bearing document), and `grader` (`qa`: `answers`, `distractors`; `json`: `expected`, `strict`; `tool_call`: `name`, `arguments`). Shared segments are referenced as `"@kestrel_persona"` and `"@kestrel_tools"`.

**Result record (one JSON line per request).** Identity (`run_id`, `request_hash`, `model`, `suite`, `task_id`, `task_hash`, `workload`, `config_id`, `repeat`), outcome (`status`, `error`, `output`, `tool_calls`, `finish_reason`), tokens (`prefill_tokens`, `decode_tokens`, `attribution` per segment, `attribution_ok`), timing (`prefill_ms`, `decode_ms`, `queue_ms`, `client_ttft_ms`, `client_e2e_ms`, `gpu_seconds`, `decode_tok_s`, `server_metrics`), GPU (`energy_j`, `gpu_util_mean`, `gpu_mem_max_mb`), variant details (`compression`, achieved kept fraction) and `env` (server version, GPU name, driver). A sample record is in Appendix B.

**Configuration identifiers.** `subset:<6 bits>` in the fixed segment order `system, instructions, tools, examples, retrieved, history` (bit order = mask order), `truncate:<rate>`, `lingua2:<rate>`, `reorder:<position>`.

---

## 6. Implementation

### 6.1 Modules

| Module | Lines | Responsibility |
|---|---|---|
| `tasks.py` | 171 | Load and validate tasks, lay segments out as chat messages, grade outputs |
| `attribution.py` | 84 | Exact per-segment token counts with the served model's tokenizer and template |
| `ablations.py` | 114 | Context variants, suites, LLMLingua-2 cache |
| `runner.py` | 324 | Streaming client, NVML sampler, durable resumable run loop, dry run |
| `analysis.py` | 360 | Grading, noise floor, exact Shapley, statistics, cost models, Pareto, calibration |
| `report.py` | 269 | Ten PNG figures |
| `cli.py` | 123 | `ccv serve | run | attribute | compress | analyze` |

### 6.2 Prompt layout

The system message holds `system`, then `## Instructions`, then `## Examples`; `history` becomes prior user/assistant turns; the final user message holds `## Documents` (numbered) and `## Question`; `tools` is passed as the request's `tools` parameter. Every piece carries its own header and separator, so removing a segment removes exactly its characters and nothing else.

### 6.3 Exact token attribution

1. Render the chat template once with every segment replaced by a unique placeholder string.
2. Substitute the real text back, recording the character span of each segment, and assert that the result equals a direct render of the real messages (this catches templates that rewrite content).
3. Tokenize with `add_special_tokens=False` and offset mapping, exactly as vLLM does. Each token belongs to the segment whose span contains its *start* offset; tokens outside every span are `template` overhead.
4. `tools` is charged the marginal cost of adding the tools parameter (render with and without), which includes the template's tool preamble.
5. `attribution_ok` records whether the local total equals the server's `usage.prompt_tokens`.

Making the local prompt byte-identical to the server's needed three fixes: normalising tool dictionaries to vLLM's exact serialised shape, pinning the date that Llama's template writes into every prompt, and shipping vLLM's Llama tool template inside the package.

### 6.4 Exact Shapley values

With six segments there are 2⁶ = 64 coalitions, all of which are measured, so the Shapley value is exact rather than sampled:

φᵢ = Σ_{S ⊆ N∖{i}} |S|! (n − |S| − 1)! / n! · [v(S ∪ {i}) − v(S)]

where v(S) is the mean metric (pass rate, output tokens, GPU-seconds, energy or price-list cost) over tasks when only segments S and the query are kept. Efficiency (Σφᵢ = v(N) − v(∅)) holds by construction and is asserted in the tests. The pairwise Shapley interaction index

Iᵢⱼ = Σ_{S ⊆ N∖{i,j}} |S|! (n − |S| − 2)! / (n − 1)! · [v(S ∪ {i,j}) − v(S ∪ {i}) − v(S ∪ {j}) + v(S)]

identifies segments that only help together. Leave-one-out values are reported alongside for contrast.

### 6.5 Statistics

- **Noise floor** per task from 30 identical repeats: standard deviation, the 95th percentile of repeat-to-repeat differences, identical-output rate, pass flip rate and timing coefficients of variation. Effects smaller than the noise floor are labelled "within noise".
- **Accuracy:** exact McNemar test on discordant pairs (`scipy.stats.binomtest`).
- **Costs:** paired bootstrap over tasks (10,000 resamples), with a null-centred p-value and +1 smoothing so it is never exactly zero.
- **Multiplicity:** Benjamini–Hochberg per model and workload.
- **Output expansion:** ratio of output tokens to the baseline; median, p90, maximum, shares above 1.5× and 2×, the capped-run rate, and how often a context reduction raised total cost.
- **Cost models:** physical (GPU-seconds, energy) and price-list (`input + 5 × output`). A linear fit of prefill time per input token and decode time per output token gives the measured decode:prefill cost ratio.

### 6.6 Resumable runner

`run_id = sha256(model, suite, task_id, task_hash, config_id, repeat, sampling)`. On start the runner reads the JSONL, tolerates a half-written final line, and skips IDs already recorded as `ok`; error records are retried. Suites run in order, requests are shuffled within a suite, and three unrecorded warm-up requests run first. Every record is appended, flushed and `fsync`ed. `--max-minutes` stops new requests before the GPU session limit.

### 6.7 Development challenges

The most instructive problems (full list in `docs/REPORT_NOTES.md`):
- Two suites measuring the same baseline produced identical run IDs, so resume silently skipped one of them; the suite was added to the hash.
- The first benchmark draft made the conversation history necessary in 39 of 45 tasks, which would have made `history` dominate every Shapley result; 24 queries were rewritten so only 16 tasks need history.
- A stale compression cache crashed a whole GPU session at the first compressed configuration; missing cache entries are now error records that are retried later.
- A review of the 45 drafted tasks before any GPU time found 33 defects, including contradictions between tasks, answers a correct reply would have failed, and gold identifiers leaked through example values in the shared tool catalogue.

---

## 7. Testing

### 7.1 Strategy

- **Automated tests** (`tests/test_ccv.py`, 12 tests, pytest, ~2 minutes, no GPU): mathematical identities, statistics, graders, every task's own gold answer, variants, server flags, exact attribution on two real tokenizers, and a full end-to-end pipeline against a fake vLLM-style HTTP server that renders prompts with the real tokenizer.
- **Manual CLI tests:** error handling and user-facing commands.
- **Offline validation:** `--dry-run` builds and attributes all 4,635 requests without a server.
- **Live validation:** sanity checks on real runs (`attribution_ok`, server metrics, energy, no duplicate IDs).

### 7.2 Automated test cases

| ID | Test | Input | Expected | Actual | Status |
|---|---|---|---|---|---|
| T1 | `test_shapley_efficiency_and_symmetry` | Random 64-coalition game; symmetric game v(S)=|S|² | Σφ = v(N) − v(∅); all φ equal for the symmetric game | As expected | Pass |
| T2 | `test_shapley_pure_pair_and_additive_games` | Pure pair game; additive game with weights 3,1,4,1,5,9 | φ = (0.5, 0.5, 0, …), interaction I₀₁ = 1; additive: φ = weights, interactions 0, leave-one-out = weights | As expected | Pass |
| T3 | `test_statistics` | Discordant pairs; all-zero and positive differences; Pareto points | McNemar p = 2·0.5¹⁰; bootstrap CI excludes 0 for positive diffs, p = 1 for zeros; correct frontier | As expected | Pass |
| T4 | `test_graders` | Correct, near-miss, hedged, fenced-JSON, extra-key, prose tool-call outputs | Accept correct (incl. JSON in code fences, args as JSON strings); reject `112.5 kg` for `12.5 kg`, hedging, extra keys under strict, prose tool calls | As expected | Pass |
| T5 | `test_every_task_grades_its_own_gold_answer` | All 45 tasks | Each grader accepts its own gold answer and rejects every listed distractor | As expected (caught 2 faulty tasks when first added) | Pass |
| T6 | `test_variants` | qa-01 under subset, reorder, LLMLingua-2 and truncate configs | Correct segments kept; gold doc at positions 0/2/4; missing cache raises a clear error; truncation within budget; original task untouched | As expected | Pass |
| T7 | `test_variants_need_tokenizer_for_truncation` | `truncate:0.5` without a tokenizer | `ValueError` mentioning tokenizer | As expected | Pass |
| T8 | `test_serve_command` | `serve_command("llama3.1-8b", …)` | Unbatched, prefix cache off, per-request metrics, fp16, Llama template and parser, extra args passed through | As expected | Pass |
| T9 | `test_attribution_is_exact[qwen2.5-0.5b]` | All 45 tasks; 6 masks on one task per workload | Segment counts sum to the total; total equals a direct render; a segment has tokens iff its bit is set | As expected | Pass |
| T10 | `test_attribution_is_exact[llama3.1-8b]` | Same, Llama-3.1 tokenizer and vendored template | Same | As expected | Pass |
| T11 | `test_pipeline_end_to_end` | 3 tasks, all four suites against a fake server; run killed after 7 requests with a half-written line, then resumed, then rerun | 3 × 107 records, no duplicates, all `attribution_ok`; calibration keeps all 3 tasks; Shapley recovers the planted structure; cost fit recovers 0.2 ms/prefill token and 25 ms/decode token; ≥ 9 figures written | As expected | Pass |
| T12 | `test_presets_have_templates_on_disk` | Model presets | Every referenced template file exists in the package | As expected | Pass |

### 7.3 Manual test cases (run 30 Oct 2026)

| ID | Description | Input | Expected | Actual | Status |
|---|---|---|---|---|---|
| M1 | Dead endpoint | `ccv run … --endpoint http://127.0.0.1:9/v1 --limit 2` | No crash; each request written as an error record for retry | Warning "could not list served models; continuing"; two `ConnectError` error records written; exit code 0 | Pass |
| M2 | Unknown task | `ccv attribute zz-99 --model qwen2.5-7b` | Clear message | `no task 'zz-99' in benchmark` | Pass |
| M3 | Empty results file | `ccv analyze empty.jsonl` | Clear message, no traceback | `no records found` | Pass |
| M4 | Invalid suite name | `ccv run --suite bogus …` | Argument error listing valid suites | `invalid choice: 'bogus' (choose from calibrate, noise, shapley, ablations)` | Pass |
| M5 | Server flags | `ccv serve --model qwen2.5-7b --print-only` | Unbatched, no prefix cache, metrics, fp16, hermes parser | Exact command printed with all flags | Pass |
| M6 | Attribution of a variant | `ccv attribute qa-01 --model qwen2.5-7b --config truncate:0.5` | Per-segment table that sums to the total | 1,573 tokens; tools 994 (63.2%), retrieved 255, system 148 … | Pass |
| M7 | Version | `ccv --version` | Version string | `ccv 0.1.0` | Pass |
| M8 | Offline validation | `ccv run --suite noise shapley ablations --dry-run` | All tasks valid; counts and time estimate | 4,635 runs; mean prompt 1,777 tokens (noise), max 2,365; 135 awaiting `ccv compress` | Pass |
| M9 | Live sanity checks | Kaggle run output | `attribution_ok`, server metrics and energy on every record | *See §8.1* | — |

---

## 8. Results

*Filled from the real runs; see [findings_report.md](findings_report.md) for full tables and figures.*

### 8.1 Live validation

_Pending the Kaggle runs._

### 8.2 Headline findings

_Pending the Kaggle runs._

---

## 9. Conclusion and future scope

### 9.1 Conclusion

_Written after the results._

### 9.2 Future scope

- **A second model.** Run the full sweep on Llama-3.1-8B (the tool, template and tests already support it) to check whether segment values transfer across model families.
- **Batched serving.** Measure cost under continuous batching with a fixed concurrency, attributing GPU time by token share, to bring the numbers closer to production.
- **More segments with sampled Shapley.** Beyond six segments, exact enumeration becomes expensive; permutation sampling or KernelSHAP would scale it.
- **Sampling-based decoding.** Repeat the study with temperature > 0 and more repeats.
- **Newer GPUs and attention kernels.** FlashAttention on Ampere or later changes the prefill:decode cost ratio.
- **Continuous integration** running the offline test suite on every commit.

---

## 10. References

1. N. F. Liu et al., "Lost in the Middle: How Language Models Use Long Contexts," *TACL*, 2024.
2. F. Shi et al., "Large Language Models Can Be Easily Distracted by Irrelevant Context," *ICML*, 2023.
3. K. Hong, A. Troynikov, J. Huber, "Context Rot: How Increasing Input Tokens Impacts LLM Performance," Chroma technical report, 2025.
4. Z. Pan et al., "LLMLingua-2: Data Distillation for Efficient and Faithful Task-Agnostic Prompt Compression," *Findings of ACL*, 2024.
5. H. Jiang et al., "LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models," *EMNLP*, 2023.
6. L. S. Shapley, "A Value for n-Person Games," in *Contributions to the Theory of Games II*, 1953.
7. S. M. Lundberg, S.-I. Lee, "A Unified Approach to Interpreting Model Predictions," *NeurIPS*, 2017.
8. M. Grabisch, M. Roubens, "An axiomatic approach to the concept of interaction among players in cooperative games," *Int. J. Game Theory*, 1999.
9. W. Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention," *SOSP*, 2023.
10. J. Lin et al., "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration," *MLSys*, 2024.
11. Q. McNemar, "Note on the sampling error of the difference between correlated proportions or percentages," *Psychometrika*, 1947.
12. B. Efron, R. J. Tibshirani, *An Introduction to the Bootstrap*, Chapman & Hall, 1993.
13. Y. Benjamini, Y. Hochberg, "Controlling the False Discovery Rate," *J. R. Stat. Soc. B*, 1995.
14. Qwen Team, "Qwen2.5 Technical Report," arXiv:2412.15115, 2024.

---

## Appendix A — CLI usage

```bash
pip install -e ".[dev]"                     # laptop
ccv serve --model qwen2.5-7b -- --attention-backend TRITON_ATTN    # GPU box
ccv run --model qwen2.5-7b --suite noise shapley ablations --out results/qwen2.5-7b.jsonl --max-minutes 500
ccv run --model qwen2.5-7b --suite noise --out results/x.jsonl --dry-run   # offline validation
ccv attribute qa-01 --model qwen2.5-7b --config truncate:0.5
ccv compress --rates 0.8 0.5 0.2
ccv analyze results/*.jsonl --out analysis/
```

## Appendix B — Sample JSONL record

_Taken from the real Qwen2.5-7B run once available._
