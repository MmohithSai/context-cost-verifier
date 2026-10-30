# Decision log

Why the project is built the way it is. **Read this before changing `attribution.py`, the server flags in `cli.py`, or anything in `runner.py` that touches timing** — several choices here are load-bearing for measurement validity, and undoing them silently invalidates results.

Format: decision · why · consequence. Dates are when the decision was made.

---

## Verified environment facts (checked 2026-09-17)

| Fact | Detail | Source |
|---|---|---|
| vLLM current release | 0.29.0, published 2026-09-09 | PyPI |
| AWQ on Turing (T4) | Supported; Marlin kernels also list Turing | vLLM quantization docs |
| T4 dtype | fp16 only, no bfloat16 | vLLM docs, T4 is compute capability 7.5 |
| Attention backend on T4 | `TRITON_ATTN`; FlashAttention (FA2) and FlashInfer both require ≥ 8.0 | vLLM attention backend table |
| Per-request metrics | `--enable-per-request-metrics` returns `metrics.{time_to_first_token_ms, generation_time_ms, queue_time_ms, mean_itl_ms, tokens_per_second}`; while streaming they arrive on the final usage chunk, which needs `stream_options.include_usage` | vLLM "Per-Request Metrics" docs |
| Tool parsers | Qwen2.5 → `hermes`; Llama 3.1 → `llama3_json` with `tool_chat_template_llama3.1_json.jinja` | vLLM tool-calling docs |
| How vLLM passes tools to a template | `tool_dicts = [tool.model_dump() for tool in request.tools]`; the serializer drops `strict` and `defer_loading` when unset but keeps `description` and `parameters` even when null | vLLM v0.29.0 source: `renderers/online_renderer.py`, `generate/base/protocol.py` |
| `chat_template_kwargs` | Accepted per request; kwargs matching template variables (such as `date_string`) are passed through | vLLM v0.29.0 source: `renderers/hf.py` |
| Models | `Qwen/Qwen2.5-7B-Instruct-AWQ` (Apache-2.0), `hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4` (Llama 3.1 license), `microsoft/llmlingua-2-xlm-roberta-large-meetingbank` (MIT), `Qwen/Qwen2.5-0.5B-Instruct` — **all ungated** | Hugging Face API |
| `llmlingua` | Last release 0.2.2, April 2024; unbounded `transformers` requirement | PyPI |
| PyPI name | `ccv` is taken; `context-cost-verifier` is free | PyPI |
| Laptop | Windows 11, Python 3.13.1, uv 0.7.5, RTX 3050 **4 GB** (too small for 7B/8B AWQ), WSL has no distro | local checks |

Installed library versions: transformers 5.17.0, scipy 1.18.1, pandas 3.0.5, numpy 2.5.3, httpx 0.28.1, matplotlib 3.11.2, jinja2 3.1.6.

---

## Measurement

**D1. Runs are unbatched and prefix caching is off** (2026-09-17)
`ccv serve` always passes `--max-num-seqs 1 --no-enable-prefix-caching`.
*Why:* a request must own the GPU for `gpu_seconds = prefill_ms + decode_ms` to mean anything, and a cached prefix would make repeated prompts (the whole noise suite) skip prefill entirely.
*Consequence:* numbers describe unbatched serving, not batched production. The report must say so; the plan lists this as a stated limitation.

**D2. `gpu_seconds = prefill_ms + decode_ms`** (2026-09-17)
*Why:* under D1 this is the wall-clock time the request occupies the device; NVML utilisation is recorded separately rather than used to weight it.
*Consequence:* if D1 is ever relaxed, this field becomes wrong and must be redefined.

**D3. Every request is streamed, and server timings win when present** (2026-09-17)
The runner always streams with `stream_options.include_usage`, measures client TTFT and total, and prefers vLLM's `metrics` values.
*Why:* streaming is the only portable way to get TTFT from an arbitrary OpenAI-compatible endpoint, while vLLM's server-side values exclude HTTP overhead. Recording both makes the two comparable and keeps the tool endpoint-agnostic.
*Consequence:* the tool-call path has to accumulate streamed deltas, which the tests cover.

**D4. Greedy decoding with `max_tokens=1024`; capped runs are censored, not dropped** (2026-09-17)
*Why:* greedy isolates context effects, and a cap bounds runaway generations so one pathological run cannot eat a session. Dropping capped runs would hide exactly the cost spikes the project is about.
*Consequence:* `finish_reason == "length"` is reported as a capped share alongside the expansion tail.

**D5. Warm-up and shuffling** (2026-09-17)
Three unrecorded warm-up requests, then runs shuffled within each suite.
*Why:* the first requests pay CUDA-graph and JIT costs, and thermal or background drift would otherwise correlate with config order.

---

## Attribution

**D6. Attribution renders the template with placeholders, then maps token offsets** (2026-09-17)
*Why:* it gives exact character spans for each segment without assuming anything about the template's structure, and asserting that the substituted render equals a direct render catches templates that rewrite content (Llama trims message content). Tokens are assigned by their **start** offset, which survives tokenizers whose offsets are odd; both tokenizers were verified to produce contiguous, exact offsets.
*Consequence:* `Prompter.attribute` renders three times per unique request; results are cached per `(task, config)`.

**D7. `tools` is charged the marginal cost of adding the tools parameter** (2026-09-17)
*Why:* the template, not us, renders tool schemas, and it also adds a tool preamble that exists only because tools are present. Charging the difference makes "what do tools cost" answerable.
*Consequence:* with Llama and no system message the template injects its own tool-calling system prompt, so `tools` jumps from 138 to 200 tokens in one example task. That is a genuine system×tools interaction and Shapley reports it as one.

**D8. Tool dicts are normalized to vLLM's exact shape** (2026-09-17)
`{"type": "function", "function": {"name", "description", "parameters"}}`, with `strict` only when set.
*Why:* the template JSON-serializes whatever dict it is given, so a different key order or a missing null field changes the prompt by a few tokens and breaks exactness.
*Consequence:* `attribution_ok` is the tripwire. If a future vLLM changes its serializer, that flag goes false and `_canon_tool` in `tasks.py` needs updating.

**D9. Llama's template date is pinned to `26 Jul 2024`** (2026-09-17)
Sent as `chat_template_kwargs` and used locally.
*Why:* the template writes today's date into every prompt, so prompt token counts would drift day to day and runs from different days would not be comparable.

**D10. vLLM's Llama tool template is vendored** into `ccv/templates/` (2026-09-17)
*Why:* the server and the local tokenizer must use the identical template; the file ships in the wheel so `ccv serve --chat-template` and `Prompter` can never diverge. It is Apache-2.0 and the notice is in `LICENSE`.

---

## Experiment design

**D11. Six segments, fixed order** (2026-09-17)
`system, instructions, tools, examples, retrieved, history`; `query` always kept; template overhead counted separately.
*Why:* 2⁶ = 64 subsets is the abstract's exact-Shapley budget, and the order is the mask bit order, so `subset:101010` is readable.
*Consequence:* changing the order or count invalidates every existing `config_id` and all recorded runs.

**D12. `run_id` includes the suite** (2026-09-17, bug fix)
*Why:* `noise` and `shapley` both measure the full-context baseline. Without the suite in the hash they shared a run ID, so one silently skipped the other or wrote a duplicate.
*Consequence:* the baseline is measured 31 times per task; analysis pools repeats of identical requests.

**D13. Grading happens at analysis time** (2026-09-17)
Records store `output` and `tool_calls`; `passed` is computed by `ccv analyze`.
*Why:* grader bugs are normal and GPU time is scarce; a grader fix must never require re-running the model.

**D14. Task content is fictional, with shared segments referenced as `"@name"`** (2026-09-17)
*Why:* fictional entities make the query-only configuration fail, which is what makes calibration meaningful; shared persona and tool catalogs keep 30 tasks consistent and cheap to author.

**D14b. Two-thirds of tasks must be answerable *without* `history`** (2026-09-22)
The query names its own record ("the record for ticket TK-5602") in 29 of 45 tasks; only 16 rely on the conversation to say which record is meant.
*Why:* the first draft of the benchmark phrased almost every query as "our order" or "that airframe", which made `history` strictly required in 39 of 45 tasks. Shapley would then have reported a huge, near-constant value for `history` in every workload, with no contrast between tasks and nothing to learn. The interesting measurement is the *difference* between tasks that need the conversation and tasks that do not.
*Consequence:* a reviewer adding a task must decide deliberately which side of that split it is on, and keep the ratio near a third. This is the concrete form of the plan's "about a third of tasks" rule ([PLAN.md](PLAN.md) §4).

**D15. Truncation strategy** (2026-09-17)
Head-of-text for `examples` and each retrieved doc; whole most-recent turns for `history`.
*Why:* that is what production systems actually do — trim documents, drop old turns.
*Consequence:* with few history turns, a nominal 0.8 truncation can drop a whole pair, so plots use the **achieved** kept-token fraction, which is recorded per run.

**D16. LLMLingua-2 is precomputed and cached in `benchmark/compressed.json`** (2026-09-17)
*Why:* `llmlingua` has been unmaintained since April 2024 and can conflict with the `transformers` version vLLM pins; caching keeps that dependency out of GPU sessions entirely and makes compression reproducible. Compression latency is recorded, since compression is not free.
*Fallback if the package breaks:* call the LLMLingua-2 token-classification model directly through `transformers` (~30 lines, keep the highest-probability tokens up to the rate).

---

## Statistics and analysis

**D17. Exact McNemar for accuracy, paired bootstrap for costs, BH for multiplicity** (2026-09-17)
McNemar is `scipy.stats.binomtest` on discordant pairs; the bootstrap is ~6 lines of numpy resampling task indices; BH is `scipy.stats.false_discovery_control`.
*Why:* with 30 paired tasks the exact test is right, and the hand-rolled bootstrap avoids `scipy.stats.bootstrap`'s BCa edge cases when every difference is zero (common here, because greedy decoding often reproduces the baseline exactly). The p-value uses the null-centred distribution with +1 smoothing, so it is never exactly 0.

**D18. Two cost models side by side** (2026-09-17)
Physical (GPU-seconds, energy) and price-list (`input + 5 × output`), plus a linear per-token fit whose slope ratio is the measured decode:prefill cost.
*Why:* the abstract's headline claim is the gap between a 5:1 price assumption and what the hardware actually does.

**D19. Figures are static PNGs using a colorblind-validated palette** (2026-09-17)
Scatter plots cap at three colours, so Pareto charts grey out non-frontier configs and colour only the frontier and the baseline.
*Why:* the deliverable is a LaTeX report, so there is no hover layer; the CSV tables are the accessible table view. The palette passed the validator (worst all-pairs CVD ΔE 9.2).

---

## Packaging and tooling

**D20. Distribution `context-cost-verifier`, command and import `ccv`** (2026-09-17) — `ccv` is taken on PyPI.

**D21. vLLM is not a dependency** (2026-09-17) — it runs as a separate server, so the laptop can author, test and analyse with no CUDA and no torch. `ccv serve` is the single source of truth for server flags, so the notebook cannot drift from the code.

**D22. `jinja2` is a direct dependency** (2026-09-17) — transformers 5 no longer installs it, and chat templates need it.

**D23. The benchmark flag is `--bench`, not the plan's `--tasks`** (2026-09-17) — one path points at the benchmark root, which holds `tasks/`, `shared/` and `compressed.json`.

**D24. Stdlib `argparse`, one file per concern, no framework** (2026-09-17) — keeps the tool small enough to read in one sitting, which matters for a project that has to be explained in a viva.

**D25. A missing compression-cache entry is an error record, not a crash** (2026-09-22)
`variant()` still raises `KeyError` when `benchmark/compressed.json` has no entry for a task, but `runner.run` now catches it, writes a normal `status: "error"` record and carries on; `--dry-run` counts the affected runs and prints them as "awaiting `ccv compress`".
*Why:* the cache goes stale every time a task is edited, which is the normal state during authoring. Raising out of `build()` killed a whole Kaggle session at the first `lingua2` config, after the noise and Shapley runs had already been paid for, and it made `--dry-run` — the offline task validator the runbook tells authors to keep clean — unusable at the same time.
*Consequence:* error records are retried on the next launch, so re-running after `ccv compress` fills them in with no extra bookkeeping.

---

## Session 5 (2026-10-30): scope, calibration and Kaggle automation

**D26. Qwen2.5-7B only; Llama-3.1-8B becomes future work** (2026-10-30)
*Why:* on 30 Oct no GPU run had happened and about two weeks remained. The plan's own cut order drops Llama's sweep first, and a second model doubles GPU time and needs a second Kaggle account.
*Consequence:* findings describe one model. Calibration uses Qwen alone, so the "passes on both models" rule in PLAN §4 becomes "passes on Qwen". The Llama path (template, date pin, attribution) stays covered by the offline tests.

**D27. Reports are drafted in Markdown** (2026-10-30) — no LaTeX on the laptop, and Markdown converts into whatever template the university requires. `report/formal_report.md`, `report/findings_report.md`.

**D28. Calibration is its own analysis table, and calibrate runs are excluded from every other table** (2026-10-30)
`analyze` writes `calibration.csv` (full-context passes out of 3, query-only passes, `keep`); all other tables use non-calibrate records only.
*Why:* calibration runs on draft tasks before the freeze, and pooling its baseline and query-only repeats into the Shapley cells would make the baseline an average of noise, Shapley *and* calibration runs, contradicting D12's "31 runs" description.
*Consequence:* a results file holding only calibration still analyses cleanly (calibration table and summary, no figures).

**D29. Kaggle is driven from the CLI; results carry between sessions as a private dataset** (2026-10-30)
*Why:* scripted pushes give an exact, logged record of every run's parameters, and nobody has to click through the UI. The CLI cannot attach a kernel's own previous output, so the restore cell searches `/kaggle/input/**/<model>.jsonl`, which matches both a UI-attached output and a dataset.
*Consequence:* per-run parameters live in a generated copy of the notebook; the committed notebook keeps its defaults.

**D30. The compression venv borrows the system pip** (2026-10-30)
*Why:* on Kaggle's Python 3.13 image, `python -m venv` fails at `ensurepip`. `--without-pip --system-site-packages` plus `python -m pip` installs `llmlingua` and `transformers<5` into the venv while leaving vLLM's environment untouched (the reason for the venv in D16).

**D31. Benchmark review fixes, and what was deliberately left** (2026-10-30)
33 edits from a three-way review (see REPORT_NOTES §3.3). The shared tool catalogue's example values were changed to ids no task uses (`KR-50001`, `SC-MAA-01`, `INV-10001`), because examples equal to gold values let `{tools, query}` pass without documents. A soft hint in tc-15's query and a few low-visibility world inconsistencies in few-shot examples were left for calibration to judge.
