# Report notes: what was done, what went wrong, and why each choice was made

This file is raw material for the findings report and the formal university report. It is organised by report chapter, not by date. Every claim points at evidence (a command, a file, a number) so it can be cited or reproduced.

Companion files: [PROGRESS.md](PROGRESS.md) (status), [DECISIONS.md](DECISIONS.md) (the decision log, D1–D…), [PLAN.md](PLAN.md) (the approved plan), [RUNBOOK.md](RUNBOOK.md) (how to run things).

---

## 1. Project timeline at a glance

| Date | Session | What happened |
|---|---|---|
| 2026-09-17 | 1 | Plan approved; the whole `ccv` package, three example tasks, test suite, Kaggle notebook and UML sources built |
| 2026-09-18 | 2 | Project documentation written; first commit pushed to GitHub |
| 2026-09-22 | 3 | Benchmark expanded from 3 to 45 draft tasks; two design faults found and fixed |
| 2026-10-30 | 4 | Status review; session 3 work committed and pushed; Windows console crash fixed |
| 2026-10-30 | 5 | Scope cut to Qwen only, local GPU smoke test, benchmark review, Kaggle runs, analysis, reports (this file) |

---

## 2. System design and implementation (formal report ch. 5–6, findings report §3)

### 2.1 Why the architecture looks the way it does

| Choice | Alternatives considered | Why this one | Evidence |
|---|---|---|---|
| vLLM runs as a **separate HTTP server**, never imported (D21) | Import vLLM as a library | The laptop (Windows, 4 GB GPU) can then author, test and analyse with no CUDA; the tool works against *any* OpenAI-compatible endpoint, which the abstract promises | `pyproject.toml` has no vLLM/torch dependency |
| **One module per concern**, stdlib `argparse` (D24) | Click/Typer, a package hierarchy | Small enough to read in one sitting, which matters for a viva | 7 modules in `ccv/` |
| `ccv serve` is the **single source of truth for server flags** | Flags written into the notebook | Notebook and code can never drift; measurement-critical flags (unbatched, no prefix cache) cannot be forgotten | `ccv/cli.py` `serve_command` |
| Results are **append-only JSONL with fsync** after each record | SQLite, CSV, one file per run | Survives a killed Kaggle session with at most one half-written line, which resume repairs; concatenation merges files from several accounts | `runner.py`, test `test_resume*` |
| **Grading at analysis time**, not in the runner (D13) | Grade while running | Grader bugs are normal; fixing one must never cost GPU time | Two grader bugs were in fact caught later (§3.2) |

### 2.2 Measurement validity: the problems and how they were handled

- **Batching would make per-request GPU time meaningless.** If two requests share the GPU, neither "owns" the time. Fix: `--max-num-seqs 1`, so `gpu_seconds = prefill_ms + decode_ms` is the true device time (D1, D2). Cost: results describe unbatched serving — stated as a limitation.
- **Prefix caching would hide prefill cost.** The noise suite sends the same prompt 30 times; with a prefix cache, 29 of them would skip prefill. Fix: `--no-enable-prefix-caching` (D1).
- **TTFT from an arbitrary endpoint.** Only streaming exposes time-to-first-token portably, but HTTP overhead pollutes it. Fix: always stream, record client timings, and prefer vLLM's server-side per-request metrics when present (D3).
- **Runaway generations.** A model that loses context may ramble until the session dies. Fix: `max_tokens=1024`; runs that hit the cap are *censored, not dropped*, because dropping them would hide exactly the cost spikes the project is about (D4).
- **Thermal drift and JIT warm-up.** Fix: three unrecorded warm-up requests, then shuffled run order inside each suite so drift does not line up with configuration (D5).

### 2.3 Exact token attribution — the hardest engineering problem

**Goal:** say exactly how many prompt tokens each of the six segments costs, using the served model's own tokenizer and chat template.

**Challenge 1 — templates insert their own tokens.** Role headers, special tokens and default system prompts belong to no segment. *Approach:* render the template with a unique placeholder for each segment, substitute the real text back while recording character spans, tokenize with offset mapping, and give each token to the span containing its start offset; everything else is `template` (D6). *Check:* the substituted render must equal a direct render, which catches templates that rewrite content (Llama trims message text).

**Challenge 2 — the prompt must be byte-identical to the server's.** Three separate causes of mismatch were found and fixed in session 1:
1. Tool dicts: the template JSON-serialises whatever dict it receives, so key order or a missing `null` field changes the token count. *Fix:* normalise tools to vLLM's exact `model_dump()` shape (D8), confirmed by reading vLLM v0.29.0 source.
2. Llama's template writes **today's date** into the prompt, so token counts would drift by day. *Fix:* pin the date to `26 Jul 2024` via `chat_template_kwargs` (D9).
3. Llama tool calling needs vLLM's own JSON tool template. *Fix:* vendor it into `ccv/templates/` so server and client load the same file (D10).

*Tripwire:* every record carries `attribution_ok` (local total == server `usage.prompt_tokens`). This is the first check on any real run.

**Challenge 3 — what do "tools" cost?** The template, not us, renders tool schemas and adds a preamble only when tools exist. *Approach:* charge `tools` the marginal cost of adding the tools parameter (D7). This exposed a genuine interaction: with Llama and no system message, the template injects its own tool-calling system prompt, so `tools` goes from 138 to 200 tokens — Shapley reports it as a system×tools interaction.

### 2.4 Resumability on free GPUs

Kaggle sessions die (time limits, quota, preemption). The runner was designed around that from the start:
- `run_id = sha256(model, suite, task_id, task_hash, config_id, repeat, sampling)`; finished `ok` IDs are skipped on restart; error records are retried.
- `--max-minutes` stops starting new requests before Kaggle kills the session, so the output is always saved.
- A test kills a run mid-write, leaves a half-written line, and asserts that resume produces no duplicates.

### 2.5 Bugs found during development (good material for the testing chapter)

| Bug | How it was found | Fix | Decision |
|---|---|---|---|
| `noise` and `shapley` both measure the baseline and **shared a `run_id`**, so one silently skipped the other | Reasoning about resume while writing the runner | `run_id` includes the suite | D12 |
| `transformers` 5 stopped installing `jinja2`; chat templates failed | Clean-venv install | Direct `jinja2` dependency | D22 |
| A missing LLMLingua-2 cache entry **raised out of `build()` and killed the whole session** — after noise and Shapley had already been paid for — and also broke `--dry-run` | Session 3, editing tasks made the cache stale | Recorded as an error record and retried after `ccv compress`; dry run reports it | D25 |
| Dry-run summary printed `≈`, which the Windows cp1252 console cannot encode | Running the dry run on Windows | ASCII output | session 4 |
| Stale `ccv.exe` launcher pointing at the project's old Desktop path | Moving the project folder | Reinstalled editable package | session 4 |
| Figure layout problems (overlapping labels, legends) | Looking at the rendered PNGs, not just running the code | Layout fixes in `report.py` | session 1 |

---

## 3. Benchmark design (findings report §4, formal report data design)

### 3.1 Design rules and why

- **Fictional world ("Kestrel Robotics")** so answers cannot come from the model's memory; this is what makes "query only" fail and calibration meaningful (D14).
- **Every task fills all six segments**, so a 64-subset sweep is meaningful on every task.
- **One gold document plus near-miss distractors** so removing or truncating `retrieved` has a measurable accuracy effect and wrong answers are recognisable.
- **Shared persona and an 8-tool catalogue** referenced as `@kestrel_persona` / `@kestrel_tools`, so 45 tasks stay consistent and cheap to write. Known consequence: the catalogue is ~994 tokens, about 55% of every prompt, so `tools` carries a large cost share *by construction* — the report must say so.

### 3.2 Two design faults caught before any GPU time was spent (session 3)

1. **History dominated everything.** In the first draft, 39 of 45 queries said "our order" or "that airframe", so every one needed `history`. Shapley would have given `history` a huge, near-constant value in every workload — no contrast, nothing learned. *Fix:* 24 tasks rewritten so the query names its own record; 16 of 45 now genuinely need history (D14b). The interesting quantity is the *difference* between the two groups.
2. **Graders rejecting correct answers.** A new test asserts that every task's grader accepts its own gold answer and rejects every listed distractor. It immediately caught two QA tasks whose answers were phrased so that a correctly formatted reply would have been marked wrong.

### 3.3 Pre-GPU review of all 45 drafts (session 5)

**Why:** the 42 drafted tasks had never been read by a person, and every defect found *after* calibration costs GPU time and can silently bias Shapley.

**Approach:** three independent reviewers (one per workload), each given the RUNBOOK §2 checklist plus three extra checks: (a) cross-task consistency of the fictional world, (b) whether a *correct, naturally phrased* answer would pass the grader, (c) whether any subset *without* the documents could still pass (which would flatten the Shapley contrast). They reported; I applied the fixes with a script that asserts each target text appears exactly the expected number of times, so no edit could land in the wrong place.

**What was found (33 edits across 20 files):**

| Kind of defect | Examples | Why it matters |
|---|---|---|
| Contradictions between tasks | qa-01's bulletin required firmware 4.2.1 for *all* K-series drones, while other tasks list different minimums per model; qa-06's "last flown" dates were *before* the delivery dates in ex-08; serials reused with conflicting histories | The model can be shown two contradicting facts in one prompt only if tasks share docs, but contradictions undermine the benchmark's credibility in a viva |
| A contradiction *inside one prompt* | ex-11 said the service plan was 9.5% of list price, but the two quotes differ by 9.49% | The persona tells the model to flag contradictions, so a careful model would answer in prose and fail the JSON grader |
| Correct answers that would fail | qa-09 accepted only "1.5% per month" (not "1.5%"); qa-11 lacked "12 business days"; qa-05 lacked "IP 55"; ex-02 expected "Whitefield Robotics" while the doc said "Whitefield Robotics (Bengaluru)" | False negatives in the baseline → task wrongly dropped at calibration |
| Distractors a good answer may mention | qa-07 listed 1.6 m (a related threshold) as a distractor | A thorough correct answer would be marked wrong |
| Guessable from world knowledge | 1.5%/month is the most common real late fee | Query-only might pass → task wrongly dropped; changed to 1.75% everywhere |
| Answer leaked by the query or tool catalogue | tc-02 / tc-10 named the order number the tool needs; the shared tool catalogue's "e.g." values were gold ids of tc-01 and tc-08; a tc-03 example used the gold room | `{tools, query}` would pass with no documents, so `retrieved` would get no Shapley credit |
| Wrong metadata | tc-12 `gold_doc` pointed at the wrong document | Breaks the reorder ablation for that task |

**Judgement calls (recorded so they can be defended):**
- When two reviewers proposed opposite fixes for the same contradiction (change qa-06's dates vs change ex-08's delivery dates), I changed qa-06: a one-task edit beats a multi-task one.
- Not applied: tc-15's query hint ("runs fine on its own") — a soft hint, not a defect; calibration decides. Low-priority world inconsistencies left as-is: one example serial (SB1-20044) used inconsistently inside few-shot examples, mixed currencies for one customer, a K4 delivery date. None is visible inside a single prompt's gold path.
- The test fixture keyed on qa-01's old history serial, so it failed after the rename — a reminder that tests embedding task text must move with the tasks.

**Evidence:** commit `dcc4976`; `pytest` 12 passed; dry run 4,635 runs, prompts 1,558–2,365 tokens.

### 3.4 Calibration (rule)

Keep tasks whose full-context run passes 3/3 and whose query-only run fails, then keep 10 per workload. Results of the real calibration are in §5 below.

---

## 4. Scope decisions made under schedule pressure (session 5)

On 30 Oct the project was about five weeks behind on GPU work (the plan expected Qwen to be finished by week 5) with about two weeks to the deadline.

| Decision | Why | Consequence |
|---|---|---|
| **Qwen2.5-7B only**; Llama-3.1-8B becomes future work | The plan's own cut order drops Llama's sweep first; a second model doubles GPU time and needs a second Kaggle account | Findings are for one model; the report states this as a limitation. The Llama code path (template, date pin) stays tested offline |
| **Reports drafted in Markdown** | No LaTeX on the laptop; Markdown converts into whatever template the university requires | `report/*.md` |
| **Smoke test on the laptop GPU first**, in a vLLM Docker container, with the 0.5B model | A Kaggle round trip (queue, install, model load) costs ~20–30 minutes per attempt; the laptop's RTX 3050 can hold the 0.5B model and catch template/metrics problems in minutes | See §5.1 |
| **Kaggle driven from the CLI** (`kaggle kernels push/status/output`) | Removes manual clicking and lets runs be logged exactly | Needs one small notebook change for resume (§5.2) |

---

## 5. Experiments (session 5 onwards)

### 5.1 Local smoke test

- **Plan:** serve Qwen2.5-0.5B in the official `vllm/vllm-openai:v0.29.0` Docker image on the laptop's RTX 3050 (Docker Desktop with the NVIDIA runtime was already installed).
- **Challenge:** the ~10 GB image download failed with `unexpected EOF` partway through (network drop); retried, since completed layers are kept.

### 5.1b LLMLingua-2 compression cache (measured, run K1c)

389 long texts (examples, each retrieved doc, each history turn) × 3 rates, compressed on the Kaggle T4 with `microsoft/llmlingua-2-xlm-roberta-large-meetingbank`. Token counts below use Qwen2.5's tokenizer.

| Target rate | Achieved kept fraction: median (min–max) | Compression time per text: median / p90 |
|---|---|---|
| 0.8 | 0.82 (0.69–0.90) | 111 ms / 125 ms |
| 0.5 | 0.53 (0.30–0.69) | 111 ms / 124 ms |
| 0.2 | 0.22 (0.08–0.36) | 111 ms / 124 ms |

LLMLingua-2 hits its target on average but varies a lot per text. That is why plots use the **achieved** fraction, not the nominal rate (D15). Compression is not free: ~0.1 s per text, i.e. several texts per request. That overhead belongs in any cost comparison against plain truncation, which is essentially instant.

### 5.2 Kaggle automation

**Approach:** drive Kaggle entirely from the Kaggle CLI: `kaggle kernels push` (a per-run copy of the notebook with `MODEL`/`SUITES`/`LIMIT`/`COMPRESS` substituted, so the committed notebook keeps its defaults), a background `kaggle kernels status` loop, and `kaggle kernels output` to download results. The API token is held in an environment variable read from a file outside the repository; it is never committed.

**Why a dataset for resume:** in the Kaggle UI you attach a previous version's output to resume; the CLI cannot attach a kernel's own output to itself. Results are therefore carried between sessions as a private dataset, and the notebook's restore cell now searches `/kaggle/input/**/<model>.jsonl` instead of requiring a `results/` folder.

### 5.3 Kaggle run log

| Run | Date (UTC) | Parameters | Kernel version | Duration | Records (ok / error) | Outcome |
|---|---|---|---|---|---|---|
| K1 | 2026-10-30 06:58 | qwen2.5-7b, `calibrate`, `COMPRESS=True` | 1 | ~6 min | 0 / 0 | **Failed: no internet in the kernel.** `pip install vllm` could not resolve pypi.org, so nothing installed and the notebook stopped at `ccv serve`. Kaggle silently disables Internet for accounts without phone verification. A second, independent bug also surfaced: `python -m venv` on Kaggle's Python 3.13 image fails at `ensurepip`, so the compression venv could never have worked. |

| K1b | 2026-10-30 ~07:40 | same, with boolean metadata | 2 | ~4 min | 0 / 0 | **Failed again, no internet** — rules out the string-valued `enable_internet` flag as the cause; the account needs phone verification. The new install assertion stopped the notebook in the first code cell with the message "install failed: is Internet on?", as intended. |
| K1c | 2026-10-30 07:19 | same, after phone verification | 3 | ~13 min | 0 / 0 | **Partial success.** Installs worked and **LLMLingua-2 compression of all 45 tasks succeeded** (`benchmark/compressed.json`, 155 KB) — the D30 venv fix worked first time. Then `vllm serve` crashed at import: vLLM 0.29 installs a PyTorch built for CUDA 13.0, Kaggle preinstalls a `torchaudio` built for CUDA 12.8, and `transformers` imports `torchaudio` whenever it is installed. Fix: `pip uninstall torchaudio` after installing vLLM (D32). Also noticed the repo was cloned into `/kaggle/working`, so the saved output contained a full copy of the repo; it is now cloned to `/kaggle/tmp`. |
| K1d | 2026-10-30 07:33 | qwen2.5-7b, `calibrate`, `COMPRESS=False` (cache committed) | 4 | | | |

**Fixes after K1:**
1. The compression venv is created `--without-pip` and uses the system pip through `--system-site-packages` (`python -m pip`), and calls `python -m ccv.cli compress`.
2. The install cell now asserts `ccv` is on the path with a message naming the likely cause, so a missing-internet run fails in the first minute with a clear reason instead of cascading errors.
3. `kernel-metadata.json` uses JSON booleans rather than the strings `"true"`.

**Lesson for the report:** the notebook had been written in week 1 but never executed. Both failures were invisible offline — exactly why the plan put a smoke test first. The cost was one 6-minute session.

### 5.3 Measured hardware numbers

_See [PROGRESS.md](PROGRESS.md) §4 once measured._

---

## 6. Non-issues worth mentioning (so nobody "fixes" them)

- **`uv run pytest` prints "Failed to canonicalize script path".** The `.venv` launchers still point at the project's old Desktop location (same cause as the stale `ccv.exe` in session 4). Use `.venv/Scripts/python.exe -m pytest` or recreate the venv; the code is unaffected.
- **UML not rendered by Claude.** Downloading the PlantUML jar was blocked by the session's permission rules; render `docs/uml/*.puml` with the PlantUML VS Code extension.
- **README "mojibake".** Viewed with Windows PowerShell 5.1's `Get-Content`, `README.md` shows `Benjaminiâ€“Hochberg` and `Â·`. The file on disk is valid UTF-8; PowerShell 5.1 decodes files without a byte-order mark as cp1252. Nothing to fix — read files with `-Encoding UTF8`.
