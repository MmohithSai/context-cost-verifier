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

### 3.3 Calibration (planned rule)

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

_To be filled in as it runs._

### 5.2 Kaggle run log

| Run | Date | Parameters | Kernel version | Duration | Records (ok / error) | Outcome |
|---|---|---|---|---|---|---|

### 5.3 Measured hardware numbers

_See [PROGRESS.md](PROGRESS.md) §4 once measured._

---

## 6. Non-issues worth mentioning (so nobody "fixes" them)

- **README "mojibake".** Viewed with Windows PowerShell 5.1's `Get-Content`, `README.md` shows `Benjaminiâ€“Hochberg` and `Â·`. The file on disk is valid UTF-8; PowerShell 5.1 decodes files without a byte-order mark as cp1252. Nothing to fix — read files with `-Encoding UTF8`.
