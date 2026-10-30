# Progress tracker

**Last updated:** 2026-10-30 · **Deadline:** mid-November 2026 · **Week:** 6 (schedule started Mon 21 Sep)

> **Behind schedule.** As of 30 Oct no GPU run has happened; the plan expected Qwen finished by now. Recommended triage: drop Llama entirely, run Qwen only (smoke → calibrate → noise/shapley/ablations ≈ 5 GPU-hours), and start the SRS, UML and test-case table now since they do not depend on results.

> Read this file first when picking the project back up. [PLAN.md](PLAN.md) is what we are building, [DECISIONS.md](DECISIONS.md) is why it is built that way, [RUNBOOK.md](RUNBOOK.md) is how to run it.

---

## 1. Where the project stands

The `ccv` tool is **built and verified end to end on the laptop, without a GPU**. Nothing has been run against a real model yet: every measurement so far comes from a fake vLLM-style server in the test suite.

| Area | State | Evidence |
|---|---|---|
| Python package (7 modules + CLI) | **Done** | `uv run pytest` → 12 passed |
| Exact token attribution | **Done, verified on both tokenizers** | Per-segment counts sum exactly to the full prompt for Qwen2.5 and Llama-3.1 |
| Resumable runner | **Done, verified** | Test kills a run mid-write, leaves a half-written line, resumes with no duplicates |
| Ablation engine (64 subsets, truncate, lingua2, reorder) | **Done** | Unit tests plus a full dry run |
| Analysis (Shapley, statistics, cost fit, Pareto) | **Done** | Test plants a known structure and asserts Shapley recovers it |
| Figures (10 PNGs) | **Done, eyeballed** | Regenerated and reviewed after layout fixes |
| Benchmark tasks | **45 drafted** (15 per workload); calibration will trim to 30 | Dry run validates all 45; a test asserts every grader accepts its own gold answer |
| LLMLingua-2 cache | **Not built** | Needs `ccv compress` on a GPU box |
| Kaggle notebook | **Written, never executed** | Needs the GitHub URL and a real run |
| GPU smoke test | **Not started** ← *the next thing to do* | |
| Findings report / formal report | **Not started** | Outlines in [PLAN.md](PLAN.md) §7–8 |
| UML diagrams | **Written, not rendered** | [`docs/uml/`](uml) |

### Verified on this machine

| Check | Result |
|---|---|
| `uv run pytest` | 12 passed (~30 s; downloads two tokenizers on first run) |
| `ccv run --dry-run --suite noise shapley ablations` | 4,635 runs planned over 45 tasks; mean prompt 1,776 tokens (noise), range 1,558–2,364; the 135 `lingua2` runs are reported as awaiting `ccv compress` |
| `ccv attribute qa-01 --model qwen2.5-7b` | 1,845 tokens: tools 994 (53.9%), retrieved 420, system 148, history 113, examples 66, instructions 58, query 13, template 33 |
| Wheel build + clean-venv install | `ccv --version` works; the Llama template ships inside the wheel |
| Palette colorblind validation | All checks pass (aqua and yellow carry a contrast warning, covered by legends and the CSV table view) |

---

## 2. Next actions, in order

### A. GPU smoke test — the first real milestone (owner: A)

This is the plan's week-1 de-risk step, and everything downstream depends on it.

1. ~~Push the repo to GitHub and set the notebook's `REPO` variable~~ — done: <https://github.com/MmohithSai/context-cost-verifier>
2. Open [`notebooks/ccv_kaggle.ipynb`](../notebooks/ccv_kaggle.ipynb) on Kaggle. Accelerator **GPU T4 x2**, Internet **On**.
3. Run with `MODEL = "qwen2.5-0.5b"`, `SUITES = "noise"`, `LIMIT = "--limit 5"`.
4. Confirm in the sanity-check cell:
   - [ ] `attribution_ok` true for every record ← *the critical one; if false see [RUNBOOK.md](RUNBOOK.md) §5*
   - [ ] `server_metrics` true for every record (per-request metrics reached the client)
   - [ ] `energy_j` non-null (NVML energy counter works on this T4)
5. Repeat with `MODEL = "qwen2.5-7b"`, still `--limit 5`, and record from the log:
   - [ ] model load time, prefill ms at ~1,800 tokens, decode tokens/s
   - [ ] whether tool calls parse (run a `tc-01` request)
   - [ ] the exact working vLLM version → pin it in the notebook and write it in [DECISIONS.md](DECISIONS.md)
6. Same for `MODEL = "llama3.1-8b"` (tool parser `llama3_json` and the vendored template).
7. **Fill in §4 below** with the measured numbers so the compute budget stops being a guess.

### B. Benchmark authoring (owner: B, parallel with A) — **drafted, needs review**

- [x] Draft ~15 tasks per workload. 45 written, all validating under `--dry-run`.
- [x] About a third of tasks need a fact that appears **only** in `history` — 16 of 45 (qa-01, 02, 04, 06, 08, 15; ex-01, 02, 06, 08, 12; tc-01, 04, 06, 07, 09). The rest name their own record in the query, so removing `history` should be survivable for them. Without that split, `history` would dominate Shapley in every workload and there would be nothing to measure.
- [x] Prompts in range: 1,558–2,364 tokens, mean 1,781. The shared 8-tool catalogue is a fixed ~994 tokens of every prompt, so variation comes from the other segments.
- [ ] **Hand-check the 42 new tasks.** They were drafted in one session and have never been read by a person. Check each against the [RUNBOOK §2](RUNBOOK.md#2-writing-a-benchmark-task) checklist, especially that the gold fact really is in exactly one doc and that the distractors are genuine near misses.
- [ ] Check the fictional world stays consistent across tasks (product specs, serials, service centres, prices are reused from task to task).

### C. Calibration and freeze (owners: A and B) — unblocked once A is done

- [ ] `ccv compress` on a GPU box, then commit `benchmark/compressed.json`. Until then `--suite ablations` runs its 135 `lingua2` configs as errors, and the dry run says so rather than stopping.
- [ ] `--suite calibrate` on both models: 45 tasks × 4 = **180 requests per model**. Keep tasks that pass 3/3 with full context and fail with query only.
- [ ] Trim to 10 per workload, then `git tag benchmark-v1.0`.
- [ ] Expect casualties. The calibration bar is deliberately strict and these are first drafts; if a workload comes out short of 10, write replacements for that workload only.

### D. Then the real runs (owner: A)

- [ ] Qwen: `noise` → `shapley` → `ablations`, resuming across Kaggle commits.
- [ ] Llama on a second account (stretch).
- [ ] `ccv analyze results/*.jsonl --out analysis/` on the laptop after each batch.

### E. Documents (owner: C plus everyone)

- [ ] Render [`docs/uml/`](uml) to PNG (PlantUML VS Code extension) for the report.
- [ ] SRS draft, using [PLAN.md](PLAN.md) §8 as the skeleton.
- [ ] Test-case table from `tests/test_ccv.py` plus manual CLI cases.

---

## 3. Blocked or waiting on a decision

| Item | Needed from | Notes |
|---|---|---|
| Kaggle account with phone verification | The team | Required for Internet access inside notebooks |
| Second Kaggle account for Llama | Teammate | Only needed when the Qwen pipeline is proven |
| Whether traces are published publicly | The team | Plan says GitHub release or HF dataset in week 8; ask before anything is made public |

---

## 4. Numbers to fill in after the smoke test

These placeholders drive the compute budget; replace them with measurements.

| Quantity | Estimate used now | Measured | Where it matters |
|---|---|---|---|
| Model load time (7B AWQ, T4) | ~10 min | _TBD_ | Kaggle session budget |
| Prefill at ~1,800 tokens | ~1 s | _TBD_ | `SEC_PER_REQUEST` in `runner.py` |
| Decode speed (7B AWQ, T4) | ~30 tok/s | _TBD_ | Run time for 3,090 requests |
| Mean seconds per request | 6 s | _TBD_ | Dry-run estimate line |
| Full sweep per model | ~5 GPU-hours | _TBD_ | Whether both models fit the schedule |
| Runs hitting `max_tokens=1024` | assumed rare | _TBD_ | Censoring in the expansion analysis |

---

## 5. Known gaps and deliberate simplifications

1. **The 45 tasks are drafts that no human has read yet.** They validate, and every grader accepts its own gold answer, but that only proves the files are well-formed. Calibration on a real model is what decides which 30 survive.
2. **`SEC_PER_REQUEST = 6`** in `runner.py` is a guess used only for the dry-run time estimate. Update after the smoke test.
3. **Reorder assumes `gold_doc`.** Tasks without it are skipped for that family. All current tasks have one.
4. **Truncation strategy is fixed:** head-of-text for `examples` and each doc, most-recent whole turns for `history`. With only 2 history pairs, an 0.8 truncation can drop a whole pair, so the achieved ratio is recorded and plots use the achieved fraction, not the nominal one.
5. **Baseline pooling.** The baseline config is measured by both `noise` (30×) and `shapley` (1×); analysis pools repeats of identical requests, so the baseline cell is an average over ~31 runs while other cells are single runs. Fine for deterministic greedy decoding; the noise floor quantifies the residual.
6. **The `tools` segment absorbs the template's tool preamble.** With Llama and no system message, the template adds its own tool-calling system prompt, which is charged to `tools` (138 → 200 tokens in one example). This is a real interaction and Shapley reports it as such.
7. **Figures are static PNGs** for the LaTeX report, so they have no hover layer; the CSV tables are their accessible table view.
8. **No CI.** Tests are run locally.
9. **The shared tool catalogue is ~994 tokens, about 55% of a full prompt.** That is a deliberate design choice (one 8-tool catalogue on every task, per [PLAN.md](PLAN.md) §4), not a property of the workloads, so `tools` will carry a large Shapley cost share by construction. The report must say so before drawing a conclusion about what tools cost in general.
10. **The mask sweep in `test_attribution_is_exact` runs on one task per workload.** Every task is still checked at full context; the six-mask sweep is a template property, and running it on all 45 tasks multiplied the suite runtime for no extra coverage.

---

## 6. Session log

### 2026-09-17 (session 1) — plan approved, core built

- Read the abstract PDF; checked the machine (Python 3.13.1, uv 0.7.5, RTX 3050 4 GB — too small for 7B, WSL has no distro) and confirmed the GPU work must happen on Kaggle.
- Verified the external facts the plan depends on: vLLM 0.29 supports AWQ on Turing, fp16 only, `TRITON_ATTN` backend, `--enable-per-request-metrics` returns per-request TTFT and decode time, all four Hugging Face models are ungated, `llmlingua` is stale (0.2.2, Apr 2024), and PyPI name `ccv` is taken so the distribution is `context-cost-verifier`.
- Asked the team three questions; got: mid-Nov deadline, Kaggle T4, formal report also required.
- Plan approved. Built the package, the three example tasks, the test suite, the notebook and the UML sources.
- Fixed along the way: `run_id` now includes the suite (the noise and Shapley baselines collided); Llama's template date pinned; tool dicts normalized to vLLM's shape; `jinja2` added as a dependency (transformers 5 dropped it); figure layout issues found by looking at the rendered PNGs.

### 2026-09-18 (session 2) — documentation and first publish

- Wrote [PLAN.md](PLAN.md), this tracker, [DECISIONS.md](DECISIONS.md) and [RUNBOOK.md](RUNBOOK.md).
- First commit, pushed to the public repo <https://github.com/MmohithSai/context-cost-verifier>; the notebook's `REPO` now points there.

### 2026-09-22 (session 3) — benchmark drafted

- Re-verified everything session 2 claimed: tests pass, and `ccv attribute qa-01` and the dry run reproduce the recorded numbers exactly.
- Drafted 42 tasks, taking the benchmark from 3 to 45 (15 per workload). The GPU tracks (A, C, D) were all blocked on a Kaggle session, so this was the only unblocked work on the critical path.
- Caught and fixed a design fault in the first draft: 39 of 45 queries were phrased as "our order" / "that airframe", so they could not be answered without `history`. That would have made `history` dominate Shapley in every workload with no contrast to measure. 24 tasks were rewritten so the query names its own record; 16 of 45 now genuinely need `history`.
- Added a test asserting every task's grader accepts its own gold answer and rejects each listed distractor. It immediately caught two QA tasks whose `answers` were phrased so that a correctly-formatted reply would have been marked wrong — the kind of bug that otherwise only shows up after the GPU time is spent.
- Fixed a fault-tolerance hole: a missing LLMLingua-2 cache entry raised straight out of `build()` and killed the whole session. It is now recorded as an error record (so resume retries it after `ccv compress`) and counted in the dry run instead of aborting it.
- Trimmed the mask sweep in `test_attribution_is_exact` to one task per workload; at 45 tasks it had become the dominant cost of the suite.

### 2026-10-30 (session 4) — status review, session 3 pushed

- Reviewed the repo against the plan: code complete, 12 tests pass, the dry run validates all 45 tasks (4,815 runs over all four suites, 135 awaiting `ccv compress`). `results/` and `report/` are still empty.
- Session 3's work had never been committed, so GitHub (and therefore the Kaggle notebook, which clones it) still had the 3-task benchmark without the D25 fix. Committed and pushed.
- Fixed a dry-run crash on Windows consoles: the summary line printed `≈`, which cp1252 cannot encode.
- The `ccv.exe` launcher in `.venv` still pointed at the project's old Desktop path ("Failed to canonicalize script path"); reinstalled the editable package from `D:\major project`.
- Still open: `notebooks/kernel-metadata.json` has the placeholder `KAGGLE_USERNAME`; set it before using the Kaggle CLI.

---

## 7. How to pick this up in a new session

```bash
uv run pytest
```

Then:
1. Read §1 and §2 of this file.
2. Skim [DECISIONS.md](DECISIONS.md) before changing anything in `attribution.py` or the server flags in `cli.py`; several non-obvious choices there are load-bearing for measurement validity.
3. Use [RUNBOOK.md](RUNBOOK.md) for anything involving Kaggle or vLLM.
4. Update §1, §2 and §6 of this file at the end of the session.
