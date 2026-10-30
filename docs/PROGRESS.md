# Progress tracker

**Last updated:** 2026-10-30 (session 5) · **Deadline:** mid-November 2026 · **Week:** 6 (schedule started Mon 21 Sep)

> **Back on track for Qwen.** Session 5 cut scope to Qwen2.5-7B only (D26), ran the first real GPU sessions, calibrated and froze the benchmark (`benchmark-v1.0`, 30 tasks) and launched the main sweep (Kaggle run K2). Llama is future work.

> Read this file first when picking the project back up. [PLAN.md](PLAN.md) is what we are building, [DECISIONS.md](DECISIONS.md) is why it is built that way, [RUNBOOK.md](RUNBOOK.md) is how to run it, and [REPORT_NOTES.md](REPORT_NOTES.md) is the report-writing journal (every challenge, option and choice, with evidence).

---

## 1. Where the project stands

| Area | State | Evidence |
|---|---|---|
| Python package (7 modules + CLI) | **Done** | `.venv/Scripts/python.exe -m pytest` → 12 passed |
| Exact token attribution | **Verified on the real server** | `attribution_ok` 240/240 on Kaggle (Qwen2.5-7B, vLLM 0.29.0) |
| Resumable runner | **Verified across Kaggle sessions** | Resume from a private dataset reran only changed tasks, no duplicates |
| Swallowed-output recovery | **Done** (D35) | Fake-server test; 33 real extraction answers recovered in run K1f |
| LLMLingua-2 cache | **Done** | `benchmark/compressed.json`, built on the T4 (run K1c) |
| Benchmark | **Frozen: `benchmark-v1.0`, 30 tasks** | Calibration table `results/evidence/calibration_v2_final.csv`; 15 drafts in `benchmark/drafts/` |
| Kaggle automation | **Done** | Kaggle CLI push/status/output; results carried in private dataset `mmohithsai/ccv-results` |
| Qwen main sweep (noise, Shapley, ablations) | **Running** — Kaggle run K2, started 08:26 UTC | 3,090 requests, ~2.8 GPU-hours |
| Analysis and figures | Code done; waiting for K2 data | |
| Formal report | **Drafted** except results | [report/formal_report.md](../report/formal_report.md) |
| Findings report | **Drafted** except results, recommendations, conclusion | [report/findings_report.md](../report/findings_report.md) |
| UML diagrams | Written, not rendered | [`docs/uml/`](uml); render with the PlantUML VS Code extension |
| Llama-3.1-8B sweep | **Cut** (D26) | Future work |

---

## 2. Next actions, in order

### A. Finish the Qwen sweep
- [ ] When K2 finishes: download output (`kaggle kernels output mmohithsai/ccv-context-cost-verifier -p <dir>`), copy `results/qwen2.5-7b.jsonl` into the repo, and check the RUNBOOK §5 table (no duplicate `run_id`, `attribution_ok` all true).
- [ ] If K2 stopped on its time budget: upload the JSONL as a new version of the `ccv-results` dataset and push the same run again; it resumes.
- [ ] If the noise floor shows non-identical repeats, give `ablations` 3 repeats (PLAN §5).

### B. Analysis
- [ ] `ccv analyze results/qwen2.5-7b.jsonl --out analysis/`; read `summary.md` and look at every figure.
- [ ] Report how often `output_recovered` occurs per config (a measure of the tools segment pulling the model to tool-call formatting).

### C. Reports
- [ ] Fill findings report §6, §7, §9 and formal report ch. 8–9 and Appendix B from `analysis/`.
- [ ] Render [`docs/uml/`](uml) to PNG.
- [ ] Revoke the Kaggle API token used in session 5 once runs are done.

### D. Release (week 8)
- [ ] RUNBOOK §7 checklist; tag `v1.0`; ask the team before publishing traces.

## 3. Blocked or waiting on a decision

| Item | Needed from | Notes |
|---|---|---|
| Revoke the session-5 Kaggle API token | Account owner | After the Qwen runs are done |
| Whether traces are published publicly | The team | Plan says GitHub release or HF dataset in week 8; ask before anything is made public |

---

## 4. Measured numbers (Kaggle T4, Qwen2.5-7B-Instruct-AWQ, vLLM 0.29.0)

| Quantity | Estimate used before | Measured | Where it matters |
|---|---|---|---|
| Server start to healthy (incl. weight download) | ~10 min | 170 s | Kaggle session budget |
| Prefill at ~1,756 tokens | ~1 s | 2,162 ms (1.24 ms/token) | Cost fit |
| Decode speed | ~30 tok/s | 35.9 tok/s (27.9 ms/token) | Cost fit |
| Mean seconds per request | 6 s | 3.27 s (`SEC_PER_REQUEST = 3.3`) | Dry-run estimate |
| Full sweep, 30 tasks | ~5 GPU-hours | ~2.8 GPU-hours estimated; K2 will confirm | Schedule |
| Runs hitting `max_tokens=1024` | assumed rare | 0 of 240 calibration runs | Censoring |
| Energy per full-context request | — | 196 J median | Energy cost model |

## 5. Known gaps and deliberate simplifications

1. **Calibration used one model** (Qwen), so tasks are calibrated to Qwen, not to a pair of models as PLAN §4 intended (D26).
2. **Extraction format guidance sits partly in the query** (D33), so `instructions` gets less Shapley credit in extraction than it would otherwise.
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

### 2026-10-30 (session 5) — first GPU runs, calibration, freeze, sweep launched

- Scope cut to Qwen only; reports in Markdown; Kaggle driven from the CLI (D26, D27, D29).
- Three-way review of all 45 drafts → 33 fixes (REPORT_NOTES §3.3).
- Added the calibration table to `analyze` (D28) and the end-to-end test.
- Kaggle runs: K1/K1b had no internet (account not phone-verified); K1c built the LLMLingua-2 cache, then vLLM died on a torchaudio/CUDA mismatch (D32); K1d was the first good run — all sanity checks pass; K1e/K1f fixed extraction (tools hijack, D33; swallowed streamed output, D35).
- Froze 30 tasks as `benchmark-v1.0` (D36) and launched the main sweep K2.
- Drafted both reports except the results chapters; wrote REPORT_NOTES.md.
- Dropped the local Docker smoke test (D34); UML rendering left to the PlantUML extension (jar download blocked).

---

## 7. How to pick this up in a new session

```bash
.venv/Scripts/python.exe -m pytest     # `uv run` fails until the venv is recreated (stale launcher paths)
```

Then:
1. Read §1 and §2 of this file.
2. Skim [DECISIONS.md](DECISIONS.md) before changing anything in `attribution.py` or the server flags in `cli.py`; several non-obvious choices there are load-bearing for measurement validity.
3. Use [RUNBOOK.md](RUNBOOK.md) for anything involving Kaggle or vLLM.
4. Update §1, §2 and §6 of this file at the end of the session.
