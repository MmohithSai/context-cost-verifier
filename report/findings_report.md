# Does Shrinking the Prompt Save Cost? Physical Measurements, Exact Token Attribution and Shapley Ablations on a T4

*Findings report (target: 10 pages). Draft in Markdown. Figures come from `analysis/` after `ccv analyze`.*

> **Draft status:** §1–5 and §8 are written. §6, §7 and §9 are filled from the Qwen2.5-7B runs. Every number in §6 must trace to a file in `analysis/`.

---

## 1. Introduction

Teams shrink LLM prompts to save money: truncate retrieved documents, drop older conversation turns, strip few-shot examples, or run a learned compressor such as LLMLingua-2. The implicit model of cost is *input tokens × input price*. That model is incomplete in two ways.

1. **Output tokens dominate.** Commercial APIs bill an output token at roughly 5× an input token. On the hardware, prefill processes the whole prompt in parallel, while decode emits one token per sequential forward pass.
2. **Context changes behaviour.** A model with less context can answer at greater length, hedge, ask for clarification, or answer wrongly. Every wrong answer has a downstream cost: a retry, an escalation, or a bad decision.

We ask three questions about a 7B instruction-tuned model served on a single T4:

- **RQ1.** Which parts of a realistic prompt carry accuracy, and which carry cost, once interactions between parts are accounted for?
- **RQ2.** When context is reduced (by truncation, LLMLingua-2 compression or reordering), how often does total cost *rise* because outputs get longer?
- **RQ3.** How does the measured physical cost ratio of decode to prefill compare with the 5:1 price-list ratio?

**Contributions.** (i) `ccv`, an open-source tool that joins physical per-request GPU cost, exact per-segment token attribution and graded accuracy on the same requests. (ii) An exact 64-subset Shapley analysis of six prompt segments, with pairwise interactions. (iii) A calibrated 30-task benchmark over three workloads (QA, JSON extraction, tool calling) set in a fictional world, so answers cannot come from memory. (iv) Raw JSONL traces for every request.

---

## 2. Related work

- **Context length and accuracy.** *Lost in the Middle* (Liu et al., 2023) shows position-dependent use of long contexts. Shi et al. (2023) show that irrelevant context distracts models. Chroma's *Context Rot* (2025) reports non-uniform degradation as inputs grow. These works measure accuracy, not cost.
- **Prompt compression.** LLMLingua (Jiang et al., 2023) and LLMLingua-2 (Pan et al., 2024) compress prompts 2–5× with modest accuracy loss. They report input savings but not output-side effects on cost.
- **Attribution.** Shapley values (Shapley, 1953) and SHAP (Lundberg & Lee, 2017) attribute model outputs to inputs, usually by sampling. With six segments we enumerate all 64 coalitions exactly, and add Shapley interaction indices (Grabisch & Roubens, 1999).
- **Serving.** vLLM (Kwon et al., 2023) batches and caches prefixes for throughput. We disable both so each request's GPU time is its own (§3.1).

---

## 3. System design

### 3.1 Measurement

Each request is streamed to a vLLM 0.29 server configured with `--max-num-seqs 1` (unbatched) and prefix caching off, so a request owns the GPU and `gpu_seconds = prefill_ms + decode_ms`. Server-side per-request timings are preferred over client timings. An NVML thread samples utilisation, memory and the energy counter every 100 ms. Decoding is greedy, with `max_tokens = 1024`; capped runs are kept and reported as censored. Three warm-up requests precede each session, and runs are shuffled within each suite to spread thermal drift.

### 3.2 Exact token attribution

The prompt is split into six ablatable segments: **system** (persona), **instructions** (format and brevity rules), **tools** (8-tool catalogue), **examples** (few-shot), **retrieved** (4–6 documents) and **history** (prior turns). The query is always kept, and template overhead is counted separately. We render the served model's chat template with placeholders, map tokens to segments by character offset, and verify that the local count equals the server's `usage.prompt_tokens` (`attribution_ok`) for every request.

### 3.3 Fault tolerance

Free GPU sessions die. Each record is appended and `fsync`ed; a content-hashed `run_id` makes reruns skip finished work; `--max-minutes` stops new requests before the session limit.

---

## 4. Benchmark

**World.** All tasks are set in "Kestrel Robotics", a fictional maker of cargo drones and warehouse robots, so the query alone cannot be answered from memory.

**Workloads.** QA (short factual answers, normalised exact match that rejects hedging between candidate values), extraction (a JSON object whose fields must all match), and tool calling (the first tool call must have the right name and arguments).

**Segments.** Every task fills all six segments. The shared tool catalogue costs ~994 tokens, about half of a full prompt. That is a design choice, so `tools` carries a large *cost* share by construction, and we interpret its cost Shapley value with this in mind.

**History split.** About a third of tasks need a fact that lives only in the conversation history (the query says "our order"). The rest name their own record. This split is what lets Shapley distinguish `history` from `retrieved`.

**Quality control.** (i) A unit test checks that every grader accepts its own gold answer and rejects every listed distractor. (ii) A three-way review of all 45 drafts before any GPU time found and fixed 33 defects: cross-task contradictions, answer phrasings a correct reply would fail, queries that leaked the answer, and gold identifiers leaked through tool-catalogue examples. (iii) **Calibration:** each draft is run 3× with full context and 1× with the query only. A task is kept only if it passes 3/3 with full context and fails with the query alone. We keep 10 per workload.

**Calibration outcome.** QA 15/15, tool calling 15/15 and extraction 13/15 drafts passed; query-only failed on all 45. Extraction needed two fixes first. Tool definitions made the model call lookup tools instead of reading the documents (42/45 full-context runs), until each extraction query said where to take the answer from. The model then wrapped its JSON answers in `<tool_call>` tags, which vLLM's streaming tool parser silently dropped (33/45 runs); the runner now recovers such output with an unstreamed request. Ten tasks per workload were frozen as `benchmark-v1.0`.

---

## 5. Method

**Suites per task.** `noise`: baseline × 30. `shapley`: all 64 subsets × 1. `ablations`: truncation at 0.8/0.5/0.2 of the long segments (head of each document and of the examples, most-recent whole history turns), LLMLingua-2 at the same three rates, and the gold document moved to first/middle/last. Truncation and LLMLingua-2 share rates so they can be compared at equal budgets; plots use the *achieved* kept fraction.

**Noise floor.** From 30 identical repeats we report the identical-output rate, the pass flip rate, and the 95th percentile of repeat-to-repeat differences in output tokens and GPU time. Smaller effects are labelled within noise.

**Shapley.** v(S) is the mean metric over tasks with only segments S (plus the query) present. We report φ for pass rate, output tokens, GPU-seconds, energy and price-list cost, plus pairwise interaction indices and leave-one-out values.

**Tests.** Exact McNemar for accuracy, paired bootstrap (10,000 resamples over tasks) for cost deltas, Benjamini–Hochberg per workload.

**Cost models.** Physical: GPU-seconds and energy. Price-list: `input + 5 × output` tokens. A linear fit of prefill ms per input token and decode ms per output token gives the measured decode:prefill ratio.

---

## 6. Results

_Pending the Kaggle runs. Planned subsections, each tied to an `analysis/` file:_

- 6.1 Noise floor (`noise_floor.csv`)
- 6.2 Shapley values for accuracy, output tokens and GPU-seconds; Shapley versus leave-one-out (`shapley.csv`, `shapley_*.png`)
- 6.3 Interactions: retrieved × history and system × tools (`interactions.csv`, heatmaps)
- 6.4 Truncation versus LLMLingua-2 at equal budget (`comparisons.csv`, `truncation_vs_lingua.png`)
- 6.5 Lost in the middle (`reorder.png`)
- 6.6 Output expansion tails and how often reduction raised cost (`expansion.csv`, CCDF)
- 6.7 Pareto frontiers under physical and price-list cost (`pareto.csv`)
- 6.8 Measured decode:prefill ratio versus 5:1 (`cost_fit.csv`)

---

## 7. Deployment recommendations

_Written from §6._

---

## 8. Limitations

- **One model, one GPU.** Qwen2.5-7B-Instruct (AWQ 4-bit) on a T4 with fp16 and Triton attention. Llama-3.1-8B is supported but was not swept, for schedule reasons. AWQ kernels and Turing hardware shape the prefill:decode ratio.
- **Unbatched serving.** Batching amortises decode across requests in production, so absolute GPU-seconds here overstate per-request production cost. The measured trade-offs are about relative changes.
- **30 tasks.** Small samples limit power. We report effect sizes with confidence intervals and correct for multiplicity rather than claim small effects.
- **Greedy decoding.** It isolates context effects, but sampling-based deployments would show more variance.
- **Synthetic world.** The fictional world prevents memorisation but is cleaner than real enterprise data, and the shared tool catalogue fixes the share of tokens spent on tools.
- **Calibration selection.** Keeping only tasks the model gets right with full context makes the baseline accuracy 100% by design. Results describe the loss from removing context, not absolute capability.

---

## 9. Conclusion

_Written from §6._

---

## References

See the formal report, Chapter 10.
