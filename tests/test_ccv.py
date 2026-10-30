"""ccv test suite: math, graders, variants, exact attribution, and a GPU-free end-to-end pipeline run.

The end-to-end test starts a fake vLLM-style server that renders prompts with the real tokenizer
(so attribution must match its token counts exactly) and behaves like a model that needs both the
answer-bearing document and a fact from the chat history, and rambles when instructions are removed.
"""

import json
import shutil
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest

from ccv import analysis
from ccv.ablations import SUITES, cache_key, long_texts, variant
from ccv.attribution import MODELS, Prompter
from ccv.cli import serve_command
from ccv.report import plot_all
from ccv.runner import read_records, run
from ccv.tasks import SEGMENTS, grade, layout, load_tasks

BENCH = Path(__file__).resolve().parents[1] / "benchmark"
SMALL = "qwen2.5-0.5b"


def _prompter(model):
    try:
        return Prompter.for_model(model)
    except OSError as e:  # offline and not cached
        pytest.skip(f"tokenizer for {model} unavailable: {e}")


# ---- math ----------------------------------------------------------------------------------

def _game(fn):
    return {m: fn({i for i in range(analysis.N) if m & analysis._bit(i)}) for m in range(64)}


def test_shapley_efficiency_and_symmetry():
    rng = np.random.default_rng(1)
    v = {m: rng.normal() for m in range(64)}
    phi = analysis.shapley(v)
    assert np.isclose(phi.sum(), v[63] - v[0])
    sym = _game(lambda s: len(s) ** 2)  # every player is interchangeable
    assert np.allclose(analysis.shapley(sym), analysis.shapley(sym)[0])


def test_shapley_pure_pair_and_additive_games():
    pair = _game(lambda s: float({0, 1} <= s))
    assert np.allclose(analysis.shapley(pair), [0.5, 0.5, 0, 0, 0, 0])
    inter = analysis.interactions(pair)
    assert np.isclose(inter[0, 1], 1.0) and np.isclose(np.abs(inter).sum(), 2.0)
    weights = [3, 1, 4, 1, 5, 9]
    additive = _game(lambda s: sum(weights[i] for i in s))
    assert np.allclose(analysis.shapley(additive), weights)
    assert np.allclose(analysis.interactions(additive), 0)
    assert np.allclose(analysis.leave_one_out(additive), weights)


def test_statistics():
    assert np.isclose(analysis.mcnemar([1] * 10, [0] * 10), 2 * 0.5**10)
    assert analysis.mcnemar([1, 0, 1], [1, 0, 1]) == 1.0
    mean, lo, hi, p = analysis.paired_bootstrap([0.0] * 10)
    assert (mean, lo, hi, p) == (0.0, 0.0, 0.0, 1.0)
    mean, lo, hi, p = analysis.paired_bootstrap([5, 6, 4, 5, 7, 5, 6, 4, 5, 6])
    assert lo <= mean <= hi and lo > 0 and p < 0.01
    assert sorted(analysis.pareto([1, 2, 3, 1], [0.5, 0.4, 0.9, 0.6])) == [2, 3]


# ---- graders & variants --------------------------------------------------------------------

@pytest.fixture(scope="module")
def tasks():
    return {t["id"]: t for t in load_tasks(BENCH)}


def test_graders(tasks):
    qa, ex, tc = tasks["qa-01"], tasks["ex-01"], tasks["tc-01"]
    assert grade(qa, "12.5 kg", None)
    assert grade(qa, "The K4 Hauler lifts 12.5 kilograms.", None)
    assert not grade(qa, "12 kg", None)
    assert not grade(qa, "112.5 kg", None)
    assert not grade(qa, "Either 12.5 kg or 9 kg.", None)  # hedging between candidates fails
    good = {"invoice_id": "INV-20931", "order_id": "KR-55120", "total": "1840.50", "currency": "eur",
            "due_date": "2026-03-14"}
    assert grade(ex, "Sure:\n```json\n" + json.dumps(good) + "\n```", None)
    assert not grade(ex, json.dumps({**good, "customer": "Lindqvist"}), None)  # strict: no extra keys
    assert not grade(ex, json.dumps({**good, "due_date": "2026-03-07"}), None)
    assert not grade(ex, "I could not find that invoice.", None)
    args = {"serial_number": "KR4-90533", "center_id": "SC-PNQ-02", "date": "2026-04-02"}
    call = lambda name, a: [{"type": "function", "function": {"name": name, "arguments": a}}]
    assert grade(tc, "", call("schedule_maintenance", json.dumps(args)))
    assert grade(tc, "", call("schedule_maintenance", args))
    assert not grade(tc, "", call("create_support_ticket", json.dumps(args)))
    assert not grade(tc, "", call("schedule_maintenance", json.dumps({**args, "date": "2026-03-31"})))
    assert not grade(tc, json.dumps({"name": "schedule_maintenance", "arguments": args}), None)
    list_tool = {"type": "tool_call", "name": "tag", "arguments": {"labels": ["a", "b"]}}
    assert grade({"grader": list_tool}, "", call("tag", json.dumps({"labels": '["a", "b"]'})))


def test_every_task_grades_its_own_gold_answer(tasks):
    """A task whose grader rejects its own answer would only fail after burning GPU time."""
    for task in tasks.values():
        g = task["grader"]
        if g["type"] == "qa":
            assert grade(task, g["answers"][0], None), task["id"]
            for bad in g.get("distractors", []):
                assert not grade(task, bad, None), (task["id"], bad)
        elif g["type"] == "json":
            assert grade(task, json.dumps(g["expected"]), None), task["id"]
            assert not grade(task, "I could not find it.", None), task["id"]
        else:
            call = [{"type": "function", "function": {"name": g["name"],
                                                      "arguments": json.dumps(g.get("arguments", {}))}}]
            assert grade(task, "", call), task["id"]
            assert not grade(task, json.dumps(g.get("arguments", {})), None), task["id"]


def test_variants(tasks):
    qa = tasks["qa-01"]
    assert len({c for c, _ in SUITES["shapley"]}) == 64
    seg, _ = variant(qa, "subset:100001")
    assert set(seg) == {"system", "history"}
    gold = qa["segments"]["retrieved"][qa["gold_doc"]]
    for pos, idx in (("first", 0), ("middle", 2), ("last", 4)):
        docs = variant(qa, f"reorder:{pos}")[0]["retrieved"]
        assert docs[idx] == gold and len(docs) == 5
    with pytest.raises(KeyError, match="ccv compress"):
        variant(qa, "lingua2:0.5", cache={})
    cache = {cache_key(t, "0.5"): {"text": t[: len(t) // 2], "ms": 1.0} for t in long_texts(qa["segments"])}
    seg, info = variant(qa, "lingua2:0.5", cache=cache)
    assert info["compress_ms"] == len(list(long_texts(qa["segments"])))
    assert seg["examples"] == qa["segments"]["examples"][: len(qa["segments"]["examples"]) // 2]
    tok = _prompter(SMALL).tok
    size = lambda s: len(tok(s, add_special_tokens=False)["input_ids"])
    for f in ("0.8", "0.5", "0.2"):
        seg, _ = variant(qa, f"truncate:{f}", tok)
        for doc, full in zip(seg["retrieved"], qa["segments"]["retrieved"]):
            assert size(doc["text"]) <= float(f) * size(full["text"]) + 1
        assert len(seg["history"]) % 2 == 0 and len(seg["history"]) < len(qa["segments"]["history"])
    assert qa["segments"]["retrieved"][0]["text"].startswith("The K3 Sparrow")  # original untouched


def test_variants_need_tokenizer_for_truncation(tasks):
    with pytest.raises(ValueError, match="tokenizer"):
        variant(tasks["qa-01"], "truncate:0.5")


def test_serve_command():
    cmd = serve_command("llama3.1-8b", 8000, ["--", "--attention-backend", "TRITON_ATTN"])
    for flag in ("--no-enable-prefix-caching", "--enable-per-request-metrics", "--dtype", "--chat-template"):
        assert flag in cmd
    assert cmd[cmd.index("--max-num-seqs") + 1] == "1"
    assert cmd[cmd.index("--tool-call-parser") + 1] == "llama3_json"
    assert cmd[-2:] == ["--attention-backend", "TRITON_ATTN"]


# ---- exact attribution ---------------------------------------------------------------------

@pytest.mark.parametrize("model", [SMALL, "llama3.1-8b"])
def test_attribution_is_exact(tasks, model):
    p = _prompter(model)
    for task in tasks.values():
        # Every task is checked at full context; the mask sweep is a template property, so one
        # task per workload covers it without paying 45x the render cost.
        sweep = task["id"] in ("qa-01", "ex-01", "tc-01")
        for mask in ("111111", "000000", "101010", "010101", "110111", "111110")[: 6 if sweep else 1]:
            seg, _ = variant(task, f"subset:{mask}")
            msgs, tools = layout(seg, task["query"])
            counts = p.attribute(msgs, tools)
            total = counts.pop("total")
            assert sum(counts.values()) == total
            assert total == p.count(p.render([{"role": r, "content": "".join(t for _, t in parts)}
                                              for r, parts in msgs], tools))
            for name, bit in zip(SEGMENTS, mask):
                assert (counts[name] > 0) == (bit == "1"), (task["id"], mask, name)
            assert counts["query"] > 0


# ---- end to end without a GPU --------------------------------------------------------------

FACTS = {  # query marker -> (fact only the gold doc has, fact only the history has, correct answer)
    "maximum payload": ("Maximum payload: 12.5 kg", "KR4-88305", "12.5 kg"),
    "invoice details": ("Total: EUR 1,840.50", "KR-55120", json.dumps(
        {"invoice_id": "INV-20931", "order_id": "KR-55120", "total": 1840.5, "currency": "EUR",
         "due_date": "2026-03-14"})),
    "Book the inspection": ("SC-PNQ-02, Hinjewadi", "KR4-90533", None),
}


def _behave(body, prompt):
    msgs = body["messages"]
    doc_fact, hist_fact, answer = next(v for k, v in FACTS.items() if k in msgs[-1]["content"])
    knows = doc_fact in prompt and any(hist_fact in m["content"] for m in msgs[:-1])
    ramble = "" if "## Instructions" in prompt else " Here is some extra detail you did not ask for." * 12
    if answer is not None:
        return (answer if knows else "unknown") + ramble, None
    if "tools" in body and knows:
        return "", {"name": "schedule_maintenance",
                    "arguments": {"serial_number": "KR4-90533", "center_id": "SC-PNQ-02", "date": "2026-04-02"}}
    return "Which service center should I use?" + ramble, None


class FakeVLLM(BaseHTTPRequestHandler):
    prompter = None

    def log_message(self, *args):
        pass

    def _json(self, obj):
        data = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self._json({"data": [{"id": SMALL}]} if self.path == "/v1/models" else {"version": "fake-vllm"})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        prompt = self.prompter.render(body["messages"], body.get("tools"))
        n_in = self.prompter.count(prompt)
        text, call = _behave(body, prompt)
        if not body.get("stream"):
            tool_calls = call and [{"id": "call_0", "type": "function", "function": {
                "name": call["name"], "arguments": json.dumps(call["arguments"])}}]
            return self._json({"choices": [{"message": {"content": None if call else text,
                                                        "tool_calls": tool_calls}}]})
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        send = lambda obj: self.wfile.write(f"data: {json.dumps(obj)}\n\n".encode())
        if call:
            args = json.dumps(call["arguments"])
            send({"choices": [{"index": 0, "delta": {"tool_calls": [
                {"index": 0, "id": "call_0", "type": "function", "function": {"name": call["name"], "arguments": ""}}]}}]})
            for i in range(0, len(args), 9):
                send({"choices": [{"index": 0, "delta": {"tool_calls": [
                    {"index": 0, "function": {"arguments": args[i:i + 9]}}]}}]})
            n_out = len(args) // 3
        else:
            words = text.split(" ")
            for i, w in enumerate(words):
                if text.startswith("{"):  # like vLLM's hermes parser, swallow it when streaming
                    break
                send({"choices": [{"index": 0, "delta": {"content": w if i == 0 else " " + w}}]})
            n_out = len(words)
        n_out = min(n_out, body["max_tokens"])
        send({"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls" if call else "stop"}]})
        send({"choices": [], "usage": {"prompt_tokens": n_in, "completion_tokens": n_out},
              "metrics": {"time_to_first_token_ms": 0.2 * n_in + 5, "generation_time_ms": 25.0 * (n_out - 1),
                          "queue_time_ms": 0.1}})
        self.wfile.write(b"data: [DONE]\n\n")


@pytest.fixture
def fake_server():
    FakeVLLM.prompter = _prompter(SMALL)
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeVLLM)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}/v1"
    server.shutdown()


def test_pipeline_end_to_end(tmp_path, fake_server):
    bench = tmp_path / "benchmark"
    shutil.copytree(BENCH, bench, ignore=shutil.ignore_patterns("compressed.json"))
    tasks = load_tasks(bench)
    # What this test proves — resume, durability, Shapley identities — does not depend on how many
    # tasks the benchmark holds, but the cost does: all 45 would mean 4,635 fake requests and a
    # 64-subset sweep per task. Every task is covered by the attribution and grader tests instead.
    tasks = [t for t in tasks if t["id"] in ("qa-01", "ex-01", "tc-01")]
    cache = {}
    for t in tasks:  # stand-in for LLMLingua-2: keep the first share of words
        for text in long_texts(t["segments"]):
            for r in ("0.8", "0.5", "0.2"):
                words = text.split()
                cache[cache_key(text, r)] = {"text": " ".join(words[: int(len(words) * float(r))]), "ms": 12.0}
    (bench / "compressed.json").write_text(json.dumps(cache), encoding="utf-8")

    out = tmp_path / "results" / "fake.jsonl"
    suites = ["calibrate", "noise", "shapley", "ablations"]
    kwargs = dict(bench=bench, endpoint=fake_server, gpu_index=-1)
    run(tasks, SMALL, suites, out, limit=7, **kwargs)
    with open(out, "a", encoding="utf-8") as f:
        f.write('{"run_id": "half-written')  # a session killed mid-write
    run(tasks, SMALL, suites, out, **kwargs)

    records = list(read_records(out))
    expected = len(tasks) * (4 + 30 + 64 + 9)
    assert len(records) == expected
    assert len({r["run_id"] for r in records}) == expected
    assert all(r["status"] == "ok" and r["attribution_ok"] and r["server_metrics"] for r in records)
    assert all(r["env"]["server"] == "fake-vllm" for r in records)
    assert any(r.get("output_recovered") is True and r["output"].startswith("{") for r in records)
    run(tasks, SMALL, suites, out, **kwargs)  # nothing left to do
    assert len(list(read_records(out))) == expected

    tables = analysis.analyze([out], tasks, tmp_path / "analysis")
    cal = tables["calibration"]
    assert len(cal) == len(tasks) and cal.keep.all() and (cal.full_runs == 3).all()
    assert len(tables["runs"].query("suite == 'calibrate'")) == 4 * len(tasks)
    shap = tables["shapley"].set_index(["workload", "metric", "segment"]).shapley
    for wl in ("qa", "extraction"):
        assert np.isclose(shap[wl, "passed", "retrieved"], 0.5) and np.isclose(shap[wl, "passed", "history"], 0.5)
    assert np.isclose(shap["tool_call", "passed", "tools"], 1 / 3)
    assert shap["all", "decode_tokens", "instructions"] < 0  # instructions keep answers short
    assert np.isclose(shap["qa", "passed", "system"], 0)
    fit = tables["cost_fit"].iloc[0]
    assert np.isclose(fit.prefill_ms_per_token, 0.2) and np.isclose(fit.decode_ms_per_token, 25.0)
    cmp = tables["comparisons"].set_index(["workload", "config_id"])
    assert cmp.loc[("all", "truncate:0.2"), "acc"] < cmp.loc[("all", "truncate:0.2"), "acc_base"]
    assert cmp.loc[("all", "subset:011111"), "acc"] == 1.0  # dropping the persona costs nothing here
    assert (tables["noise_floor"].identical_output_rate == 1).all()
    assert tables["pareto"].frontier_gpu_seconds.any()
    assert (tmp_path / "analysis" / "summary.md").read_text(encoding="utf-8").startswith("# ccv analysis")

    figures = plot_all(tables, tmp_path / "analysis")
    assert len(figures) >= 9 and all(p.stat().st_size > 10_000 for p in figures)


def test_presets_have_templates_on_disk():
    from ccv.attribution import TEMPLATES

    for preset in MODELS.values():
        if preset.get("template"):
            assert (TEMPLATES / preset["template"]).exists()
