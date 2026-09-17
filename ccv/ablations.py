"""Context variants: segment subsets, truncation, LLMLingua-2 compression and gold-doc reordering."""

import copy
import json
import time
from pathlib import Path

from .tasks import SEGMENTS, digest

BASELINE = "subset:111111"
RATES = ("0.8", "0.5", "0.2")  # strings so cache keys never depend on float formatting
LINGUA_MODEL = "microsoft/llmlingua-2-xlm-roberta-large-meetingbank"
SUITES = {
    "calibrate": [(BASELINE, 3), ("subset:000000", 1)],
    "noise": [(BASELINE, 30)],
    "shapley": [(f"subset:{m:06b}", 1) for m in range(64)],
    "ablations": [(f"{kind}:{r}", 1) for kind in ("truncate", "lingua2") for r in RATES]
    + [(f"reorder:{pos}", 1) for pos in ("first", "middle", "last")],
}


def variant(task, config_id, tok=None, cache=None):
    """Segments for one config, plus compression cost info (LLMLingua-2 configs only)."""
    kind, arg = config_id.split(":")
    seg = copy.deepcopy(task["segments"])
    if kind == "subset":
        return {k: seg[k] for k, bit in zip(SEGMENTS, arg) if bit == "1"}, None
    if kind == "reorder":
        docs = seg["retrieved"]
        gold = docs.pop(task["gold_doc"])
        docs.insert({"first": 0, "middle": len(docs) // 2, "last": len(docs)}[arg], gold)
        return seg, None
    if kind == "truncate":
        if tok is None:
            raise ValueError("truncation is measured in model tokens: pass a preset model or --tokenizer")
        f = float(arg)
        seg["examples"] = _head(tok, seg["examples"], f)
        for doc in seg["retrieved"]:
            doc["text"] = _head(tok, doc["text"], f)
        seg["history"] = _recent(tok, seg["history"], f)
        return seg, None
    if kind == "lingua2":
        spent = 0.0

        def squeeze(text):
            nonlocal spent
            key = cache_key(text, arg)
            if key not in (cache or {}):
                raise KeyError(f"no LLMLingua-2 result for task {task['id']} at rate {arg}; run `ccv compress` first")
            spent += cache[key]["ms"]
            return cache[key]["text"]

        seg["examples"] = squeeze(seg["examples"])
        for doc in seg["retrieved"]:
            doc["text"] = squeeze(doc["text"])
        for msg in seg["history"]:
            msg["content"] = squeeze(msg["content"])
        return seg, {"method": "llmlingua-2", "rate": float(arg), "compress_ms": round(spent, 1)}
    raise ValueError(f"unknown config '{config_id}'")


def _head(tok, text, f):
    ids = tok(text, add_special_tokens=False)["input_ids"]
    return tok.decode(ids[: int(len(ids) * f)], clean_up_tokenization_spaces=False)


def _recent(tok, turns, f):
    """Drop the oldest user/assistant pairs until the history fits in f of its tokens (like chat windowing)."""
    size = lambda ts: sum(len(tok(m["content"], add_special_tokens=False)["input_ids"]) for m in ts)
    budget = f * size(turns)
    while turns and size(turns) > budget:
        turns = turns[2:]
    return turns


def long_texts(seg):
    """The texts that truncation and compression act on."""
    yield seg["examples"]
    yield from (d["text"] for d in seg["retrieved"])
    yield from (m["content"] for m in seg["history"])


def cache_key(text, rate):
    return digest([LINGUA_MODEL, str(rate), text])


def load_cache(bench):
    path = Path(bench) / "compressed.json" if bench else None
    return json.loads(path.read_text(encoding="utf-8")) if path and path.exists() else {}


def compress_all(tasks, bench, rates=RATES, device=None):
    """Precompute LLMLingua-2 compressions once, so GPU runs never need llmlingua installed."""
    import torch
    from llmlingua import PromptCompressor

    path = Path(bench) / "compressed.json"
    cache = load_cache(bench)
    todo = sorted({(text, r) for t in tasks for text in long_texts(t["segments"]) for r in rates})
    todo = [(text, r) for text, r in todo if cache_key(text, r) not in cache]
    if not todo:
        print("compression cache is up to date")
        return cache
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    compressor = PromptCompressor(model_name=LINGUA_MODEL, use_llmlingua2=True, device_map=device)
    compressor.compress_prompt("Warm-up text so the first timed call is not slower.", rate=0.5)
    for i, (text, r) in enumerate(todo, 1):
        t0 = time.perf_counter()
        out = compressor.compress_prompt(text, rate=float(r), force_tokens=["\n", "?"])["compressed_prompt"]
        cache[cache_key(text, r)] = {"text": out, "ms": round((time.perf_counter() - t0) * 1000, 1)}
        if i % 25 == 0 or i == len(todo):
            path.write_text(json.dumps(cache, indent=1, sort_keys=True, ensure_ascii=False), encoding="utf-8")
            print(f"compressed {i}/{len(todo)} on {device}")
    return cache
