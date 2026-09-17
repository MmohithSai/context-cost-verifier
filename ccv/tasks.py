"""Benchmark tasks: YAML loading, prompt layout from segments, and pass/fail graders."""

import hashlib
import json
import re
import string
from pathlib import Path

import yaml

# Ablatable prompt segments. The order is fixed: it is also the bit order of subset masks.
SEGMENTS = ("system", "instructions", "tools", "examples", "retrieved", "history")
GRADER_FOR = {"qa": "qa", "extraction": "json", "tool_call": "tool_call"}
EXPECTED_KEY = {"qa": "answers", "json": "expected", "tool_call": "name"}


def digest(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]


def _canon_tool(tool: dict) -> dict:
    # Same keys and order vLLM hands to the chat template (tool.model_dump()),
    # so the local render matches the server's prompt token for token.
    f = tool["function"]
    fn = {"name": f["name"], "description": f.get("description"), "parameters": f.get("parameters")}
    if f.get("strict") is not None:
        fn["strict"] = f["strict"]
    return {"type": "function", "function": fn}


def load_tasks(bench) -> list[dict]:
    """Load every <bench>/tasks/**/*.yaml; "@name" segment values come from <bench>/shared/name.yaml."""
    bench = Path(bench)
    tasks = []
    for path in sorted((bench / "tasks").rglob("*.yaml")):
        task = yaml.safe_load(path.read_text(encoding="utf-8"))
        seg = task.get("segments") if isinstance(task, dict) else None
        for key, value in (seg or {}).items():
            if isinstance(value, str) and value.startswith("@"):
                seg[key] = yaml.safe_load((bench / "shared" / f"{value[1:]}.yaml").read_text(encoding="utf-8"))
        _validate(task, path)
        seg["tools"] = [_canon_tool(t) for t in seg["tools"]]
        task["hash"] = digest({k: task.get(k) for k in ("workload", "segments", "query", "grader", "gold_doc")})
        tasks.append(task)
    ids = [t["id"] for t in tasks]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate task ids under {bench / 'tasks'}")
    return tasks


def _validate(task, path):
    def need(ok, msg):
        if not ok:
            raise ValueError(f"{path}: {msg}")

    need(isinstance(task, dict), "task file must be a mapping")
    need(isinstance(task.get("id"), str), "missing id")
    need(task.get("workload") in GRADER_FOR, f"workload must be one of {list(GRADER_FOR)}")
    seg = task.get("segments") or {}
    for key in SEGMENTS:
        need(seg.get(key), f"segment '{key}' is missing or empty")
    for key in ("system", "instructions", "examples"):
        need(isinstance(seg[key], str), f"segment '{key}' must be text")
    need(all(isinstance(d, dict) and d.get("title") and d.get("text") for d in seg["retrieved"]),
         "every retrieved doc needs a title and text")
    need(len(seg["history"]) % 2 == 0 and all(
        isinstance(m, dict) and m.get("role") == ("user", "assistant")[i % 2] and m.get("content")
        for i, m in enumerate(seg["history"])), "history must be user/assistant pairs, starting with user")
    need(all(isinstance(t, dict) and t.get("type") == "function" and (t.get("function") or {}).get("name")
             for t in seg["tools"]), "tools must be OpenAI function tools")
    need(isinstance(task.get("query"), str) and task["query"].strip(), "missing query")
    grader = task.get("grader") or {}
    need(grader.get("type") == GRADER_FOR[task["workload"]],
         f"grader type must be '{GRADER_FOR[task['workload']]}' for workload '{task['workload']}'")
    need(EXPECTED_KEY[grader["type"]] in grader, f"grader needs '{EXPECTED_KEY[grader['type']]}'")
    if "gold_doc" in task:
        need(isinstance(task["gold_doc"], int) and 0 <= task["gold_doc"] < len(seg["retrieved"]),
             "gold_doc must index into retrieved")


def layout(seg: dict, query: str):
    """Chat messages as (role, [(segment, text), ...]) plus the tools list; missing segments are skipped.

    Each piece carries its own header and leading separator, so dropping a segment
    removes exactly its characters from the prompt.
    """
    system = []
    for name, header in (("system", ""), ("instructions", "## Instructions\n"), ("examples", "## Examples\n")):
        if seg.get(name):
            system.append((name, ("\n\n" if system else "") + header + seg[name].strip()))
    msgs = [("system", system)] if system else []
    msgs += [(m["role"], [("history", m["content"].strip())]) for m in seg.get("history") or []]
    user = []
    if seg.get("retrieved"):
        docs = "\n\n".join(f"[{i}] {d['title'].strip()}\n{d['text'].strip()}"
                           for i, d in enumerate(seg["retrieved"], 1))
        user.append(("retrieved", "## Documents\n" + docs))
    user.append(("query", ("\n\n" if user else "") + "## Question\n" + query.strip()))
    msgs.append(("user", user))
    return msgs, seg.get("tools") or None


def to_messages(msgs) -> list[dict]:
    return [{"role": role, "content": "".join(text for _, text in parts)} for role, parts in msgs]


# ---- graders -------------------------------------------------------------------------------

def _norm(text) -> str:
    """SQuAD-style normalization; punctuation becomes a space so '12.5' never equals '125'."""
    text = str(text).lower().translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def _contains(text, phrases) -> bool:
    haystack = f" {_norm(text)} "
    return any(f" {_norm(p)} " in haystack for p in phrases)


def _first_json_object(text):
    decoder = json.JSONDecoder()
    for m in re.finditer(r"\{", text):
        try:
            obj, _ = decoder.raw_decode(text, m.start())
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def _match(expected, got) -> bool:
    if isinstance(got, str) and isinstance(expected, (list, dict)):
        try:
            got = json.loads(got)  # Llama 3.1 sometimes sends arrays/objects as JSON strings
        except ValueError:
            return False
    if isinstance(expected, bool):
        return str(got).lower() == str(expected).lower()
    if isinstance(expected, (int, float)):
        try:
            return abs(float(got) - expected) <= 1e-6 * max(1.0, abs(expected))
        except (TypeError, ValueError):
            return False
    if isinstance(expected, str):
        return isinstance(got, (str, int, float)) and _norm(got) == _norm(expected)
    if isinstance(expected, list):
        return isinstance(got, list) and len(got) == len(expected) and all(map(_match, expected, got))
    if isinstance(expected, dict):
        return isinstance(got, dict) and all(k in got and _match(v, got[k]) for k, v in expected.items())
    return got == expected


def grade(task: dict, output, tool_calls) -> bool:
    """Machine-checkable pass/fail for one model response."""
    g, output = task["grader"], output if isinstance(output, str) else ""
    if g["type"] == "qa":
        return _contains(output, g["answers"]) and not _contains(output, g.get("distractors", []))
    if g["type"] == "json":
        obj = _first_json_object(output)
        return (obj is not None and _match(g["expected"], obj)
                and not (g.get("strict") and set(obj) - set(g["expected"])))
    if not isinstance(tool_calls, list) or not tool_calls:
        return False  # a tool call written as plain text counts as a failure
    fn = tool_calls[0].get("function") or {}
    args = fn.get("arguments") or "{}"
    try:
        args = json.loads(args) if isinstance(args, str) else args
    except ValueError:
        return False
    return fn.get("name") == g["name"] and _match(g.get("arguments", {}), args)
