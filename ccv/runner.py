"""Resumable, unbatched runner for any OpenAI-compatible endpoint, with physical GPU measurements."""

import json
import os
import random
import socket
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

from . import __version__
from .ablations import SUITES, load_cache, variant
from .attribution import MODELS, Prompter
from .tasks import digest, layout, to_messages

SEC_PER_REQUEST = 6  # rough T4 estimate for dry-run planning; re-measure with a smoke run


class RequestError(Exception):
    def __init__(self, status, text):
        super().__init__(f"HTTP {status}: {text}")
        self.status = status


def read_records(path):
    """Yield records from a JSONL file, skipping a partial line left by a killed session."""
    path = Path(path)
    if not path.exists():
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _terminate_partial_line(path):
    if path.exists() and path.stat().st_size:
        with open(path, "rb+") as f:
            f.seek(-1, os.SEEK_END)
            if f.read(1) != b"\n":
                f.write(b"\n")


def _append(path, record):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def plan(tasks, suites, seed=0):
    """(suite, task, config_id, repeat) in suite order; shuffled within a suite to spread drift."""
    runs = []
    for suite in suites:
        batch = [(suite, t, cfg, rep) for t in tasks for cfg, n in SUITES[suite] for rep in range(n)
                 if not cfg.startswith("reorder") or "gold_doc" in t]
        random.Random(seed).shuffle(batch)
        runs += batch
    return runs


class Gpu:
    """Samples one local GPU through NVML while a request runs."""

    def __init__(self, index):
        import pynvml  # provided by nvidia-ml-py

        pynvml.nvmlInit()
        self.nv, self.handle = pynvml, pynvml.nvmlDeviceGetHandleByIndex(index)
        name = pynvml.nvmlDeviceGetName(self.handle)
        self.name = name.decode() if isinstance(name, bytes) else name

    def _energy_mj(self):
        try:
            return self.nv.nvmlDeviceGetTotalEnergyConsumption(self.handle)
        except self.nv.NVMLError:
            return None

    def __enter__(self):
        self.util, self.mem, self.stop = [], 0, threading.Event()
        self.e0 = self._energy_mj()
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.thread.start()
        return self

    def _sample(self):
        while not self.stop.wait(0.1):
            self.util.append(self.nv.nvmlDeviceGetUtilizationRates(self.handle).gpu)
            self.mem = max(self.mem, self.nv.nvmlDeviceGetMemoryInfo(self.handle).used)

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join()
        e1 = self._energy_mj()
        self.stats = {
            "gpu_util_mean": round(sum(self.util) / len(self.util), 1) if self.util else None,
            "gpu_mem_max_mb": round(self.mem / 2**20) if self.mem else None,
            "energy_j": round((e1 - self.e0) / 1000, 3) if None not in (e1, self.e0) else None,
        }


class NoGpu:
    name = None
    stats = {"gpu_util_mean": None, "gpu_mem_max_mb": None, "energy_j": None}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass


def stream_chat(client, url, body, headers):
    """One streamed chat completion: output, tool calls, usage, server metrics and client timings."""
    t0 = time.perf_counter()
    first = finish = None
    text, calls, usage, metrics = [], {}, {}, {}
    with client.stream("POST", url, json=body, headers=headers) as r:
        if r.status_code >= 400:
            r.read()
            raise RequestError(r.status_code, r.text[:500])
        for line in r.iter_lines():
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            usage = chunk.get("usage") or usage
            metrics = chunk.get("metrics") or metrics  # vLLM --enable-per-request-metrics
            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") or {}
                if first is None and (delta.get("content") or delta.get("tool_calls")):
                    first = time.perf_counter()
                text.append(delta.get("content") or "")
                for tc in delta.get("tool_calls") or []:
                    call = calls.setdefault(tc.get("index", 0), {"name": "", "arguments": ""})
                    fn = tc.get("function") or {}
                    call["name"] += fn.get("name") or ""
                    call["arguments"] += fn.get("arguments") or ""
                finish = choice.get("finish_reason") or finish
    end = time.perf_counter()
    return {
        "output": "".join(text),
        "tool_calls": [{"type": "function", "function": calls[i]} for i in sorted(calls)] or None,
        "usage": usage, "metrics": metrics, "finish_reason": finish,
        "client_ttft_ms": round((first - t0) * 1000, 2) if first else None,
        "client_e2e_ms": round((end - t0) * 1000, 2),
    }


def _pick(server, client):
    return server if server is not None else client


def measure(res, gpu_stats):
    """Physical accounting for one request: server timings when available, client timings otherwise."""
    usage, m, ttft = res["usage"], res["metrics"], res["client_ttft_ms"]
    prefill = _pick(m.get("time_to_first_token_ms"), ttft)
    decode = _pick(m.get("generation_time_ms"), None if ttft is None else res["client_e2e_ms"] - ttft)
    n_out = usage.get("completion_tokens")
    return {
        "prefill_tokens": usage.get("prompt_tokens"),
        "decode_tokens": n_out,
        "finish_reason": res["finish_reason"],
        "prefill_ms": prefill,
        "decode_ms": decode,
        "queue_ms": m.get("queue_time_ms"),
        "client_ttft_ms": ttft,
        "client_e2e_ms": res["client_e2e_ms"],
        "decode_tok_s": round((n_out - 1) / (decode / 1000), 2) if n_out and n_out > 1 and decode else None,
        # Unbatched serving: the request owns the GPU from first scheduling to its last token.
        "gpu_seconds": round((prefill + decode) / 1000, 4) if None not in (prefill, decode) else None,
        "server_metrics": bool(m),
        **gpu_stats,
        "output": res["output"],
        "tool_calls": res["tool_calls"],
    }


def _server_version(client, endpoint):
    try:
        return client.get(endpoint.rstrip("/").removesuffix("/v1") + "/version").json().get("version")
    except (httpx.HTTPError, ValueError, AttributeError):
        return None


def _check_model(client, endpoint, model, headers):
    try:
        served = [m["id"] for m in client.get(f"{endpoint}/models", headers=headers).json()["data"]]
    except (httpx.HTTPError, ValueError, KeyError, TypeError):
        print("warning: could not list served models; continuing")
        return
    if model not in served:
        raise SystemExit(f"endpoint does not serve '{model}' (it serves: {', '.join(served)})")


def run(tasks, model, suites, out, bench=None, endpoint="http://localhost:8000/v1", api_key=None,
        tokenizer=None, max_tokens=1024, limit=None, max_minutes=None, gpu_index=0, dry_run=False):
    out, endpoint = Path(out), endpoint.rstrip("/")
    preset = MODELS.get(model, {})
    prompter = Prompter.for_model(model, tokenizer)
    cache = load_cache(bench)
    sampling = {"temperature": 0, "seed": 0, "max_tokens": max_tokens}

    built = {}

    def build(task, cfg):
        """Request body, compression info and exact attribution; identical for every repeat."""
        key = (task["id"], cfg)
        if key not in built:
            seg, compression = variant(task, cfg, prompter and prompter.tok, cache)
            msgs, tools = layout(seg, task["query"])
            body = {"model": model, "messages": to_messages(msgs), **sampling}
            if tools:
                body["tools"] = tools
            if preset.get("template_kwargs"):
                body["chat_template_kwargs"] = preset["template_kwargs"]
            built[key] = (body, compression, prompter.attribute(msgs, tools) if prompter else None)
        return built[key]

    done = set() if dry_run else {r["run_id"] for r in read_records(out) if r.get("status") == "ok"}
    todo = []
    for suite, task, cfg, rep in plan(tasks, suites):
        # suite is part of the id: the noise and shapley suites both measure the baseline, independently
        run_id = digest([model, suite, task["id"], task["hash"], cfg, rep, sampling])
        if run_id not in done:
            todo.append((run_id, suite, task, cfg, rep))
    if limit:
        todo = todo[:limit]
    print(f"{len(todo)} runs to do ({len(done)} already recorded in {out})")

    if dry_run:
        return _dry_run(todo, build)

    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    gpu = NoGpu()
    if gpu_index >= 0 and httpx.URL(endpoint).host in ("localhost", "127.0.0.1", "0.0.0.0"):
        try:
            gpu = Gpu(gpu_index)
        except Exception as e:  # no NVML, no driver, or nvidia-ml-py not installed
            print(f"GPU sampling disabled: {e}")
    deadline = time.monotonic() + max_minutes * 60 if max_minutes else None
    url = f"{endpoint}/chat/completions"
    stream = {"stream": True, "stream_options": {"include_usage": True}}

    with httpx.Client(timeout=httpx.Timeout(600, connect=10)) as client:
        if not todo:
            return
        _check_model(client, endpoint, model, headers)
        env = {"ccv": __version__, "server": _server_version(client, endpoint), "gpu": gpu.name,
               "host": socket.gethostname(), "sampling": sampling}
        warm = {**build(todo[0][2], todo[0][3])[0], "max_tokens": 32, **stream}
        for _ in range(3):  # warm-up, not recorded
            try:
                stream_chat(client, url, warm, headers)
            except (httpx.HTTPError, RequestError, ValueError):
                pass
        out.parent.mkdir(parents=True, exist_ok=True)
        _terminate_partial_line(out)
        for i, (run_id, suite, task, cfg, rep) in enumerate(todo, 1):
            if deadline and time.monotonic() > deadline:
                print(f"time budget reached after {i - 1} runs; run the same command again to resume")
                break
            rec = {"run_id": run_id, "model": model, "suite": suite, "task_id": task["id"],
                   "task_hash": task["hash"], "workload": task["workload"],
                   "config_id": cfg, "repeat": rep}
            try:
                body, compression, attribution = build(task, cfg)
            except KeyError as e:
                # A stale compression cache must not cost the rest of the session; resume retries it.
                rec.update(status="error", error=f"KeyError: {e}", request_hash=None, env=env,
                           ts=datetime.now(timezone.utc).isoformat(timespec="seconds"))
                _append(out, rec)
                print(f"[{i}/{len(todo)}] {suite} {task['id']} {cfg} r{rep} error {e}", flush=True)
                continue
            rec["request_hash"] = digest(body)
            for attempt in range(3):
                try:
                    with gpu:
                        res = stream_chat(client, url, {**body, **stream}, headers)
                    rec.update(status="ok", error=None, **measure(res, gpu.stats))
                    break
                except (httpx.HTTPError, RequestError, ValueError) as e:
                    rec.update(status="error", error=f"{type(e).__name__}: {e}")
                    if isinstance(e, RequestError) and e.status < 500:
                        break  # the request itself is bad; retrying will not help
                    if attempt < 2:
                        time.sleep(5 * 2**attempt)
            ok = rec["status"] == "ok"
            rec.update(attribution=attribution,
                       attribution_ok=attribution["total"] == rec["prefill_tokens"] if ok and attribution else None,
                       compression=compression, env=env, ts=datetime.now(timezone.utc).isoformat(timespec="seconds"))
            _append(out, rec)
            detail = (f"in={rec['prefill_tokens']} out={rec['decode_tokens']} prefill={rec['prefill_ms']}ms "
                      f"decode={rec['decode_ms']}ms" if ok else rec["error"])
            print(f"[{i}/{len(todo)}] {suite} {task['id']} {cfg} r{rep} {rec['status']} {detail}", flush=True)


def _dry_run(todo, build):
    stats = {}
    for _, suite, task, cfg, _ in todo:
        s = stats.setdefault(suite, {"runs": 0, "tokens": [], "uncompressed": 0})
        s["runs"] += 1
        try:
            _, _, attribution = build(task, cfg)
        except KeyError:  # no LLMLingua-2 cache entry yet; keep validating the other tasks
            s["uncompressed"] += 1
            continue
        if attribution:
            s["tokens"].append(attribution["total"])
    for suite, s in stats.items():
        t = s["tokens"]
        size = f"prompt tokens mean={sum(t) / len(t):.0f} max={max(t)}" if t else "no local tokenizer"
        missing = f", {s['uncompressed']} awaiting `ccv compress`" if s["uncompressed"] else ""
        print(f"{suite:>10}: {s['runs']:5d} runs, {size}{missing}")
    total = sum(s["runs"] for s in stats.values())
    print(f"total {total} runs ~ {total * SEC_PER_REQUEST / 3600:.1f} GPU-hours at ~{SEC_PER_REQUEST}s/request")
    return stats
