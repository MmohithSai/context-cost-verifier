"""Command-line interface: ccv serve | run | attribute | compress | analyze."""

import argparse
import subprocess
from pathlib import Path

from . import __version__
from .ablations import BASELINE, RATES, SUITES
from .attribution import MODELS, TEMPLATES


def serve_command(model, port, extra):
    """The one place server flags live: unbatched, no prefix cache, per-request timings, tool parsing."""
    m = MODELS[model]
    cmd = ["vllm", "serve", m["hf"], "--served-model-name", model, "--port", str(port),
           "--dtype", "half",  # T4 (Turing) has no bfloat16
           "--max-model-len", "8192", "--max-num-batched-tokens", "8192",  # whole prompt in one prefill step
           "--max-num-seqs", "1",  # unbatched: each request owns the GPU
           "--no-enable-prefix-caching",  # repeats must pay full prefill
           "--enable-per-request-metrics", "--seed", "0",
           "--enable-auto-tool-choice", "--tool-call-parser", m["parser"]]
    if m.get("template"):
        cmd += ["--chat-template", str(TEMPLATES / m["template"])]
    return cmd + [a for a in extra if a != "--"]


def main(argv=None):
    p = argparse.ArgumentParser(prog="ccv", description="Context Cost Verifier: measure whether shrinking "
                                "LLM context really saves cost, before you ship it.")
    p.add_argument("--version", action="version", version=f"ccv {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve", help="start vLLM for a preset model with measurement-safe flags")
    s.add_argument("--model", required=True, choices=MODELS)
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--print-only", action="store_true", help="print the vLLM command instead of running it")
    s.add_argument("extra", nargs=argparse.REMAINDER, help="extra vLLM arguments after --")

    r = sub.add_parser("run", help="run benchmark suites against an OpenAI-compatible endpoint (resumable)")
    r.add_argument("--model", required=True, help=f"served model name; presets {list(MODELS)} get exact "
                   "token attribution")
    r.add_argument("--suite", nargs="+", required=True, choices=SUITES, help="run in the given order")
    r.add_argument("--out", required=True, type=Path, help="JSONL results file (appended, resumable)")
    r.add_argument("--bench", type=Path, default=Path("benchmark"))
    r.add_argument("--endpoint", default="http://localhost:8000/v1")
    r.add_argument("--api-key")
    r.add_argument("--tokenizer", help="HF tokenizer id for a model without a preset")
    r.add_argument("--max-tokens", type=int, default=1024)
    r.add_argument("--max-minutes", type=float, help="stop starting new requests after this long")
    r.add_argument("--limit", type=int, help="run at most this many requests (smoke tests)")
    r.add_argument("--gpu-index", type=int, default=0, help="local NVML GPU to sample; -1 disables")
    r.add_argument("--dry-run", action="store_true", help="build every request and print a plan; no server")

    a = sub.add_parser("attribute", help="print the exact per-segment token breakdown of one task")
    a.add_argument("task_id")
    a.add_argument("--model", required=True)
    a.add_argument("--config", default=BASELINE, help="e.g. subset:101111, truncate:0.5, reorder:last")
    a.add_argument("--bench", type=Path, default=Path("benchmark"))
    a.add_argument("--tokenizer")

    c = sub.add_parser("compress", help="precompute LLMLingua-2 compressions into <bench>/compressed.json")
    c.add_argument("--bench", type=Path, default=Path("benchmark"))
    c.add_argument("--rates", nargs="+", default=list(RATES))
    c.add_argument("--device", help="cuda or cpu (default: cuda when available)")

    z = sub.add_parser("analyze", help="grade runs, compute statistics, write tables and figures")
    z.add_argument("results", nargs="+", type=Path)
    z.add_argument("--bench", type=Path, default=Path("benchmark"))
    z.add_argument("--out", type=Path, default=Path("analysis"))

    args = p.parse_args(argv)

    if args.cmd == "serve":
        cmd = serve_command(args.model, args.port, args.extra)
        print(" ".join(cmd), flush=True)
        if not args.print_only:
            try:
                raise SystemExit(subprocess.call(cmd))
            except FileNotFoundError:
                raise SystemExit("vllm is not installed here (it needs Linux + an NVIDIA GPU; see the notebook)")
        return

    from .tasks import load_tasks

    tasks = load_tasks(args.bench)
    if args.cmd == "run":
        from .runner import run

        run(tasks, args.model, args.suite, args.out, bench=args.bench, endpoint=args.endpoint,
            api_key=args.api_key, tokenizer=args.tokenizer, max_tokens=args.max_tokens, limit=args.limit,
            max_minutes=args.max_minutes, gpu_index=args.gpu_index, dry_run=args.dry_run)
    elif args.cmd == "attribute":
        from .ablations import load_cache, variant
        from .attribution import Prompter
        from .tasks import layout

        task = next((t for t in tasks if t["id"] == args.task_id), None)
        if task is None:
            raise SystemExit(f"no task '{args.task_id}' in {args.bench}")
        prompter = Prompter.for_model(args.model, args.tokenizer)
        if prompter is None:
            raise SystemExit("attribution needs a preset model or --tokenizer")
        seg, _ = variant(task, args.config, prompter.tok, load_cache(args.bench))
        counts = prompter.attribute(*layout(seg, task["query"]))
        total = counts.pop("total")
        print(f"{task['id']} ({task['workload']}) under {args.config} on {args.model}")
        for name, n in counts.items():
            print(f"  {name:<13}{n:>6}  {n / total:6.1%}")
        print(f"  {'total':<13}{total:>6}")
    elif args.cmd == "compress":
        from .ablations import compress_all

        compress_all(tasks, args.bench, args.rates, args.device)
    elif args.cmd == "analyze":
        from .analysis import analyze
        from .report import plot_all

        tables = analyze(args.results, tasks, args.out)
        plot_all(tables, args.out)


if __name__ == "__main__":
    main()
